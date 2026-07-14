package ai.frozo.edgegate.benchmark

import android.content.Context
import android.os.Debug
import android.util.Log
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.bean.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale

/**
 * GenieX LLM benchmark — runs models from the GenieX catalog on NPU/GPU/CPU
 * and reports performance metrics back to EdgeGate.
 *
 * Unlike the llama.cpp path (which needs a GGUF file), GenieX downloads
 * and caches models automatically. Users just pick a model name.
 */
class GenieXBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "GenieXBenchmark"

        /** Available compute units. */
        val COMPUTE_UNITS = listOf("npu", "gpu", "cpu", "hybrid")

        /**
         * Models the user can pick from for the Snapdragon (GenieX) path.
         *
         * The GenieX SDK has no "list available models" API, so this list is curated
         * here. Per Qualcomm's GenieX docs, each `modelId` is NOT a short alias — it is
         * one of two resolvable forms passed to pullFlow (HubSource.AUTO):
         *   • "ai-hub-models/<Name>"  → Qualcomm AI Hub pre-compiled bundle, runs on the
         *                                Hexagon NPU (best perf). Confirmed: ai-hub-models/Qwen3-4B.
         *   • "<org>/<repo>-GGUF"     → a HuggingFace GGUF repo, run via llama.cpp on
         *                                CPU/GPU/NPU. Use a Q4_0 quant for NPU support.
         * If a download fails with "model not found", that exact id/repo isn't published
         * on the hub — check https://aihub.qualcomm.com/models and huggingface.co.
         */
        val MODEL_CATALOG = listOf(
            // — GGUF via llama.cpp — these download & run on ANY Snapdragon (default) —
            // — Qwen GGUF (official Qwen HF repos) —
            ModelEntry("Qwen3-0.6B", "Qwen/Qwen3-0.6B-GGUF", "0.6B params, fastest, good for testing"),
            ModelEntry("Qwen3-1.7B", "Qwen/Qwen3-1.7B-GGUF", "1.7B params, balanced speed/quality"),
            ModelEntry("Qwen3-4B (GGUF)", "Qwen/Qwen3-4B-GGUF", "4B params, higher quality"),
            // — Llama GGUF (unsloth HF repos) —
            ModelEntry("Llama-3.2-1B", "unsloth/Llama-3.2-1B-Instruct-GGUF", "1B params, very fast"),
            ModelEntry("Llama-3.2-3B", "unsloth/Llama-3.2-3B-Instruct-GGUF", "3B params, needs 8GB+ RAM"),
            // — Microsoft Phi GGUF — confirmed in GenieX docs —
            ModelEntry("Phi-4-Mini", "unsloth/Phi-4-mini-instruct-GGUF", "3.8B params, strong reasoning"),
            // — Google Gemma (Qualcomm HF repo) — confirmed on huggingface.co/qualcomm —
            ModelEntry("Gemma-4-E2B", "qualcomm/Gemma-4-E2B-it", "2B params, Google's efficient model"),
            ModelEntry("Gemma-2-2B", "unsloth/gemma-2-2b-it-GGUF", "2B params, older Gemma"),
            // — Mistral GGUF —
            ModelEntry("Mistral-7B", "bartowski/Mistral-7B-Instruct-v0.3-GGUF", "7B params, flagship-class, needs 12GB+ RAM"),
            // — NPU pre-compiled (Qualcomm AI Hub) — Snapdragon 8 Elite ONLY.
            //   The hub has no NPU build for mid-tier chips (e.g. SM7550), so this
            //   download returns rc=-100010 "not found on hub" on those devices.
            ModelEntry("Qwen3-4B (NPU · 8 Elite only)", "ai-hub-models/Qwen3-4B", "NPU pre-compiled — only downloads on Snapdragon 8 Elite / Elite Gen 5"),
        )
    }

    data class ModelEntry(
        val displayName: String,
        val modelId: String,
        val description: String,
    )

    data class GenieXResult(
        val modelName: String,
        val computeUnit: String,
        val ttftMs: Double,
        val prefillSpeed: Double,    // tok/s
        val decodeSpeed: Double,     // tok/s
        val promptTokens: Long,
        val generatedTokens: Long,
        val peakMemoryMb: Double,
        val loadTimeMs: Long,
        val generatedText: String,
        val success: Boolean,
        val error: String? = null,
    )

    data class ComparisonResult(
        val modelName: String,
        val results: List<GenieXResult>,
        val bestComputeUnit: String?,
        val recommendation: String,
    )

    interface ProgressCallback {
        fun onProgress(status: String)
        fun onModelDownloading(model: String, progress: Int)
        fun onToken(token: String)
    }

    // ---- Model residency: load / unload from RAM ----
    // A GenieX LlmWrapper pins the model in native memory until close() is called.
    // We keep the wrapper resident so it can be reused across runs, and free it on
    // demand so RAM drops after testing.
    private var loadedLlm: LlmWrapper? = null
    private var loadedKey: String? = null   // "modelId|computeUnit" of the resident model

    /** True if a model is currently held in RAM. */
    fun isLoaded(): Boolean = loadedLlm != null

    /** Display name of the resident model, or null if nothing is loaded. */
    fun loadedModelName(): String? {
        val id = loadedKey?.substringBefore('|') ?: return null
        return MODEL_CATALOG.find { it.modelId == id }?.displayName ?: id
    }

    /** Free the resident model's native memory. Safe to call when nothing is loaded. */
    fun unload() {
        try { loadedLlm?.close() } catch (_: Exception) {}
        loadedLlm = null
        loadedKey = null
    }

    /** Absolute process RSS in MB (reads /proc) — identical to LlmBenchmark's measurement. */
    private fun getRssMemoryMb(): Float {
        try {
            val status = File("/proc/self/status").readText()
            val match = Regex("VmRSS:\\s+(\\d+)\\s+kB").find(status)
            if (match != null) return match.groupValues[1].toFloat() / 1024f
        } catch (_: Exception) {}
        val rt = Runtime.getRuntime()
        return (rt.totalMemory() - rt.freeMemory()) / (1024f * 1024f)
    }

    /**
     * Load a model into RAM (without generating), or reuse it if already resident.
     * Loading a different model/compute-unit frees the previous one first.
     * Returns null on success, or a human-readable error message on failure.
     */
    suspend fun load(modelId: String, computeUnit: String, threads: Int = 4): String? = withContext(Dispatchers.IO) {
        val key = "$modelId|$computeUnit|$threads"
        if (loadedLlm != null && loadedKey == key) return@withContext null
        unload()  // free any other model before loading a new one

        // The native loader SIGSEGVs on a bad path, so the model must be downloaded
        // and we must pass its real on-disk paths (not the id).
        if (modelId !in ModelManagerWrapper.list()) {
            return@withContext "Model not downloaded. Tap \"Download Model\" first, then run."
        }
        val paths = ModelManagerWrapper.getPaths(modelId)
        val modelPath = paths?.model_path
        if (paths == null || modelPath.isNullOrEmpty() || !File(modelPath).exists()) {
            return@withContext "Model files missing on disk. Re-download the model."
        }
        val createInput = LlmCreateInput(
            model_name = modelId,
            model_path = modelPath,
            tokenizer_path = paths.tokenizer_path,
            config = ModelConfig(nCtx = 1024, nThreads = threads, nThreadsBatch = threads, nGpuLayers = if (computeUnit == "gpu") 999 else 0),
            runtime_id = paths.runtime_id,
            compute_unit = computeUnit,
        )
        val buildResult = LlmWrapper.builder().llmCreateInput(createInput).build()
        if (buildResult.isFailure) {
            return@withContext buildResult.exceptionOrNull()?.message ?: "Failed to load model"
        }
        loadedLlm = buildResult.getOrThrow()
        loadedKey = key
        null
    }

    /**
     * Benchmark a single model on a specific compute unit.
     */
    suspend fun benchmark(
        modelId: String,
        computeUnit: String,
        prompt: String = "Explain edge AI in one sentence.",
        maxTokens: Int = 64,
        threads: Int = 4,
        callback: ProgressCallback? = null,
    ): GenieXResult = withContext(Dispatchers.IO) {
        val modelName = MODEL_CATALOG.find { it.modelId == modelId }?.displayName ?: modelId

        callback?.onProgress("Loading $modelName on ${computeUnit.uppercase()}...")

        val loadStart = System.currentTimeMillis()

        try {
            // Load into RAM (or reuse the already-resident model). This resolves the
            // real on-disk paths and guards against the native SIGSEGV; the model stays
            // resident afterwards and is freed only via unload().
            val loadError = load(modelId, computeUnit, threads)
            if (loadError != null) {
                return@withContext GenieXResult(
                    modelName = modelName, computeUnit = computeUnit,
                    ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                    promptTokens = 0, generatedTokens = 0,
                    peakMemoryMb = 0.0, loadTimeMs = 0,
                    generatedText = "",
                    success = false,
                    error = loadError,
                )
            }
            val llm = loadedLlm!!
            val loadTimeMs = System.currentTimeMillis() - loadStart

            callback?.onProgress("Generating on ${computeUnit.uppercase()}...")

            // Build chat messages
            val messages = arrayOf(
                ChatMessage("user", prompt)
            )

            // Apply chat template
            val templateResult = llm.applyChatTemplate(
                messages = messages,
                tools = null,
                enableThinking = false,
            )

            if (templateResult.isFailure) {
                return@withContext GenieXResult(
                    modelName = modelName, computeUnit = computeUnit,
                    ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                    promptTokens = 0, generatedTokens = 0,
                    peakMemoryMb = 0.0, loadTimeMs = loadTimeMs,
                    generatedText = "",
                    success = false,
                    error = "Chat template failed: ${templateResult.exceptionOrNull()?.message}",
                )
            }

            val formattedPrompt = templateResult.getOrThrow().formattedText

            // Generate via streaming — GenieX exposes generation as a Flow<LlmStreamResult>.
            // Temperature lives on SamplerConfig; token budget is GenerationConfig.maxTokens.
            val sb = StringBuilder()
            val genConfig = GenerationConfig()
            genConfig.maxTokens = maxTokens
            genConfig.samplerConfig = SamplerConfig().apply { temperature = 0.1f }

            var profile: ProfilingData? = null
            var streamError: Throwable? = null

            llm.generateStreamFlow(formattedPrompt, genConfig).collect { result ->
                when (result) {
                    is LlmStreamResult.Token -> {
                        sb.append(result.text)
                        callback?.onToken(result.text)
                    }
                    is LlmStreamResult.Completed -> profile = result.profile
                    is LlmStreamResult.Error -> streamError = result.throwable
                }
            }

            val finalProfile = profile
            if (streamError != null || finalProfile == null) {
                return@withContext GenieXResult(
                    modelName = modelName, computeUnit = computeUnit,
                    ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                    promptTokens = 0, generatedTokens = 0,
                    peakMemoryMb = 0.0, loadTimeMs = loadTimeMs,
                    generatedText = sb.toString(),
                    success = false,
                    error = "Generation failed: ${streamError?.message ?: "no profiling data returned"}",
                )
            }

            // Absolute process RSS (same measurement as the Universal/llama.cpp card).
            // A native-heap delta misses the mmap'd GGUF and reports near-zero.
            val peakMemoryMb = getRssMemoryMb().toDouble()

            GenieXResult(
                modelName = modelName,
                computeUnit = computeUnit,
                ttftMs = finalProfile.ttftMs,
                prefillSpeed = finalProfile.prefillSpeed,
                decodeSpeed = finalProfile.decodingSpeed,
                promptTokens = finalProfile.promptTokens,
                generatedTokens = finalProfile.generatedTokens,
                peakMemoryMb = peakMemoryMb,
                loadTimeMs = loadTimeMs,
                generatedText = sb.toString(),
                success = true,
            )
        } catch (e: Exception) {
            Log.e(TAG, "GenieX benchmark failed", e)
            GenieXResult(
                modelName = modelName, computeUnit = computeUnit,
                ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                promptTokens = 0, generatedTokens = 0,
                peakMemoryMb = 0.0, loadTimeMs = 0,
                generatedText = "",
                success = false,
                error = e.message,
            )
        } finally {
            // Model stays resident in RAM (reusable across runs); freed only via unload().
        }
    }

    /**
     * Compare a model across multiple compute units (NPU vs GPU vs CPU).
     */
    suspend fun compareComputeUnits(
        modelId: String,
        computeUnits: List<String> = listOf("npu", "gpu", "cpu"),
        prompt: String = "Explain what edge AI is in one sentence.",
        maxTokens: Int = 128,
        callback: ProgressCallback? = null,
    ): ComparisonResult {
        val modelName = MODEL_CATALOG.find { it.modelId == modelId }?.displayName ?: modelId
        val results = mutableListOf<GenieXResult>()

        for (unit in computeUnits) {
            callback?.onProgress("Testing $modelName on ${unit.uppercase()}...")
            val result = benchmark(modelId, unit, prompt, maxTokens, callback)
            results.add(result)
        }

        val successResults = results.filter { it.success }
        val best = successResults.minByOrNull { it.ttftMs }

        val recommendation = buildRecommendation(modelName, results)

        return ComparisonResult(
            modelName = modelName,
            results = results,
            bestComputeUnit = best?.computeUnit,
            recommendation = recommendation,
        )
    }

    /**
     * Recommend the best model for this device based on available RAM.
     */
    fun recommendModelsForDevice(ramGb: Double): List<ModelRecommendation> {
        return MODEL_CATALOG.map { model ->
            val estimatedRamGb = estimateModelRam(model.modelId)
            val fits = ramGb >= estimatedRamGb * 1.3  // 30% headroom
            val speed = if (fits) estimateSpeed(model.modelId, ramGb) else "Too large"

            ModelRecommendation(
                model = model,
                fits = fits,
                estimatedRamGb = estimatedRamGb,
                expectedSpeed = speed,
                recommendation = when {
                    !fits -> "Not enough RAM (needs ${String.format(Locale.US, "%.1f", estimatedRamGb)} GB, you have ${String.format(Locale.US, "%.1f", ramGb)} GB)"
                    estimatedRamGb < ramGb * 0.5 -> "Recommended — runs comfortably with headroom"
                    else -> "Should work — may be tight on memory"
                },
            )
        }
    }

    data class ModelRecommendation(
        val model: ModelEntry,
        val fits: Boolean,
        val estimatedRamGb: Double,
        val expectedSpeed: String,
        val recommendation: String,
    )

    // ========================================================================
    // Helpers
    // ========================================================================

    private fun buildRecommendation(modelName: String, results: List<GenieXResult>): String {
        val successful = results.filter { it.success }
        if (successful.isEmpty()) return "All compute units failed for $modelName."

        val fastest = successful.minByOrNull { it.ttftMs }!!
        val mostEfficient = successful.maxByOrNull { it.decodeSpeed }!!

        return buildString {
            append("Best for $modelName on this device:\n")
            append("• Fastest first token: ${fastest.computeUnit.uppercase()} (${String.format(Locale.US, "%.0f", fastest.ttftMs)} ms)\n")
            append("• Fastest decode: ${mostEfficient.computeUnit.uppercase()} (${String.format(Locale.US, "%.1f", mostEfficient.decodeSpeed)} tok/s)\n")

            val npuResult = results.find { it.computeUnit == "npu" && it.success }
            val cpuResult = results.find { it.computeUnit == "cpu" && it.success }
            if (npuResult != null && cpuResult != null && cpuResult.ttftMs > 0) {
                val speedup = cpuResult.ttftMs / npuResult.ttftMs
                append("• NPU speedup vs CPU: ${String.format(Locale.US, "%.1f", speedup)}x")
            }
        }
    }

    private fun estimateModelRam(modelId: String): Double {
        return when {
            modelId.contains("0.6b", ignoreCase = true) -> 0.5
            modelId.contains("1.7b", ignoreCase = true) -> 1.5
            modelId.contains("2b", ignoreCase = true) -> 2.0
            modelId.contains("3b", ignoreCase = true) -> 3.0
            modelId.contains("4b", ignoreCase = true) -> 3.5
            modelId.contains("7b", ignoreCase = true) -> 6.0
            modelId.contains("8b", ignoreCase = true) -> 6.5
            else -> 2.0
        }
    }

    private fun estimateSpeed(modelId: String, ramGb: Double): String {
        val modelSize = estimateModelRam(modelId)
        return when {
            modelSize < 1.0 -> "Very fast (30+ tok/s)"
            modelSize < 2.0 -> "Fast (15-25 tok/s)"
            modelSize < 4.0 -> "Moderate (8-15 tok/s)"
            else -> "Slow (3-8 tok/s)"
        }
    }

    /**
     * Format a single result for display.
     */
    fun formatResult(result: GenieXResult): String {
        if (!result.success) return "Failed: ${result.error}"

        return buildString {
            append("═══ ${result.modelName} on ${result.computeUnit.uppercase()} ═══\n")
            append("TTFT:          ${String.format(Locale.US, "%.1f", result.ttftMs)} ms\n")
            append("Prefill:       ${String.format(Locale.US, "%.1f", result.prefillSpeed)} tok/s\n")
            append("Decode:        ${String.format(Locale.US, "%.1f", result.decodeSpeed)} tok/s\n")
            append("Tokens:        ${result.promptTokens} prompt → ${result.generatedTokens} generated\n")
            append("Peak Memory:   ${String.format(Locale.US, "%.1f", result.peakMemoryMb)} MB\n")
            append("Load Time:     ${result.loadTimeMs} ms\n")
        }
    }

    /**
     * Format comparison results for display.
     */
    fun formatComparison(comparison: ComparisonResult): String {
        return buildString {
            append("═══ ${comparison.modelName} — Compute Unit Comparison ═══\n\n")

            // Table header
            append(String.format(Locale.US, "%-8s %8s %10s %10s %8s\n",
                "Unit", "TTFT(ms)", "Decode", "Memory", "Status"))
            append("─".repeat(50) + "\n")

            for (r in comparison.results) {
                if (r.success) {
                    val best = if (r.computeUnit == comparison.bestComputeUnit) " ★" else ""
                    append(String.format(Locale.US, "%-8s %8.0f %8.1f t/s %7.0f MB %s%s\n",
                        r.computeUnit.uppercase(),
                        r.ttftMs,
                        r.decodeSpeed,
                        r.peakMemoryMb,
                        "✓",
                        best,
                    ))
                } else {
                    append(String.format(Locale.US, "%-8s %8s %10s %10s %s\n",
                        r.computeUnit.uppercase(), "—", "—", "—", "✗ ${r.error?.take(20)}"))
                }
            }

            append("\n${comparison.recommendation}")
        }
    }
}
