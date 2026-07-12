package ai.frozo.edgegate.benchmark

import android.content.Context
import android.os.Debug
import android.util.Log
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.bean.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
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

        /** Popular models users can pick from. */
        val MODEL_CATALOG = listOf(
            ModelEntry("Qwen3-0.6B", "qwen3-0.6b", "0.6B params, fast, good for testing"),
            ModelEntry("Qwen3-1.7B", "qwen3-1.7b", "1.7B params, balanced speed/quality"),
            ModelEntry("Llama-3.2-3B", "llama-3.2-3b-instruct", "3B params, needs 8GB+ RAM"),
            ModelEntry("Phi-4-Mini", "phi-4-mini-instruct", "3.8B params, strong reasoning"),
            ModelEntry("Gemma-4-2B", "gemma-4-e2b-it", "2B params, Google's efficient model"),
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

    /**
     * Benchmark a single model on a specific compute unit.
     */
    suspend fun benchmark(
        modelId: String,
        computeUnit: String,
        prompt: String = "Explain what edge AI is in one sentence.",
        maxTokens: Int = 128,
        callback: ProgressCallback? = null,
    ): GenieXResult = withContext(Dispatchers.IO) {
        val modelName = MODEL_CATALOG.find { it.modelId == modelId }?.displayName ?: modelId

        callback?.onProgress("Loading $modelName on ${computeUnit.uppercase()}...")

        val memBefore = Debug.getNativeHeapAllocatedSize()
        val loadStart = System.currentTimeMillis()

        var llm: LlmWrapper? = null
        try {
            val createInput = LlmCreateInput(
                model_name = modelId,
                model_path = modelId,  // GenieX resolves from cache
                config = ModelConfig(
                    nCtx = 1024,
                    nGpuLayers = if (computeUnit == "gpu") 999 else 0,
                ),
                compute_unit = computeUnit,
            )

            val buildResult = LlmWrapper.builder()
                .llmCreateInput(createInput)
                .build()

            if (buildResult.isFailure) {
                return@withContext GenieXResult(
                    modelName = modelName,
                    computeUnit = computeUnit,
                    ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                    promptTokens = 0, generatedTokens = 0,
                    peakMemoryMb = 0.0, loadTimeMs = 0,
                    generatedText = "",
                    success = false,
                    error = buildResult.exceptionOrNull()?.message ?: "Failed to load model",
                )
            }

            llm = buildResult.getOrThrow()
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

            val formattedPrompt = templateResult.getOrThrow().formattedPrompt

            // Generate with streaming
            val sb = StringBuilder()
            val genConfig = GenerationConfig(maxNewTokens = maxTokens, temperature = 0.1f)

            val generateResult = llm.generate(formattedPrompt, genConfig)

            if (generateResult.isFailure) {
                return@withContext GenieXResult(
                    modelName = modelName, computeUnit = computeUnit,
                    ttftMs = 0.0, prefillSpeed = 0.0, decodeSpeed = 0.0,
                    promptTokens = 0, generatedTokens = 0,
                    peakMemoryMb = 0.0, loadTimeMs = loadTimeMs,
                    generatedText = "",
                    success = false,
                    error = "Generation failed: ${generateResult.exceptionOrNull()?.message}",
                )
            }

            val output = generateResult.getOrThrow()
            val profile = output.profileData
            val memAfter = Debug.getNativeHeapAllocatedSize()
            val peakMemoryMb = (memAfter - memBefore).coerceAtLeast(0) / (1024.0 * 1024.0)

            GenieXResult(
                modelName = modelName,
                computeUnit = computeUnit,
                ttftMs = profile.ttftMs,
                prefillSpeed = profile.prefillSpeed,
                decodeSpeed = profile.decodingSpeed,
                promptTokens = profile.promptTokens,
                generatedTokens = profile.generatedTokens,
                peakMemoryMb = peakMemoryMb,
                loadTimeMs = loadTimeMs,
                generatedText = output.fullText,
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
            try { llm?.close() } catch (_: Exception) {}
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
