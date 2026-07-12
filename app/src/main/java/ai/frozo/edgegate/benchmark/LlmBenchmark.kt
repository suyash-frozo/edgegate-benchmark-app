package ai.frozo.edgegate.benchmark

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * LLM Benchmark Engine using llama.cpp via JNI.
 *
 * Runs actual text generation on the phone's hardware and measures:
 * - TTFT (Time to First Token)
 * - Tokens per second
 * - Total generation time
 * - Peak memory usage
 *
 * Supports GGUF models: Gemma 2B, Qwen 1.5B, Llama 1B/3B, Phi-3, etc.
 */

data class LlmResult(
    val modelName: String,
    val prompt: String,
    val generatedText: String,
    val totalTokens: Int,
    val promptTokens: Int,
    val generationTokens: Int,
    val ttftMs: Float,              // Time to first token
    val tokensPerSecond: Float,     // Generation throughput
    val totalTimeMs: Float,         // Total wall time
    val promptEvalTimeMs: Float,    // Prompt processing time
    val generationTimeMs: Float,    // Token generation time
    val peakMemoryMb: Float,
    val modelSizeBytes: Long,
    val success: Boolean,
    val error: String? = null,
)

data class LlmBenchmarkConfig(
    val prompt: String = "Explain edge AI in one sentence.",
    val maxTokens: Int = 64,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val threads: Int = 4,
    val gpuLayers: Int = 0,         // 0 = CPU only, >0 = offload to GPU
    val warmupRuns: Int = 1,
    val measurementRuns: Int = 3,
)

class LlmBenchmark(private val context: Context) {

    interface ProgressCallback {
        fun onProgress(status: String)
        fun onToken(token: String)     // Called for each generated token (streaming)
    }

    companion object {
        private var libraryLoaded = false

        fun isAvailable(): Boolean {
            return try {
                if (!libraryLoaded) {
                    System.loadLibrary("llama_jni")
                    libraryLoaded = true
                }
                true
            } catch (e: UnsatisfiedLinkError) {
                false
            }
        }
    }

    // JNI native methods
    private external fun nativeLoadModel(modelPath: String, threads: Int, gpuLayers: Int): Long
    private external fun nativeGenerate(
        modelPtr: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
    ): String  // Returns JSON with all metrics

    private external fun nativeUnloadModel(modelPtr: Long)
    private external fun nativeGetSystemInfo(): String

    /**
     * Run LLM benchmark on a GGUF model file.
     *
     * Returns results for each measurement run + aggregated median.
     */
    suspend fun benchmark(
        modelFile: File,
        config: LlmBenchmarkConfig = LlmBenchmarkConfig(),
        callback: ProgressCallback? = null,
    ): LlmResult = withContext(Dispatchers.Default) {

        if (!isAvailable()) {
            return@withContext LlmResult(
                modelName = modelFile.name,
                prompt = config.prompt,
                generatedText = "",
                totalTokens = 0,
                promptTokens = 0,
                generationTokens = 0,
                ttftMs = 0f,
                tokensPerSecond = 0f,
                totalTimeMs = 0f,
                promptEvalTimeMs = 0f,
                generationTimeMs = 0f,
                peakMemoryMb = 0f,
                modelSizeBytes = modelFile.length(),
                success = false,
                error = "llama.cpp native library not available. Rebuild with NDK.",
            )
        }

        callback?.onProgress("Loading model: ${modelFile.name}...")

        var modelPtr = 0L
        try {
            // Load model
            val loadStart = System.nanoTime()
            modelPtr = nativeLoadModel(modelFile.absolutePath, config.threads, config.gpuLayers)
            val loadTimeMs = (System.nanoTime() - loadStart) / 1_000_000f
            callback?.onProgress("Model loaded in ${"%.1f".format(loadTimeMs)} ms")

            if (modelPtr == 0L) {
                return@withContext buildFailResult(modelFile, config, "Failed to load model")
            }

            // Warmup
            if (config.warmupRuns > 0) {
                callback?.onProgress("Warmup (${config.warmupRuns} runs)...")
                for (i in 0 until config.warmupRuns) {
                    nativeGenerate(modelPtr, "Hi", 5, config.temperature, config.topP)
                }
            }

            // Measurement runs
            val results = mutableListOf<LlmResult>()
            for (i in 0 until config.measurementRuns) {
                callback?.onProgress("Run ${i + 1}/${config.measurementRuns}...")

                val memBefore = getRssMemoryMb()
                val resultJson = nativeGenerate(
                    modelPtr, config.prompt, config.maxTokens,
                    config.temperature, config.topP,
                )
                val memAfter = getRssMemoryMb()

                // Parse the JSON result from native code
                val parsed = parseNativeResult(resultJson, modelFile, config, memAfter)
                results.add(parsed)

                if (parsed.generatedText.isNotEmpty()) {
                    callback?.onToken(parsed.generatedText)
                }
            }

            // Aggregate: use median of measurement runs
            val sorted = results.filter { it.success }.sortedBy { it.tokensPerSecond }
            if (sorted.isEmpty()) {
                return@withContext buildFailResult(modelFile, config, "All runs failed")
            }

            val median = sorted[sorted.size / 2]
            return@withContext median

        } catch (e: Exception) {
            return@withContext buildFailResult(modelFile, config, "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            if (modelPtr != 0L) {
                try { nativeUnloadModel(modelPtr) } catch (_: Exception) {}
            }
        }
    }

    private fun parseNativeResult(
        json: String,
        modelFile: File,
        config: LlmBenchmarkConfig,
        peakMemoryMb: Float,
    ): LlmResult {
        return try {
            val obj = org.json.JSONObject(json)
            LlmResult(
                modelName = modelFile.name,
                prompt = config.prompt,
                generatedText = obj.optString("text", ""),
                totalTokens = obj.optInt("total_tokens", 0),
                promptTokens = obj.optInt("prompt_tokens", 0),
                generationTokens = obj.optInt("generation_tokens", 0),
                ttftMs = obj.optDouble("ttft_ms", 0.0).toFloat(),
                tokensPerSecond = obj.optDouble("tokens_per_second", 0.0).toFloat(),
                totalTimeMs = obj.optDouble("total_time_ms", 0.0).toFloat(),
                promptEvalTimeMs = obj.optDouble("prompt_eval_time_ms", 0.0).toFloat(),
                generationTimeMs = obj.optDouble("generation_time_ms", 0.0).toFloat(),
                peakMemoryMb = peakMemoryMb,
                modelSizeBytes = modelFile.length(),
                success = !obj.optBoolean("error", false),
                error = obj.optString("error_message", null),
            )
        } catch (e: Exception) {
            buildFailResult(modelFile, config, "Failed to parse result: ${e.message}")
        }
    }

    private fun buildFailResult(modelFile: File, config: LlmBenchmarkConfig, error: String): LlmResult {
        return LlmResult(
            modelName = modelFile.name, prompt = config.prompt, generatedText = "",
            totalTokens = 0, promptTokens = 0, generationTokens = 0,
            ttftMs = 0f, tokensPerSecond = 0f, totalTimeMs = 0f,
            promptEvalTimeMs = 0f, generationTimeMs = 0f, peakMemoryMb = 0f,
            modelSizeBytes = modelFile.length(), success = false, error = error,
        )
    }

    private fun getRssMemoryMb(): Float {
        try {
            val status = File("/proc/self/status").readText()
            val match = Regex("VmRSS:\\s+(\\d+)\\s+kB").find(status)
            if (match != null) return match.groupValues[1].toFloat() / 1024f
        } catch (_: Exception) {}
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024f * 1024f)
    }

    fun formatResult(result: LlmResult): String {
        if (!result.success) {
            return "LLM Benchmark FAILED: ${result.error}"
        }
        return buildString {
            appendLine("═══ LLM BENCHMARK ═══")
            appendLine("Model:      ${result.modelName}")
            appendLine("Size:       ${"%.1f".format(result.modelSizeBytes / (1024f * 1024f))} MB")
            appendLine()
            appendLine("═══ PERFORMANCE ═══")
            appendLine("TTFT:           ${"%.1f".format(result.ttftMs)} ms")
            appendLine("Tokens/sec:     ${"%.1f".format(result.tokensPerSecond)}")
            appendLine("Total time:     ${"%.0f".format(result.totalTimeMs)} ms")
            appendLine("Prompt eval:    ${"%.0f".format(result.promptEvalTimeMs)} ms")
            appendLine("Generation:     ${"%.0f".format(result.generationTimeMs)} ms")
            appendLine("Peak memory:    ${"%.1f".format(result.peakMemoryMb)} MB")
            appendLine()
            appendLine("═══ TOKENS ═══")
            appendLine("Prompt:     ${result.promptTokens} tokens")
            appendLine("Generated:  ${result.generationTokens} tokens")
            appendLine("Total:      ${result.totalTokens} tokens")
            appendLine()
            appendLine("═══ OUTPUT ═══")
            appendLine(result.generatedText.take(200))
            if (result.generatedText.length > 200) append("...")
        }
    }
}
