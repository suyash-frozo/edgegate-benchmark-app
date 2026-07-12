package ai.frozo.edgegate.benchmark

import android.content.Context
import android.os.Debug
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Benchmark engine that runs TFLite models on real hardware delegates.
 *
 * Tests CPU, GPU (OpenCL/Vulkan via TFLite GPU delegate), and
 * NNAPI (routes to Hexagon DSP on Qualcomm, APU on MediaTek, etc.)
 *
 * Produces the SAME metrics as Qualcomm AI Hub:
 *   - inference_time_ms (median of N, warmup excluded)
 *   - peak_memory_mb
 *   - compute unit identification (which delegate ran fastest = which HW unit)
 */

data class DelegateResult(
    val delegate: String,           // "cpu", "gpu", "nnapi"
    val inferenceTimeMs: Float,     // Median of measurement runs
    val p50Ms: Float,
    val p95Ms: Float,
    val p99Ms: Float,
    val peakMemoryMb: Float,
    val initTimeMs: Float,          // Delegate initialization time
    val allLatenciesMs: List<Float>,
    val warmupRuns: Int,
    val measurementRuns: Int,
    val success: Boolean,
    val error: String? = null,
)

data class BenchmarkConfig(
    val warmupRuns: Int = 5,
    val measurementRuns: Int = 50,
    val numThreads: Int = 4,
    val delegates: List<String> = listOf("cpu", "gpu", "nnapi"),
)

class BenchmarkEngine(private val context: Context) {

    interface ProgressCallback {
        fun onProgress(delegate: String, status: String)
    }

    /**
     * Run benchmark on a TFLite model file across all requested delegates.
     * Returns results for each delegate.
     */
    fun benchmark(
        modelFile: File,
        config: BenchmarkConfig = BenchmarkConfig(),
        callback: ProgressCallback? = null,
    ): List<DelegateResult> {
        val results = mutableListOf<DelegateResult>()

        for (delegate in config.delegates) {
            callback?.onProgress(delegate, "Initializing $delegate...")
            val result = benchmarkDelegate(modelFile, delegate, config, callback)
            results.add(result)
        }

        return results
    }

    private fun benchmarkDelegate(
        modelFile: File,
        delegate: String,
        config: BenchmarkConfig,
        callback: ProgressCallback?,
    ): DelegateResult {
        var interpreter: Interpreter? = null
        var gpuDelegate: GpuDelegate? = null
        var nnapiDelegate: NnApiDelegate? = null

        try {
            // Build interpreter with the specified delegate
            val options = Interpreter.Options().apply {
                numThreads = config.numThreads
            }

            val initStart = System.nanoTime()

            when (delegate) {
                "gpu" -> {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                }
                "nnapi" -> {
                    nnapiDelegate = NnApiDelegate()
                    options.addDelegate(nnapiDelegate)
                }
                // "cpu" — no delegate needed
            }

            interpreter = Interpreter(modelFile, options)
            val initTimeMs = (System.nanoTime() - initStart) / 1_000_000f

            // Prepare input/output buffers
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)
            val inputBuffer = createBuffer(inputTensor.shape(), inputTensor.dataType())
            val outputBuffer = createBuffer(outputTensor.shape(), outputTensor.dataType())

            // Warmup runs
            callback?.onProgress(delegate, "Warming up ${delegate.uppercase()} (${config.warmupRuns} runs)...")
            for (i in 0 until config.warmupRuns) {
                interpreter.run(inputBuffer, outputBuffer)
                inputBuffer.rewind()
                outputBuffer.rewind()
            }

            // Measurement runs
            val latencies = mutableListOf<Float>()
            val memBefore = getMemoryUsageMb()

            for (i in 0 until config.measurementRuns) {
                if (i % 10 == 0) {
                    callback?.onProgress(delegate, "${delegate.uppercase()}: ${i}/${config.measurementRuns}")
                }

                val start = System.nanoTime()
                interpreter.run(inputBuffer, outputBuffer)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                latencies.add(elapsed)

                inputBuffer.rewind()
                outputBuffer.rewind()
            }

            val memAfter = getMemoryUsageMb()

            // Compute statistics
            val sorted = latencies.sorted()
            val n = sorted.size
            val median = sorted[n / 2]
            val p50 = sorted[(n * 0.50).toInt()]
            val p95 = sorted[(n * 0.95).toInt().coerceAtMost(n - 1)]
            val p99 = sorted[(n * 0.99).toInt().coerceAtMost(n - 1)]

            callback?.onProgress(delegate, "${delegate.uppercase()}: ${median} ms (done)")

            return DelegateResult(
                delegate = delegate,
                inferenceTimeMs = median,
                p50Ms = p50,
                p95Ms = p95,
                p99Ms = p99,
                peakMemoryMb = memAfter,
                initTimeMs = initTimeMs,
                allLatenciesMs = sorted,
                warmupRuns = config.warmupRuns,
                measurementRuns = config.measurementRuns,
                success = true,
            )

        } catch (e: Exception) {
            return DelegateResult(
                delegate = delegate,
                inferenceTimeMs = 0f,
                p50Ms = 0f,
                p95Ms = 0f,
                p99Ms = 0f,
                peakMemoryMb = 0f,
                initTimeMs = 0f,
                allLatenciesMs = emptyList(),
                warmupRuns = config.warmupRuns,
                measurementRuns = config.measurementRuns,
                success = false,
                error = "${e.javaClass.simpleName}: ${e.message}",
            )
        } finally {
            interpreter?.close()
            gpuDelegate?.close()
            nnapiDelegate?.close()
        }
    }

    private fun createBuffer(shape: IntArray, dataType: org.tensorflow.lite.DataType): ByteBuffer {
        val size = shape.fold(1) { acc, dim -> acc * dim }
        val byteSize = when (dataType) {
            org.tensorflow.lite.DataType.FLOAT32 -> size * 4
            org.tensorflow.lite.DataType.UINT8 -> size
            org.tensorflow.lite.DataType.INT8 -> size
            org.tensorflow.lite.DataType.INT32 -> size * 4
            org.tensorflow.lite.DataType.INT64 -> size * 8
            else -> size * 4
        }
        return ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder())
    }

    private fun getMemoryUsageMb(): Float {
        val runtime = Runtime.getRuntime()
        val nativeHeap = Debug.getNativeHeapAllocatedSize()
        val javaUsed = runtime.totalMemory() - runtime.freeMemory()
        return (nativeHeap + javaUsed) / (1024f * 1024f)
    }

    companion object {
        fun formatResults(results: List<DelegateResult>, modelName: String, chip: String): String {
            val sb = StringBuilder()
            sb.appendLine("Model: $modelName")
            sb.appendLine("Chip:  $chip")
            sb.appendLine()
            sb.appendLine("%-8s %10s %10s %10s %8s".format("Delegate", "Median", "P95", "Memory", "Status"))
            sb.appendLine("-".repeat(50))

            val cpuResult = results.find { it.delegate == "cpu" }
            val best = results.filter { it.success }.minByOrNull { it.inferenceTimeMs }

            for (r in results.sortedBy { if (it.success) it.inferenceTimeMs else Float.MAX_VALUE }) {
                val speedup = if (cpuResult != null && r.success && cpuResult.success && r.inferenceTimeMs > 0) {
                    "%.1fx".format(cpuResult.inferenceTimeMs / r.inferenceTimeMs)
                } else ""

                val marker = if (r == best) " BEST" else ""
                if (r.success) {
                    sb.appendLine("%-8s %8.3f ms %8.3f ms %7.1f MB %s%s".format(
                        r.delegate.uppercase(),
                        r.inferenceTimeMs,
                        r.p95Ms,
                        r.peakMemoryMb,
                        speedup,
                        marker
                    ))
                } else {
                    sb.appendLine("%-8s %10s %10s %10s %s".format(
                        r.delegate.uppercase(), "FAILED", "-", "-", r.error ?: ""
                    ))
                }
            }

            if (best != null && cpuResult != null && cpuResult.success) {
                val speedup = cpuResult.inferenceTimeMs / best.inferenceTimeMs
                sb.appendLine()
                sb.appendLine("Best: ${best.delegate.uppercase()} at ${"%.3f".format(best.inferenceTimeMs)} ms (${
                    "%.1f".format(speedup)}x vs CPU)")
            }

            return sb.toString()
        }
    }
}
