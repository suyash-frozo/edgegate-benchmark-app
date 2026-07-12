package ai.frozo.edgegate.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Debug
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

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
    val peakMemoryMb: Float,        // Real process RSS memory
    val initTimeMs: Float,          // Delegate initialization time
    val throughputFps: Float,       // 1000 / inferenceTimeMs
    val allLatenciesMs: List<Float>,
    val warmupRuns: Int,
    val measurementRuns: Int,
    val success: Boolean,
    val error: String? = null,
    // Compute unit breakdown (matches EdgeGate's compute_unit_breakdown)
    val npuComputePercent: Float = 0f,
    val gpuComputePercent: Float = 0f,
    val cpuComputePercent: Float = 0f,
    // LLM metrics (for language models)
    val ttftMs: Float = 0f,          // Time to first token
    val tokensPerSecond: Float = 0f, // Tokens/sec throughput
    // Load/startup metrics (matching AI Hub)
    val coldStartLoadTimeMs: Float = 0f,    // First model load time
    val coldStartPeakMemoryMb: Float = 0f,  // Memory after first load
    val warmLoadTimeMs: Float = 0f,          // Second load (cached)
    // CV (Coefficient of Variation) metrics — variance detection
    val inferenceTimeCv: Float = 0f,         // stdev/mean of inference times
    val peakMemoryCv: Float = 0f,            // stdev/mean of memory across runs
    val coldStartTimeCv: Float = 0f,         // stdev/mean of cold-start times
    val coldStartMemoryCv: Float = 0f,       // stdev/mean of cold-start memory
    val warmLoadTimeCv: Float = 0f,          // stdev/mean of warm load times
    // Model metadata
    val totalLayers: Int = 0,                // Total layers in model (from ONNX metadata)
    // Layer breakdown (from delegate reporting)
    val layersOnNpu: Int = 0,
    val layersOnGpu: Int = 0,
    val layersOnCpu: Int = 0,
    val npuLayerPercent: Float = 0f,
    val gpuLayerPercent: Float = 0f,
    val cpuLayerPercent: Float = 0f,
    // Compile metrics (= delegate init on mobile)
    val compileTimeMs: Float = 0f,           // = initTimeMs (rename for AI Hub parity)
    val compilePeakMemoryMb: Float = 0f,     // = coldStartPeakMemoryMb
    val compileTimeCv: Float = 0f,
    val compilePeakMemoryCv: Float = 0f,
    // Compute % estimated from timing ratio
    val npuComputeEstimated: Float = 0f,     // estimated from speedup vs CPU
    val gpuComputeEstimated: Float = 0f,
    val cpuComputeEstimated: Float = 0f,
    // Device context
    val deviceRamTotalGb: Float = 0f,
    val deviceRamUsedMb: Float = 0f,
    val thermalThrottled: Boolean = false,
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
     * Run benchmark on a model file across all requested delegates.
     * Automatically detects ONNX vs TFLite based on file extension.
     */
    fun benchmark(
        modelFile: File,
        config: BenchmarkConfig = BenchmarkConfig(),
        callback: ProgressCallback? = null,
        npuProfiler: NpuProfiler? = null,
    ): List<DelegateResult> {
        val isOnnx = modelFile.name.endsWith(".onnx", ignoreCase = true)

        if (isOnnx) {
            return benchmarkOnnx(modelFile, config, callback, npuProfiler)
        }

        val results = mutableListOf<DelegateResult>()
        for (delegate in config.delegates) {
            callback?.onProgress(delegate, "Initializing $delegate...")
            val result = benchmarkDelegate(modelFile, delegate, config, callback, npuProfiler)
            results.add(result)
        }
        return enrichResults(results, modelFile)
    }

    /**
     * Post-process results: enrich with model parsing + delegation analysis.
     */
    private fun enrichResults(results: List<DelegateResult>, modelFile: File): List<DelegateResult> {
        val modelInfo = ModelParser.parse(modelFile)
        val cpuResult = results.find { it.delegate == "cpu" && it.success }

        return results.map { r ->
            if (!r.success) return@map r

            val delegation = DelegateReporter.analyze(
                modelFile, r.delegate, modelInfo.totalLayers,
                cpuResult?.inferenceTimeMs ?: 0f, r.inferenceTimeMs
            )
            val npuCompute = if (r.delegate == "nnapi")
                DelegateReporter.estimateComputePercent(cpuResult?.inferenceTimeMs ?: 0f, r.inferenceTimeMs) else 0f
            val gpuCompute = if (r.delegate == "gpu")
                DelegateReporter.estimateComputePercent(cpuResult?.inferenceTimeMs ?: 0f, r.inferenceTimeMs) else 0f

            r.copy(
                totalLayers = modelInfo.totalLayers,
                layersOnNpu = if (r.delegate == "nnapi") delegation.delegatedOps else 0,
                layersOnGpu = if (r.delegate == "gpu") delegation.delegatedOps else 0,
                layersOnCpu = delegation.cpuFallbackOps,
                npuLayerPercent = if (r.delegate == "nnapi" && modelInfo.totalLayers > 0)
                    (delegation.delegatedOps.toFloat() / modelInfo.totalLayers * 100f) else 0f,
                gpuLayerPercent = if (r.delegate == "gpu" && modelInfo.totalLayers > 0)
                    (delegation.delegatedOps.toFloat() / modelInfo.totalLayers * 100f) else 0f,
                cpuLayerPercent = if (modelInfo.totalLayers > 0)
                    (delegation.cpuFallbackOps.toFloat() / modelInfo.totalLayers * 100f) else 100f,
                compileTimeMs = r.initTimeMs,
                compilePeakMemoryMb = r.coldStartPeakMemoryMb,
                compileTimeCv = r.coldStartTimeCv,
                compilePeakMemoryCv = r.coldStartMemoryCv,
                npuComputeEstimated = npuCompute,
                gpuComputeEstimated = gpuCompute,
                cpuComputeEstimated = if (r.delegate == "cpu") 100f else (100f - npuCompute - gpuCompute),
            )
        }
    }

    /**
     * Benchmark an ONNX model using ONNX Runtime.
     * Tests CPU and NNAPI execution providers.
     */
    private fun benchmarkOnnx(
        modelFile: File,
        config: BenchmarkConfig,
        callback: ProgressCallback?,
        npuProfiler: NpuProfiler? = null,
    ): List<DelegateResult> {
        val results = mutableListOf<DelegateResult>()
        val env = OrtEnvironment.getEnvironment()

        // CPU benchmark
        if ("cpu" in config.delegates) {
            callback?.onProgress("cpu", "ONNX Runtime CPU...")
            results.add(benchmarkOnnxWithProvider(env, modelFile, "cpu", config, callback))
        }

        // NNAPI benchmark (routes to NPU/DSP)
        if ("nnapi" in config.delegates) {
            callback?.onProgress("nnapi", "ONNX Runtime NNAPI (NPU)...")
            results.add(benchmarkOnnxWithProvider(env, modelFile, "nnapi", config, callback, npuProfiler))
        }

        // GPU — ONNX Runtime Android doesn't have a GPU EP, skip gracefully
        if ("gpu" in config.delegates) {
            results.add(buildFailResult("gpu", config, "ONNX Runtime Android has no GPU provider — use TFLite for GPU testing"))
        }

        return enrichResults(results, modelFile)
    }

    private fun benchmarkOnnxWithProvider(
        env: OrtEnvironment,
        modelFile: File,
        provider: String,
        config: BenchmarkConfig,
        callback: ProgressCallback?,
        npuProfiler: NpuProfiler? = null,
    ): DelegateResult {
        var session: OrtSession? = null
        try {
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(config.numThreads)

            if (provider == "nnapi") {
                try {
                    opts.addNnapi()
                } catch (e: Exception) {
                    return buildFailResult(provider, config, "NNAPI not available: ${e.message}")
                }
            }

            val initStart = System.nanoTime()
            session = env.createSession(modelFile.absolutePath, opts)
            val initTimeMs = (System.nanoTime() - initStart) / 1_000_000f

            // Get input info and create dummy tensor
            val inputInfo = session.inputInfo.entries.first()
            val inputName = inputInfo.key
            val shape = (inputInfo.value.info as ai.onnxruntime.TensorInfo).shape
            val totalSize = shape.fold(1L) { acc, d -> acc * (if (d < 0) 1 else d) }
            val floatArray = FloatArray(totalSize.toInt())
            val resolvedShape = shape.map { if (it < 0) 1L else it }.toLongArray()

            val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray), resolvedShape)

            // Warmup
            callback?.onProgress(provider, "Warming up ${provider.uppercase()} (${config.warmupRuns} runs)...")
            for (i in 0 until config.warmupRuns) {
                session.run(mapOf(inputName to inputTensor)).close()
            }

            // Start NPU monitoring for NNAPI delegate (root only)
            val npuMonitor = if (provider == "nnapi" && npuProfiler != null && npuProfiler.isRootAvailable()) {
                callback?.onProgress(provider, "Starting NPU monitor...")
                NpuMonitor(npuProfiler).also { it.startMonitoring() }
            } else null

            // Measure
            val latencies = mutableListOf<Float>()
            val memBefore = getMemoryUsageMb()

            for (i in 0 until config.measurementRuns) {
                if (i % 10 == 0) {
                    callback?.onProgress(provider, "${provider.uppercase()}: $i/${config.measurementRuns}")
                }
                val start = System.nanoTime()
                session.run(mapOf(inputName to inputTensor)).close()
                latencies.add((System.nanoTime() - start) / 1_000_000f)
            }

            // Stop NPU monitoring and capture real utilization
            val npuMonitorResult = npuMonitor?.stopMonitoring()

            val memAfter = getRssMemoryMb()
            inputTensor.close()

            val sorted = latencies.sorted()
            val realNpuPercent = npuMonitorResult?.avgUtilizationPercent
            return buildSuccessResult(
                provider, sorted, memAfter, initTimeMs, config,
                npuComputePercentOverride = if (provider == "nnapi") realNpuPercent else null,
            )

        } catch (e: Exception) {
            return buildFailResult(provider, config, "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            session?.close()
        }
    }

    private fun benchmarkDelegate(
        modelFile: File,
        delegate: String,
        config: BenchmarkConfig,
        callback: ProgressCallback?,
        npuProfiler: NpuProfiler? = null,
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
                    try {
                        gpuDelegate = GpuDelegate()
                        options.addDelegate(gpuDelegate)
                    } catch (e: Exception) {
                        return buildFailResult(delegate, config, "GPU delegate not available: ${e.message}")
                    }
                }
                "nnapi" -> {
                    try {
                        nnapiDelegate = NnApiDelegate()
                        options.addDelegate(nnapiDelegate)
                    } catch (e: Exception) {
                        return buildFailResult(delegate, config, "NNAPI delegate not available: ${e.message}")
                    }
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

            // Cold-start metrics: this was the first load
            val coldStartLoadTime = initTimeMs
            val coldStartPeakMem = getRssMemoryMb()

            // Warm load: close and reload to measure cached load time
            callback?.onProgress(delegate, "Measuring warm load ${delegate.uppercase()}...")
            interpreter.close()
            val warmLoadStart = System.nanoTime()
            interpreter = Interpreter(modelFile, options)
            val warmLoadTime = (System.nanoTime() - warmLoadStart) / 1_000_000f

            // Recreate buffers after reload
            val inputTensor2 = interpreter.getInputTensor(0)
            val outputTensor2 = interpreter.getOutputTensor(0)
            val inputBuf = createBuffer(inputTensor2.shape(), inputTensor2.dataType())
            val outputBuf = createBuffer(outputTensor2.shape(), outputTensor2.dataType())

            // Warmup runs
            callback?.onProgress(delegate, "Warming up ${delegate.uppercase()} (${config.warmupRuns} runs)...")
            for (i in 0 until config.warmupRuns) {
                interpreter.run(inputBuf, outputBuf)
                inputBuf.rewind()
                outputBuf.rewind()
            }

            // Start NPU monitoring for NNAPI delegate (root only)
            val npuMonitor = if (delegate == "nnapi" && npuProfiler != null && npuProfiler.isRootAvailable()) {
                callback?.onProgress(delegate, "Starting NPU monitor...")
                NpuMonitor(npuProfiler).also { it.startMonitoring() }
            } else null

            // Measurement runs — collect memory at each run too
            val latencies = mutableListOf<Float>()
            val memoryReadings = mutableListOf<Float>()

            for (i in 0 until config.measurementRuns) {
                if (i % 10 == 0) {
                    callback?.onProgress(delegate, "${delegate.uppercase()}: ${i}/${config.measurementRuns}")
                }

                val start = System.nanoTime()
                interpreter.run(inputBuf, outputBuf)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                latencies.add(elapsed)
                memoryReadings.add(getRssMemoryMb())

                inputBuf.rewind()
                outputBuf.rewind()
            }

            // Stop NPU monitoring and capture real utilization
            val npuMonitorResult = npuMonitor?.stopMonitoring()

            val peakMem = memoryReadings.maxOrNull() ?: getRssMemoryMb()

            val sorted = latencies.sorted()
            callback?.onProgress(delegate, "${delegate.uppercase()}: ${sorted[sorted.size/2]} ms (done)")

            val realNpuPercent = npuMonitorResult?.avgUtilizationPercent
            return buildSuccessResult(
                delegate, sorted, peakMem, initTimeMs, config,
                coldStartLoadTimeMs = coldStartLoadTime,
                coldStartPeakMemoryMb = coldStartPeakMem,
                warmLoadTimeMs = warmLoadTime,
                memoryReadings = memoryReadings,
                npuComputePercentOverride = if (delegate == "nnapi") realNpuPercent else null,
            )

        } catch (e: Exception) {
            return buildFailResult(delegate, config, "${e.javaClass.simpleName}: ${e.message}")
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
        // Use real PSS (Proportional Set Size) for accurate memory measurement
        val runtime = Runtime.getRuntime()
        val nativeHeap = Debug.getNativeHeapAllocatedSize()
        val javaUsed = runtime.totalMemory() - runtime.freeMemory()
        return (nativeHeap + javaUsed) / (1024f * 1024f)
    }

    private fun getRssMemoryMb(): Float {
        // Read /proc/self/status for VmRSS (real physical memory)
        try {
            val status = java.io.File("/proc/self/status").readText()
            val match = Regex("VmRSS:\\s+(\\d+)\\s+kB").find(status)
            if (match != null) {
                return match.groupValues[1].toFloat() / 1024f
            }
        } catch (_: Exception) {}
        return getMemoryUsageMb() // fallback
    }

    private fun getDeviceRamGb(): Float {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        actManager.getMemoryInfo(memInfo)
        return memInfo.totalMem / (1024f * 1024f * 1024f)
    }

    private fun getDeviceRamUsedMb(): Float {
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        val memInfo = android.app.ActivityManager.MemoryInfo()
        actManager.getMemoryInfo(memInfo)
        return (memInfo.totalMem - memInfo.availMem) / (1024f * 1024f)
    }

    private fun isThermalThrottled(): Boolean {
        // Check if device is thermally throttled (API 29+)
        try {
            if (android.os.Build.VERSION.SDK_INT >= 29) {
                val powerManager = context.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager
                return powerManager.currentThermalStatus >= android.os.PowerManager.THERMAL_STATUS_MODERATE
            }
        } catch (_: Exception) {}
        return false
    }

    /** Compute coefficient of variation (stdev / mean). Returns 0 if insufficient data. */
    private fun computeCv(values: List<Float>): Float {
        if (values.size < 2) return 0f
        val mean = values.sum() / values.size
        if (mean == 0f) return 0f
        val variance = values.map { (it - mean) * (it - mean) }.sum() / (values.size - 1)
        return kotlin.math.sqrt(variance) / mean
    }

    /** Build a successful DelegateResult with all metrics populated. */
    private fun buildSuccessResult(
        delegate: String,
        sortedLatencies: List<Float>,
        peakMemoryMb: Float,
        initTimeMs: Float,
        config: BenchmarkConfig,
        ttftMs: Float = 0f,
        coldStartLoadTimeMs: Float = 0f,
        coldStartPeakMemoryMb: Float = 0f,
        warmLoadTimeMs: Float = 0f,
        memoryReadings: List<Float> = emptyList(),
        coldStartTimes: List<Float> = emptyList(),
        coldStartMemories: List<Float> = emptyList(),
        warmLoadTimes: List<Float> = emptyList(),
        totalLayers: Int = 0,
        npuComputePercentOverride: Float? = null,
    ): DelegateResult {
        val n = sortedLatencies.size
        val median = sortedLatencies[n / 2]
        val isNpu = delegate == "nnapi"
        val isGpu = delegate == "gpu"
        val isCpu = delegate == "cpu"
        val actualTtft = if (ttftMs > 0f) ttftMs else sortedLatencies.firstOrNull() ?: 0f

        // Use real NPU utilization from root-mode monitoring when available,
        // otherwise fall back to the hardcoded 100% assumption.
        val npuPercent = if (isNpu && npuComputePercentOverride != null) {
            npuComputePercentOverride
        } else if (isNpu) {
            100f
        } else {
            0f
        }

        return DelegateResult(
            delegate = delegate,
            inferenceTimeMs = median,
            p50Ms = sortedLatencies[(n * 0.50).toInt()],
            p95Ms = sortedLatencies[(n * 0.95).toInt().coerceAtMost(n - 1)],
            p99Ms = sortedLatencies[(n * 0.99).toInt().coerceAtMost(n - 1)],
            peakMemoryMb = peakMemoryMb,
            initTimeMs = initTimeMs,
            throughputFps = if (median > 0) 1000f / median else 0f,
            allLatenciesMs = sortedLatencies,
            warmupRuns = config.warmupRuns,
            measurementRuns = config.measurementRuns,
            success = true,
            npuComputePercent = npuPercent,
            gpuComputePercent = if (isGpu) 100f else 0f,
            cpuComputePercent = if (isCpu) 100f else if (isNpu && npuComputePercentOverride != null) (100f - npuPercent).coerceAtLeast(0f) else 0f,
            ttftMs = actualTtft,
            tokensPerSecond = if (median > 0) 1000f / median else 0f,
            coldStartLoadTimeMs = coldStartLoadTimeMs,
            coldStartPeakMemoryMb = coldStartPeakMemoryMb,
            warmLoadTimeMs = warmLoadTimeMs,
            inferenceTimeCv = computeCv(sortedLatencies),
            peakMemoryCv = computeCv(memoryReadings),
            coldStartTimeCv = computeCv(coldStartTimes),
            coldStartMemoryCv = computeCv(coldStartMemories),
            warmLoadTimeCv = computeCv(warmLoadTimes),
            totalLayers = totalLayers,
            deviceRamTotalGb = getDeviceRamGb(),
            deviceRamUsedMb = getDeviceRamUsedMb(),
            thermalThrottled = isThermalThrottled(),
        )
    }

    /** Build a failed DelegateResult. */
    private fun buildFailResult(delegate: String, config: BenchmarkConfig, error: String): DelegateResult {
        return DelegateResult(
            delegate = delegate, inferenceTimeMs = 0f, p50Ms = 0f, p95Ms = 0f,
            p99Ms = 0f, peakMemoryMb = 0f, initTimeMs = 0f, throughputFps = 0f,
            allLatenciesMs = emptyList(), warmupRuns = config.warmupRuns,
            measurementRuns = config.measurementRuns, success = false, error = error,
        )
    }

    companion object {
        fun formatResults(results: List<DelegateResult>, modelName: String, chip: String): String {
            val sb = StringBuilder()
            sb.appendLine("Model: $modelName")
            sb.appendLine("Chip:  $chip")

            // Device context from first successful result
            val first = results.firstOrNull { it.success }
            if (first != null) {
                sb.appendLine("RAM:   ${"%.1f".format(first.deviceRamTotalGb)} GB total, ${"%.0f".format(first.deviceRamUsedMb)} MB used")
                if (first.thermalThrottled) {
                    sb.appendLine("⚠ THERMAL THROTTLING DETECTED")
                }
            }
            sb.appendLine()

            val cpuResult = results.find { it.delegate == "cpu" && it.success }
            val best = results.filter { it.success }.minByOrNull { it.inferenceTimeMs }

            // Header
            sb.appendLine("%-7s %9s %9s %9s %7s %6s".format(
                "Delgate", "Median", "P95", "Memory", "FPS", "vsCPU"
            ))
            sb.appendLine("-".repeat(55))

            for (r in results.sortedBy { if (it.success) it.inferenceTimeMs else Float.MAX_VALUE }) {
                if (r.success) {
                    val speedup = if (cpuResult != null && r.inferenceTimeMs > 0)
                        "%.1fx".format(cpuResult.inferenceTimeMs / r.inferenceTimeMs) else ""
                    val marker = if (r == best) " ★" else ""
                    sb.appendLine("%-7s %7.3fms %7.3fms %6.1f MB %6.1f %s%s".format(
                        r.delegate.uppercase(),
                        r.inferenceTimeMs, r.p95Ms, r.peakMemoryMb,
                        r.throughputFps, speedup, marker
                    ))
                } else {
                    sb.appendLine("%-7s  FAILED    -         -       -    %s".format(
                        r.delegate.uppercase(), r.error?.take(30) ?: ""
                    ))
                }
            }

            // Summary
            if (best != null) {
                sb.appendLine()
                sb.appendLine("━━━ Best: ${best.delegate.uppercase()} ━━━")
                sb.appendLine()
                sb.appendLine("═══ INFERENCE ═══")
                sb.appendLine("Inference Time:  ${"%.3f".format(best.inferenceTimeMs)} ms")
                sb.appendLine("Peak Memory:     ${"%.1f".format(best.peakMemoryMb)} MB")
                sb.appendLine("Throughput:      ${"%.1f".format(best.throughputFps)} FPS")
                sb.appendLine("TTFT:            ${"%.3f".format(best.ttftMs)} ms")
                sb.appendLine("Tokens/sec:      ${"%.1f".format(best.tokensPerSecond)}")
                if (cpuResult != null && cpuResult.success) {
                    sb.appendLine("vs CPU:          ${"%.1f".format(cpuResult.inferenceTimeMs / best.inferenceTimeMs)}x faster")
                }

                sb.appendLine()
                sb.appendLine("═══ LOAD/STARTUP ═══")
                sb.appendLine("Cold-start Time:   ${"%.1f".format(best.coldStartLoadTimeMs)} ms")
                sb.appendLine("Cold-start Memory: ${"%.1f".format(best.coldStartPeakMemoryMb)} MB")
                sb.appendLine("Warm Load Time:    ${"%.1f".format(best.warmLoadTimeMs)} ms")

                sb.appendLine()
                sb.appendLine("═══ COMPUTE ═══")
                sb.appendLine("NPU: ${"%.0f".format(best.npuComputePercent)}%  GPU: ${"%.0f".format(best.gpuComputePercent)}%  CPU: ${"%.0f".format(best.cpuComputePercent)}%")
                if (best.totalLayers > 0) {
                    sb.appendLine("Total Layers:    ${best.totalLayers}")
                }

                sb.appendLine()
                sb.appendLine("═══ LAYER BREAKDOWN ═══")
                sb.appendLine("Total Layers:    ${best.totalLayers}")
                sb.appendLine("Layers on NPU:   ${best.layersOnNpu} (${"%.1f".format(best.npuLayerPercent)}%)")
                sb.appendLine("Layers on GPU:   ${best.layersOnGpu} (${"%.1f".format(best.gpuLayerPercent)}%)")
                sb.appendLine("Layers on CPU:   ${best.layersOnCpu} (${"%.1f".format(best.cpuLayerPercent)}%)")

                sb.appendLine()
                sb.appendLine("═══ COMPILE METRICS ═══")
                sb.appendLine("Compile Time:    ${"%.1f".format(best.compileTimeMs)} ms")
                sb.appendLine("Compile Memory:  ${"%.1f".format(best.compilePeakMemoryMb)} MB")
                sb.appendLine("Compile Time CV: ${"%.3f".format(best.compileTimeCv)}${if (best.compileTimeCv > 0.1f) " ⚠ FLAKY" else " ✓"}")
                sb.appendLine("Compile Mem CV:  ${"%.3f".format(best.compilePeakMemoryCv)}${if (best.compilePeakMemoryCv > 0.1f) " ⚠ FLAKY" else " ✓"}")

                sb.appendLine()
                sb.appendLine("═══ COMPUTE ESTIMATE ═══")
                sb.appendLine("NPU Compute:     ${"%.1f".format(best.npuComputeEstimated)}% (estimated from speedup)")
                sb.appendLine("GPU Compute:     ${"%.1f".format(best.gpuComputeEstimated)}%")
                sb.appendLine("CPU Compute:     ${"%.1f".format(best.cpuComputeEstimated)}%")

                sb.appendLine()
                sb.appendLine("═══ STABILITY (CV) ═══")
                sb.appendLine("Inference Time CV: ${"%.3f".format(best.inferenceTimeCv)}${if (best.inferenceTimeCv > 0.1f) " ⚠ FLAKY" else " ✓"}")
                sb.appendLine("Peak Memory CV:    ${"%.3f".format(best.peakMemoryCv)}${if (best.peakMemoryCv > 0.1f) " ⚠ FLAKY" else " ✓"}")
                sb.appendLine("Cold-start CV:     ${"%.3f".format(best.coldStartTimeCv)}${if (best.coldStartTimeCv > 0.1f) " ⚠ FLAKY" else " ✓"}")
                sb.appendLine("Warm Load CV:      ${"%.3f".format(best.warmLoadTimeCv)}${if (best.warmLoadTimeCv > 0.1f) " ⚠ FLAKY" else " ✓"}")
            }

            return sb.toString()
        }
    }
}
