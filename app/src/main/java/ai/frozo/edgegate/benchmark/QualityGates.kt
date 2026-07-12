package ai.frozo.edgegate.benchmark

import kotlin.math.abs
import kotlin.math.sqrt

// ============================================================================
// Data Models (matching EdgeGate web: edgegate/services/run.py)
// ============================================================================

data class Gate(
    val metric: String,      // "inference_time_ms", "peak_memory_mb", "throughput_fps", "ttft_ms"
    val operator: String,    // "lt", "lte", "gt", "gte", "eq"
    val threshold: Float,
)

data class GateResult(
    val metric: String,
    val operator: String,
    val threshold: Float,
    val actualValue: Float?,
    val passed: Boolean,
    val description: String?,
)

data class GatesEvaluation(
    val passed: Boolean,
    val results: List<GateResult>,
    val flakyMetrics: List<String>,
)

// ============================================================================
// Default Gate Presets
// ============================================================================

val VISION_MODEL_GATES = listOf(
    Gate("inference_time_ms", "lte", 50f),
    Gate("peak_memory_mb", "lte", 200f),
    Gate("throughput_fps", "gte", 20f),
)

val LLM_MODEL_GATES = listOf(
    Gate("ttft_ms", "lte", 500f),
    Gate("tokens_per_second", "gte", 5f),
    Gate("peak_memory_mb", "lte", 3000f),
)

// All metrics available for custom gates
val AVAILABLE_METRICS = listOf(
    "inference_time_ms",
    "peak_memory_mb",
    "throughput_fps",
    "ttft_ms",
    "tokens_per_second",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "init_time_ms",
    "cold_start_load_time_ms",
    "cold_start_peak_memory_mb",
    "warm_load_time_ms",
    "total_layers",
    "layers_on_npu",
    "layers_on_gpu",
    "layers_on_cpu",
    "npu_layer_percent",
    "gpu_layer_percent",
    "cpu_layer_percent",
    "compile_time_ms",
    "compile_peak_memory_mb",
    "compile_time_cv",
    "compile_peak_memory_cv",
    "npu_compute_estimated",
    "gpu_compute_estimated",
    "cpu_compute_estimated",
)

val AVAILABLE_OPERATORS = listOf("lt", "lte", "gt", "gte", "eq")

// ============================================================================
// Gate Evaluation (mirrors Python evaluate_gate / evaluate_gates exactly)
// ============================================================================

/**
 * Evaluate a single gate against metrics.
 * Matches edgegate/services/run.py evaluate_gate().
 */
fun evaluateGate(gate: Gate, metrics: Map<String, Float>): GateResult {
    val actual = metrics[gate.metric]

    if (actual == null) {
        return GateResult(
            metric = gate.metric,
            operator = gate.operator,
            threshold = gate.threshold,
            actualValue = null,
            passed = false,
            description = "Metric '${gate.metric}' not available",
        )
    }

    val passed = when (gate.operator) {
        "lt"  -> actual < gate.threshold
        "lte" -> actual <= gate.threshold
        "gt"  -> actual > gate.threshold
        "gte" -> actual >= gate.threshold
        "eq"  -> abs(actual - gate.threshold) < 1e-6f
        else  -> false
    }

    return GateResult(
        metric = gate.metric,
        operator = gate.operator,
        threshold = gate.threshold,
        actualValue = actual,
        passed = passed,
        description = null,
    )
}

/**
 * Evaluate all gates against metrics.
 * Matches edgegate/services/run.py evaluate_gates().
 */
fun evaluateGates(
    gates: List<Gate>,
    metrics: Map<String, Float>,
    allLatencies: List<Float> = emptyList(),
): GatesEvaluation {
    val results = gates.map { evaluateGate(it, metrics) }
    val allPassed = results.all { it.passed }
    val flaky = detectFlakyMetrics(allLatencies)

    return GatesEvaluation(
        passed = allPassed,
        results = results,
        flakyMetrics = flaky,
    )
}

// ============================================================================
// Flaky Metric Detection (mirrors Python detect_flaky_metrics exactly)
// ============================================================================

/**
 * Detect flaky metrics based on coefficient of variation.
 * CV > 0.10 (10%) = flaky.
 * Matches edgegate/services/run.py detect_flaky_metrics().
 */
fun detectFlakyMetrics(
    latencies: List<Float>,
    cvThreshold: Float = 0.10f,
): List<String> {
    if (latencies.size < 2) return emptyList()

    val mean = latencies.average().toFloat()
    if (mean <= 0f) return emptyList()

    val variance = latencies.map { (it - mean) * (it - mean) }.sum() / (latencies.size - 1)
    val stdev = sqrt(variance)
    val cv = stdev / mean

    return if (cv > cvThreshold) {
        listOf("inference_time_ms")
    } else {
        emptyList()
    }
}

// ============================================================================
// Metric Extraction from DelegateResult
// ============================================================================

/**
 * Extract a flat metrics map from a DelegateResult, keyed the same way
 * as the Python backend so gate metric names match.
 */
fun extractMetrics(result: DelegateResult): Map<String, Float> {
    val m = mutableMapOf<String, Float>()
    if (result.success) {
        m["inference_time_ms"] = result.inferenceTimeMs
        m["peak_memory_mb"] = result.peakMemoryMb
        m["throughput_fps"] = result.throughputFps
        m["ttft_ms"] = result.ttftMs
        m["tokens_per_second"] = result.tokensPerSecond
        m["p50_ms"] = result.p50Ms
        m["p95_ms"] = result.p95Ms
        m["p99_ms"] = result.p99Ms
        m["init_time_ms"] = result.initTimeMs
        m["cold_start_load_time_ms"] = result.coldStartLoadTimeMs
        m["cold_start_peak_memory_mb"] = result.coldStartPeakMemoryMb
        m["warm_load_time_ms"] = result.warmLoadTimeMs
        m["inference_time_cv"] = result.inferenceTimeCv
        m["peak_memory_cv"] = result.peakMemoryCv
        m["cold_start_time_cv"] = result.coldStartTimeCv
        m["warm_load_time_cv"] = result.warmLoadTimeCv
        // Layer breakdown metrics
        m["total_layers"] = result.totalLayers.toFloat()
        m["layers_on_npu"] = result.layersOnNpu.toFloat()
        m["layers_on_gpu"] = result.layersOnGpu.toFloat()
        m["layers_on_cpu"] = result.layersOnCpu.toFloat()
        m["npu_layer_percent"] = result.npuLayerPercent
        m["gpu_layer_percent"] = result.gpuLayerPercent
        m["cpu_layer_percent"] = result.cpuLayerPercent
        // Compile metrics
        m["compile_time_ms"] = result.compileTimeMs
        m["compile_peak_memory_mb"] = result.compilePeakMemoryMb
        m["compile_time_cv"] = result.compileTimeCv
        m["compile_peak_memory_cv"] = result.compilePeakMemoryCv
        // Compute estimates
        m["npu_compute_estimated"] = result.npuComputeEstimated
        m["gpu_compute_estimated"] = result.gpuComputeEstimated
        m["cpu_compute_estimated"] = result.cpuComputeEstimated
    }
    return m
}

// ============================================================================
// Formatting
// ============================================================================

private fun operatorSymbol(op: String): String = when (op) {
    "lt"  -> "<"
    "lte" -> "\u2264"   // <=
    "gt"  -> ">"
    "gte" -> "\u2265"   // >=
    "eq"  -> "="
    else  -> op
}

/**
 * Format gate evaluation results as a readable text block.
 */
fun formatGatesResult(evaluation: GatesEvaluation): String {
    val sb = StringBuilder()
    sb.appendLine("\u2550\u2550\u2550 QUALITY GATES \u2550\u2550\u2550")

    val passedCount = evaluation.results.count { it.passed }
    val totalCount = evaluation.results.size

    for (r in evaluation.results) {
        val icon = if (r.passed) "\u2705" else "\u274C"
        val sym = operatorSymbol(r.operator)
        val actualStr = if (r.actualValue != null) {
            "actual: ${"%.1f".format(r.actualValue)}"
        } else {
            r.description ?: "N/A"
        }
        val status = if (r.passed) "PASS" else "FAIL"
        sb.appendLine("$icon ${r.metric} $sym ${"%.1f".format(r.threshold)}    $actualStr    $status")
    }

    sb.appendLine()
    sb.appendLine("Result: $passedCount/$totalCount PASSED")

    val flakyStr = if (evaluation.flakyMetrics.isEmpty()) "none"
    else evaluation.flakyMetrics.joinToString(", ")
    sb.appendLine("Flaky metrics: $flakyStr")

    return sb.toString()
}
