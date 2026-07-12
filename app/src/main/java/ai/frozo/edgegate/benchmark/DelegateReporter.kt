package ai.frozo.edgegate.benchmark

import java.io.File

data class DelegationInfo(
    val totalOps: Int,
    val delegatedOps: Int,
    val cpuFallbackOps: Int,
    val delegateName: String,
)

object DelegateReporter {
    /**
     * Determine how many ops were delegated vs fell back to CPU.
     *
     * Strategy: Compare execution time of CPU-only vs delegated.
     * Then use the op count from model parsing + timing ratio to estimate.
     *
     * For TFLite: count ops by running with and without delegate,
     * comparing which ops got accelerated based on timing difference.
     */
    fun analyze(
        modelFile: File,
        delegate: String,
        totalLayers: Int,
        cpuTimeMs: Float,
        delegateTimeMs: Float,
    ): DelegationInfo {
        if (totalLayers <= 0 || cpuTimeMs <= 0f) {
            return DelegationInfo(totalLayers, 0, totalLayers, delegate)
        }

        if (delegate == "cpu") {
            return DelegationInfo(totalLayers, 0, totalLayers, "cpu")
        }

        // Estimate delegation from speedup ratio
        // If delegate is 10x faster than CPU, ~90% of ops were delegated
        // If delegate ≈ CPU speed, most ops fell back
        val speedup = cpuTimeMs / delegateTimeMs
        val delegationRatio = when {
            speedup >= 5f -> 0.95f   // Almost all ops delegated
            speedup >= 3f -> 0.85f
            speedup >= 2f -> 0.70f
            speedup >= 1.5f -> 0.50f
            speedup >= 1.1f -> 0.30f
            else -> 0.05f            // Barely any delegation (maybe overhead made it slower)
        }

        val delegatedOps = (totalLayers * delegationRatio).toInt().coerceIn(0, totalLayers)
        val cpuOps = totalLayers - delegatedOps

        return DelegationInfo(
            totalOps = totalLayers,
            delegatedOps = delegatedOps,
            cpuFallbackOps = cpuOps,
            delegateName = delegate,
        )
    }

    /**
     * Estimate compute percentage from timing data.
     * Uses the ratio of CPU-only time vs delegated time.
     */
    fun estimateComputePercent(
        cpuOnlyTimeMs: Float,
        delegateTimeMs: Float,
    ): Float {
        if (cpuOnlyTimeMs <= 0f || delegateTimeMs <= 0f) return 0f
        // The time saved by the delegate = work done by accelerator
        val timeSaved = cpuOnlyTimeMs - delegateTimeMs
        if (timeSaved <= 0f) return 0f  // Delegate was slower = no acceleration
        return ((timeSaved / cpuOnlyTimeMs) * 100f).coerceIn(0f, 100f)
    }
}
