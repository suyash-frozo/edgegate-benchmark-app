package ai.frozo.edgegate.benchmark

import android.content.Context
import android.os.Build
import android.util.Log
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Root-mode NPU hardware profiler and NNAPI device enumerator.
 *
 * Two modes of operation:
 * 1. **No root required** — [detectNpuHardware] probes well-known sysfs/dev nodes
 *    to enumerate which accelerators are physically present (CPU, GPU, NPU/DSP).
 * 2. **Root required** — [readNpuMetrics] reads real-time frequency, utilization,
 *    and power-level counters from vendor-specific sysfs paths via `su`.
 *
 * Usage:
 * ```
 * val profiler = NpuProfiler(context)
 * val devices  = profiler.detectNpuHardware()  // always works
 * val root     = profiler.isRootAvailable()     // one-time check
 * val metrics  = profiler.readNpuMetrics()       // real data when rooted
 * ```
 */

// ── Data classes ────────────────────────────────────────────────────────────────

data class NnDevice(
    val name: String,       // "qti-hta", "mtk-neuron", "samsung-npu", "gpu", "cpu"
    val type: Int,          // 1=OTHER, 2=CPU, 3=GPU, 4=ACCELERATOR
    val typeName: String,   // "CPU", "GPU", "ACCELERATOR", "OTHER"
    val version: String,    // driver / description string
)

data class NpuMetrics(
    val npuFrequencyMhz: Int,
    val npuUtilizationPercent: Float,
    val dramUtilizationPercent: Float,
    val npuPowerLevel: Int,
    val isRootAvailable: Boolean,
)

data class NpuMonitorResult(
    val avgUtilizationPercent: Float,
    val peakUtilizationPercent: Float,
    val readings: List<Float>,
)

// ── NpuProfiler ─────────────────────────────────────────────────────────────────

class NpuProfiler(private val context: Context) {

    companion object {
        private const val TAG = "NpuProfiler"

        // NNAPI device type constants (mirrors NeuralNetworksDevice)
        const val DEVICE_TYPE_OTHER = 1
        const val DEVICE_TYPE_CPU = 2
        const val DEVICE_TYPE_GPU = 3
        const val DEVICE_TYPE_ACCELERATOR = 4

        private fun typeToName(type: Int): String = when (type) {
            DEVICE_TYPE_CPU         -> "CPU"
            DEVICE_TYPE_GPU         -> "GPU"
            DEVICE_TYPE_ACCELERATOR -> "ACCELERATOR"
            else                    -> "OTHER"
        }
    }

    // Cache the root check — it involves spawning a process.
    @Volatile
    private var rootChecked = false

    @Volatile
    private var rootAvailable = false

    // ── 1. NPU Hardware Enumeration (no root) ──────────────────────────────

    /**
     * Detect hardware accelerators by probing well-known device/sysfs nodes.
     * Always returns at least a CPU entry.
     */
    fun detectNpuHardware(): List<NnDevice> {
        val devices = mutableListOf<NnDevice>()

        // Always add CPU
        devices.add(
            NnDevice(
                name = "cpu",
                type = DEVICE_TYPE_CPU,
                typeName = "CPU",
                version = Build.HARDWARE,
            )
        )

        // Qualcomm Hexagon DSP / HTA
        if (fileExists("/sys/class/devfreq/soc:qcom,cdsp-cdsp-l3-lat") ||
            fileExists("/dev/cdsp") ||
            fileExists("/sys/devices/platform/soc/soc:qcom,npu") ||
            fileExists("/dev/adsprpc-smd")) {
            devices.add(
                NnDevice(
                    name = "qti-hta",
                    type = DEVICE_TYPE_ACCELERATOR,
                    typeName = "ACCELERATOR",
                    version = "Qualcomm Hexagon DSP",
                )
            )
        }

        // MediaTek MDLA / APU
        if (fileExists("/sys/devices/platform/soc/soc:mdla") ||
            fileExists("/dev/mdla0") ||
            fileExists("/dev/apusys") ||
            fileExists("/sys/devices/platform/soc/soc:apusys")) {
            devices.add(
                NnDevice(
                    name = "mtk-neuron",
                    type = DEVICE_TYPE_ACCELERATOR,
                    typeName = "ACCELERATOR",
                    version = "MediaTek MDLA",
                )
            )
        }

        // Samsung NPU (Exynos)
        if (fileExists("/dev/vertex0") ||
            fileExists("/dev/npu0") ||
            fileExists("/sys/devices/platform/exynos-sci")) {
            devices.add(
                NnDevice(
                    name = "samsung-npu",
                    type = DEVICE_TYPE_ACCELERATOR,
                    typeName = "ACCELERATOR",
                    version = "Samsung NPU",
                )
            )
        }

        // Google Edge TPU (Tensor SoC)
        if (fileExists("/dev/edgetpu") ||
            fileExists("/sys/devices/platform/edgetpu-gsa")) {
            devices.add(
                NnDevice(
                    name = "google-edgetpu",
                    type = DEVICE_TYPE_ACCELERATOR,
                    typeName = "ACCELERATOR",
                    version = "Google Edge TPU",
                )
            )
        }

        // GPU — Mali or Adreno
        val hasMali = fileExists("/dev/mali0") || fileExists("/dev/mali") ||
            fileExists("/sys/devices/platform/mali")
        val hasAdreno = fileExists("/dev/kgsl-3d0")
        if (hasMali || hasAdreno) {
            val gpuName = when {
                hasAdreno -> "Adreno GPU"
                hasMali   -> "Mali GPU"
                else      -> "GPU"
            }
            devices.add(
                NnDevice(
                    name = "gpu",
                    type = DEVICE_TYPE_GPU,
                    typeName = "GPU",
                    version = gpuName,
                )
            )
        }

        return devices
    }

    // ── 2. Root Detection ───────────────────────────────────────────────────

    /**
     * Check if we can execute commands as root.
     * Result is cached after the first call.
     */
    fun isRootAvailable(): Boolean {
        if (rootChecked) return rootAvailable
        rootAvailable = try {
            val process = Runtime.getRuntime().exec(arrayOf("su", "-c", "id"))
            val output = process.inputStream.bufferedReader().readLine() ?: ""
            process.waitFor(3, TimeUnit.SECONDS)
            output.contains("uid=0")
        } catch (e: Exception) {
            Log.d(TAG, "Root not available: ${e.message}")
            false
        }
        rootChecked = true
        return rootAvailable
    }

    // ── 3. Root-Mode NPU Metrics ────────────────────────────────────────────

    /**
     * Read real-time NPU metrics from vendor sysfs paths.
     * Requires root — returns zeroed metrics when root is unavailable.
     */
    fun readNpuMetrics(): NpuMetrics {
        if (!isRootAvailable()) {
            return NpuMetrics(0, 0f, 0f, 0, false)
        }

        val chipVendor = detectChipVendor()
        return when (chipVendor) {
            "qualcomm" -> readQualcommNpu()
            "mediatek" -> readMediatekNpu()
            "samsung"  -> readSamsungNpu()
            "google"   -> readGoogleNpu()
            else       -> NpuMetrics(0, 0f, 0f, 0, true)
        }
    }

    private fun readQualcommNpu(): NpuMetrics {
        // CDSP frequency (Hexagon compute DSP — runs NPU workloads)
        val freqKhz = readRootFileInt("/sys/class/devfreq/soc:qcom,cdsp-cdsp-l3-lat/cur_freq")
            ?: readRootFileInt("/sys/devices/platform/soc/soc:qcom,npu/devfreq/soc:qcom,npu/cur_freq")
            ?: 0

        // NPU power level (lower = higher performance)
        val powerLevel = readRootFileInt("/sys/kernel/debug/msm_npu/npu_power_level")
            ?: readRootFileInt("/sys/devices/platform/soc/soc:qcom,npu/power_level")
            ?: 0

        // Estimate utilization from CDSP busy time if available
        val utilization = readRootFileFloat(
            "/sys/class/devfreq/soc:qcom,cdsp-cdsp-l3-lat/load"
        ) ?: estimateUtilFromFreq(freqKhz)

        // DRAM / memory-latency frequency as proxy for memory bandwidth utilization
        val dramFreqKhz = readRootFileInt("/sys/class/devfreq/soc:qcom,memlat-cpu0/cur_freq") ?: 0
        val dramUtil = estimateUtilFromFreq(dramFreqKhz)

        return NpuMetrics(
            npuFrequencyMhz = freqKhz / 1000,
            npuUtilizationPercent = utilization,
            dramUtilizationPercent = dramUtil,
            npuPowerLevel = powerLevel,
            isRootAvailable = true,
        )
    }

    private fun readMediatekNpu(): NpuMetrics {
        // MediaTek MDLA / APU paths
        val utilization = readRootFileFloat("/sys/devices/platform/soc/soc:mdla/utilization")
            ?: readRootFileFloat("/d/mdla/utilization")
            ?: 0f

        val freqKhz = readRootFileInt("/sys/devices/platform/soc/soc:mdla/frequency")
            ?: readRootFileInt("/sys/devices/platform/soc/soc:apusys/frequency")
            ?: 0

        return NpuMetrics(
            npuFrequencyMhz = if (freqKhz > 10_000) freqKhz / 1000 else freqKhz,
            npuUtilizationPercent = utilization,
            dramUtilizationPercent = 0f,
            npuPowerLevel = 0,
            isRootAvailable = true,
        )
    }

    private fun readSamsungNpu(): NpuMetrics {
        // Samsung Exynos NPU paths
        val utilization = readRootFileFloat("/sys/devices/platform/exynos-sci/npu_utilization")
            ?: readRootFileFloat("/sys/kernel/debug/npu/utilization")
            ?: 0f

        val freqKhz = readRootFileInt("/sys/devices/platform/exynos-sci/npu_freq")
            ?: readRootFileInt("/sys/kernel/debug/npu/frequency")
            ?: 0

        return NpuMetrics(
            npuFrequencyMhz = if (freqKhz > 10_000) freqKhz / 1000 else freqKhz,
            npuUtilizationPercent = utilization,
            dramUtilizationPercent = 0f,
            npuPowerLevel = 0,
            isRootAvailable = true,
        )
    }

    private fun readGoogleNpu(): NpuMetrics {
        // Google Tensor Edge TPU paths
        val utilization = readRootFileFloat("/sys/devices/platform/edgetpu-gsa/utilization")
            ?: 0f

        val freqKhz = readRootFileInt("/sys/devices/platform/edgetpu-gsa/frequency") ?: 0

        return NpuMetrics(
            npuFrequencyMhz = if (freqKhz > 10_000) freqKhz / 1000 else freqKhz,
            npuUtilizationPercent = utilization,
            dramUtilizationPercent = 0f,
            npuPowerLevel = 0,
            isRootAvailable = true,
        )
    }

    // ── Root file helpers ───────────────────────────────────────────────────

    private fun readRootFile(path: String): String? {
        return try {
            val process = Runtime.getRuntime().exec(arrayOf("su", "-c", "cat $path"))
            val output = process.inputStream.bufferedReader().readText()
            process.waitFor(2, TimeUnit.SECONDS)
            if (process.exitValue() == 0) output.trim() else null
        } catch (e: Exception) {
            null
        }
    }

    private fun readRootFileInt(path: String): Int? {
        return readRootFile(path)?.trim()?.toLongOrNull()?.toInt()
    }

    private fun readRootFileFloat(path: String): Float? {
        return readRootFile(path)?.trim()?.toFloatOrNull()
    }

    /**
     * Rough utilization estimate from frequency.
     * When the governor scales frequency up, the device is busier.
     * This is a heuristic — real utilization needs vendor counters.
     */
    private fun estimateUtilFromFreq(currentFreqKhz: Int): Float {
        if (currentFreqKhz <= 0) return 0f
        // Assume a typical max around 1.5 GHz for CDSP / NPU
        val maxFreqKhz = 1_500_000
        return ((currentFreqKhz.toFloat() / maxFreqKhz) * 100f).coerceIn(0f, 100f)
    }

    private fun fileExists(path: String): Boolean {
        return try {
            File(path).exists()
        } catch (e: SecurityException) {
            false
        }
    }

    /** Reuse vendor detection logic consistent with [DeviceDetector]. */
    private fun detectChipVendor(): String {
        val lower = Build.HARDWARE.lowercase() + " " + Build.BOARD.lowercase()
        return when {
            "qcom" in lower || "sm" in lower || "msm" in lower || "sdm" in lower -> "qualcomm"
            "mt" in lower || "mediatek" in lower -> "mediatek"
            "exynos" in lower || ("samsung" in lower && "slsi" in lower) -> "samsung"
            "tensor" in lower || "gs" in lower -> "google"
            else -> "unknown"
        }
    }
}

// ── 4. Real-time NPU Monitor (root only) ───────────────────────────────────

/**
 * Monitors NPU utilization during a benchmark window.
 *
 * Usage:
 * ```
 * val monitor = NpuMonitor(profiler)
 * monitor.startMonitoring()
 * // ... run benchmark ...
 * val result = monitor.stopMonitoring()
 * // result.avgUtilizationPercent, result.peakUtilizationPercent
 * ```
 */
class NpuMonitor(private val profiler: NpuProfiler) {

    companion object {
        private const val SAMPLE_INTERVAL_MS = 100L
    }

    @Volatile
    private var monitoring = false
    private val readings = mutableListOf<Float>()
    private var monitorThread: Thread? = null

    /**
     * Start sampling NPU utilization in the background.
     * No-op if root is unavailable.
     */
    fun startMonitoring() {
        if (!profiler.isRootAvailable()) return
        readings.clear()
        monitoring = true
        monitorThread = Thread({
            while (monitoring) {
                try {
                    val metrics = profiler.readNpuMetrics()
                    if (metrics.npuUtilizationPercent > 0f) {
                        synchronized(readings) {
                            readings.add(metrics.npuUtilizationPercent)
                        }
                    }
                    Thread.sleep(SAMPLE_INTERVAL_MS)
                } catch (_: InterruptedException) {
                    break
                } catch (_: Exception) {
                    // Ignore transient read failures
                }
            }
        }, "NpuMonitor")
        monitorThread?.isDaemon = true
        monitorThread?.start()
    }

    /**
     * Stop sampling and return aggregated utilization data.
     * Safe to call even if monitoring was never started.
     */
    fun stopMonitoring(): NpuMonitorResult {
        monitoring = false
        monitorThread?.join(1_000)
        monitorThread = null

        val snapshot = synchronized(readings) { readings.toList() }
        val avg = if (snapshot.isNotEmpty()) snapshot.sum() / snapshot.size else 0f
        val peak = snapshot.maxOrNull() ?: 0f

        return NpuMonitorResult(
            avgUtilizationPercent = avg,
            peakUtilizationPercent = peak,
            readings = snapshot,
        )
    }
}
