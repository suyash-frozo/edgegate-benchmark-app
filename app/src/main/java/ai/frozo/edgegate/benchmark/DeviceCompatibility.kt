package ai.frozo.edgegate.benchmark

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.PowerManager
import android.os.StatFs
import java.io.File

// ── Data classes ────────────────────────────────────────────────────────────────

data class ModelRequirement(
    val name: String,
    val sizeGb: Float,
    val minRamGb: Float,
    val format: String, // "gguf", "tflite", "onnx"
    val supportsGpu: Boolean,
    val supportsNnapi: Boolean,
)

enum class CompatStatus { COMPATIBLE, TIGHT, INCOMPATIBLE }

data class ModelCompatResult(
    val model: ModelRequirement,
    val status: CompatStatus,
    val reason: String,
)

data class DelegateSupport(
    val supported: Boolean,
    val detail: String,
)

data class DeviceReport(
    val device: DeviceInfo,
    val ramTotalGb: Float,
    val ramAvailableGb: Float,
    val storageFreeGb: Float,
    val gpuModel: String,
    val vulkanVersion: String?,
    val npuType: String,
    val nnapiVersion: Int,
    val thermalStatus: String,
    val batteryLevel: Int,
    val isCharging: Boolean,
    val cpuCores: Int,
    val cpuMaxFreqMhz: Int,
    val modelCompatibility: List<ModelCompatResult>,
    val delegateSupport: Map<String, DelegateSupport>,
    val warnings: List<String>,
    val readyToBenchmark: Boolean,
    // NPU hardware profiling
    val nnDevices: List<NnDevice> = emptyList(),
    val rootAvailable: Boolean = false,
    val npuMetrics: NpuMetrics? = null,
)

// ── Main checker ────────────────────────────────────────────────────────────────

object DeviceCompatibility {

    // Known model catalog
    private val KNOWN_MODELS = listOf(
        ModelRequirement("Gemma 2B Q4",        1.5f,  2.5f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Qwen2.5 1.5B Q4",    1.0f,  2.0f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Llama 3.2 1B Q4",    0.7f,  1.5f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Phi-3 Mini Q4",      2.3f,  3.5f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Llama 3.2 3B Q4",    1.8f,  3.0f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Gemma 7B Q4",        4.5f,  6.0f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("Llama 3.1 8B Q4",    5.0f,  7.0f, "gguf",   supportsGpu = true,  supportsNnapi = false),
        ModelRequirement("MobileNet v1 quant", 0.004f,0.1f, "tflite", supportsGpu = true,  supportsNnapi = true),
        ModelRequirement("MobileNet v2 FP32",  0.014f,0.1f, "onnx",   supportsGpu = false, supportsNnapi = true),
        ModelRequirement("YOLOv8n",            0.012f,0.2f, "onnx",   supportsGpu = false, supportsNnapi = true),
    )

    /**
     * Run a full device analysis and return a comprehensive report.
     */
    fun analyze(context: Context): DeviceReport {
        val deviceInfo = DeviceDetector.detect(context)

        // Hardware details
        val (ramTotal, ramAvailable) = getRamInfo(context)
        val storageFree = getStorageFreeGb()
        val gpuModel = deviceInfo.gpu
        val vulkanVersion = getVulkanVersion(context)
        val npuType = deviceInfo.npu
        val nnapiVersion = getNnapiVersion()
        val thermalStatus = getThermalStatus(context)
        val (batteryLevel, isCharging) = getBatteryInfo(context)
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val cpuMaxFreq = getCpuMaxFreqMhz()

        // NPU hardware profiling
        val npuProfiler = NpuProfiler(context)
        val nnDevices = npuProfiler.detectNpuHardware()
        val rootAvailable = npuProfiler.isRootAvailable()
        val npuMetrics = if (rootAvailable) npuProfiler.readNpuMetrics() else null

        // Delegate support
        val delegateSupport = buildDelegateSupport(
            deviceInfo, vulkanVersion, cpuCores
        )

        // Model compatibility
        val modelCompat = KNOWN_MODELS.map { model ->
            checkModelCompat(model, ramAvailable, storageFree)
        }

        // Warnings
        val warnings = buildWarnings(
            batteryLevel, thermalStatus, ramAvailable, ramTotal
        )

        val readyToBenchmark = warnings.none { it.startsWith("[CRITICAL]") }

        return DeviceReport(
            device = deviceInfo,
            ramTotalGb = ramTotal,
            ramAvailableGb = ramAvailable,
            storageFreeGb = storageFree,
            gpuModel = gpuModel,
            vulkanVersion = vulkanVersion,
            npuType = npuType,
            nnapiVersion = nnapiVersion,
            thermalStatus = thermalStatus,
            batteryLevel = batteryLevel,
            isCharging = isCharging,
            cpuCores = cpuCores,
            cpuMaxFreqMhz = cpuMaxFreq,
            modelCompatibility = modelCompat,
            delegateSupport = delegateSupport,
            warnings = warnings,
            readyToBenchmark = readyToBenchmark,
            nnDevices = nnDevices,
            rootAvailable = rootAvailable,
            npuMetrics = npuMetrics,
        )
    }

    // ── Hardware detection helpers ──────────────────────────────────────────

    private fun getRamInfo(context: Context): Pair<Float, Float> {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(memInfo)
        val totalGb = memInfo.totalMem / (1024f * 1024f * 1024f)
        val availGb = memInfo.availMem / (1024f * 1024f * 1024f)
        return totalGb to availGb
    }

    private fun getStorageFreeGb(): Float {
        return try {
            val stat = StatFs(Environment.getDataDirectory().path)
            val freeBytes = stat.availableBlocksLong * stat.blockSizeLong
            freeBytes / (1024f * 1024f * 1024f)
        } catch (e: Exception) {
            0f
        }
    }

    private fun getVulkanVersion(context: Context): String? {
        val pm = context.packageManager
        return when {
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x00401000) -> "1.1"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x00400000) -> "1.0"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL, 1) -> "1.1"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL, 0) -> "1.0"
            else -> null
        }
    }

    private fun getNnapiVersion(): Int {
        // NNAPI feature level maps from SDK_INT
        // API 27 = NNAPI 1.0, API 28 = 1.1, API 29 = 1.2, API 30 = 1.3
        // API 31 = NNAPI 5, API 33 = NNAPI 7, API 34 = NNAPI 8
        val sdk = Build.VERSION.SDK_INT
        return when {
            sdk >= 34 -> 8
            sdk >= 33 -> 7
            sdk >= 32 -> 6
            sdk >= 31 -> 5
            sdk >= 30 -> 4  // NNAPI 1.3
            sdk >= 29 -> 3  // NNAPI 1.2
            sdk >= 28 -> 2  // NNAPI 1.1
            sdk >= 27 -> 1  // NNAPI 1.0
            else -> 0       // No NNAPI
        }
    }

    private fun getThermalStatus(context: Context): String {
        if (Build.VERSION.SDK_INT >= 29) {
            try {
                val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
                return when (pm.currentThermalStatus) {
                    PowerManager.THERMAL_STATUS_NONE -> "Normal"
                    PowerManager.THERMAL_STATUS_LIGHT -> "Light"
                    PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
                    PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
                    PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                    PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                    PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                    else -> "Unknown"
                }
            } catch (_: Exception) { }
        }
        return "Unknown (API < 29)"
    }

    private fun getBatteryInfo(context: Context): Pair<Int, Boolean> {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val batteryIntent = context.registerReceiver(null, filter)
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batteryPct = if (scale > 0) (level * 100) / scale else -1

        val plugged = batteryIntent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val isCharging = plugged != 0

        return batteryPct to isCharging
    }

    private fun getCpuMaxFreqMhz(): Int {
        // Read max frequency from the first CPU core via sysfs
        return try {
            val freq = File("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
                .readText().trim().toLongOrNull() ?: 0L
            (freq / 1000).toInt() // kHz -> MHz
        } catch (_: Exception) {
            // Fallback: try all cores, pick the highest
            try {
                val cores = Runtime.getRuntime().availableProcessors()
                var maxFreq = 0L
                for (i in 0 until cores) {
                    val f = File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq")
                    if (f.exists()) {
                        val v = f.readText().trim().toLongOrNull() ?: 0L
                        if (v > maxFreq) maxFreq = v
                    }
                }
                (maxFreq / 1000).toInt()
            } catch (_: Exception) {
                0
            }
        }
    }

    // ── Model compatibility ─────────────────────────────────────────────────

    private fun checkModelCompat(
        model: ModelRequirement,
        availRamGb: Float,
        storageFreeGb: Float,
    ): ModelCompatResult {
        // Check storage first
        if (storageFreeGb < model.sizeGb) {
            return ModelCompatResult(
                model, CompatStatus.INCOMPATIBLE,
                "not enough storage (need ${"%.1f".format(model.sizeGb)} GB, have ${"%.1f".format(storageFreeGb)} GB)"
            )
        }

        // Check RAM
        return when {
            availRamGb >= model.minRamGb * 1.3f -> ModelCompatResult(
                model, CompatStatus.COMPATIBLE,
                "fits easily"
            )
            availRamGb >= model.minRamGb -> ModelCompatResult(
                model, CompatStatus.TIGHT,
                "tight, may OOM"
            )
            else -> ModelCompatResult(
                model, CompatStatus.INCOMPATIBLE,
                "not enough RAM (need ${"%.1f".format(model.minRamGb)} GB, have ${"%.1f".format(availRamGb)} GB)"
            )
        }
    }

    // ── Delegate support ────────────────────────────────────────────────────

    private fun buildDelegateSupport(
        device: DeviceInfo,
        vulkanVersion: String?,
        cpuCores: Int,
    ): Map<String, DelegateSupport> {
        val map = LinkedHashMap<String, DelegateSupport>()

        // CPU — always available
        map["CPU"] = DelegateSupport(
            supported = true,
            detail = "${cpuCores}x ARM cores (4 threads)"
        )

        // GPU — needs Vulkan >= 1.0 or GLES 3.1
        val gpuSupported = vulkanVersion != null
        map["GPU"] = if (gpuSupported) {
            DelegateSupport(true, "${device.gpu} (Vulkan $vulkanVersion)")
        } else {
            DelegateSupport(false, "No Vulkan support detected")
        }

        // NNAPI — SDK >= 27
        val nnapiSupported = device.sdkVersion >= 27
        map["NNAPI"] = if (nnapiSupported) {
            DelegateSupport(true, "${device.npu}")
        } else {
            DelegateSupport(false, "Requires Android 8.1+ (API 27)")
        }

        // QNN — Qualcomm only
        val isQualcomm = device.chipVendor == "qualcomm"
        map["QNN"] = if (isQualcomm) {
            DelegateSupport(true, "Qualcomm Hexagon DSP via QNN SDK")
        } else {
            DelegateSupport(false, "Not a Qualcomm device")
        }

        return map
    }

    // ── Pre-flight warnings ─────────────────────────────────────────────────

    private fun buildWarnings(
        batteryLevel: Int,
        thermalStatus: String,
        ramAvailableGb: Float,
        ramTotalGb: Float,
    ): List<String> {
        val warnings = mutableListOf<String>()

        if (batteryLevel in 0..19) {
            warnings.add("Battery at $batteryLevel% — results may be throttled")
        }

        if (thermalStatus in listOf("Moderate", "Severe", "Critical", "Emergency", "Shutdown")) {
            warnings.add("[CRITICAL] Device is hot ($thermalStatus) — cool down first")
        }

        if (ramAvailableGb < 1.0f) {
            warnings.add("Available RAM is low (${"%.1f".format(ramAvailableGb)} GB) — close other apps")
        }

        val ramUsedRatio = if (ramTotalGb > 0) (ramTotalGb - ramAvailableGb) / ramTotalGb else 0f
        if (ramUsedRatio > 0.50f) {
            warnings.add(
                "Background apps using ${"%.0f".format(ramUsedRatio * 100)}%% of RAM — consider freeing memory"
            )
        }

        return warnings
    }

    // ── Formatted report ────────────────────────────────────────────────────

    fun formatReport(report: DeviceReport): String = buildString {
        val d = report.device

        // Device profile
        appendLine("\u2550\u2550\u2550 DEVICE PROFILE \u2550\u2550\u2550")
        appendLine("${d.manufacturer} ${d.model} (${d.chip})")
        appendLine("Android ${d.androidVersion} | API ${d.sdkVersion} | ${d.abi}")
        appendLine()

        // Hardware
        appendLine("\u2550\u2550\u2550 HARDWARE \u2550\u2550\u2550")
        appendLine("RAM:     ${"%.1f".format(report.ramTotalGb)} GB total | ${"%.1f".format(report.ramAvailableGb)} GB available")
        appendLine("Storage: ${"%.0f".format(report.storageFreeGb)} GB free")
        appendLine("CPU:     ${report.cpuCores} cores" +
            if (report.cpuMaxFreqMhz > 0) " @ ${report.cpuMaxFreqMhz} MHz" else "")
        appendLine("GPU:     ${report.gpuModel}" +
            if (report.vulkanVersion != null) " (Vulkan ${report.vulkanVersion})" else " (no Vulkan)")
        appendLine("NPU:     ${report.npuType} (NNAPI v${report.nnapiVersion})")
        appendLine("Battery: ${report.batteryLevel}%" +
            if (report.isCharging) " (charging)" else "")
        appendLine("Thermal: ${report.thermalStatus}")
        appendLine()

        // Hardware accelerators (from NpuProfiler)
        appendLine("\u2550\u2550\u2550 HARDWARE ACCELERATORS \u2550\u2550\u2550")
        for (dev in report.nnDevices) {
            appendLine("\u2705 ${dev.typeName} \u2014 ${dev.version}")
        }
        if (report.rootAvailable && report.npuMetrics != null) {
            val m = report.npuMetrics
            if (m.npuFrequencyMhz > 0) {
                appendLine("   Frequency: ${m.npuFrequencyMhz} MHz")
            }
            appendLine("   Utilization: ${"%.0f".format(m.npuUtilizationPercent)}%" +
                if (m.npuUtilizationPercent == 0f) " (idle)" else "")
            appendLine("Root: Available \u2713 (real NPU metrics enabled)")
        } else {
            appendLine("Root: Not available (enable for real NPU utilization %)")
        }
        appendLine()

        // Model compatibility
        appendLine("\u2550\u2550\u2550 MODEL COMPATIBILITY \u2550\u2550\u2550")
        for (mc in report.modelCompatibility) {
            val icon = when (mc.status) {
                CompatStatus.COMPATIBLE   -> "\u2705"
                CompatStatus.TIGHT        -> "\u26a0\ufe0f"
                CompatStatus.INCOMPATIBLE -> "\u274c"
            }
            val size = if (mc.model.sizeGb >= 1.0f)
                "${"%.1f".format(mc.model.sizeGb)} GB"
            else
                "${(mc.model.sizeGb * 1024).toInt()} MB"
            appendLine("$icon %-22s %8s \u2014 %s".format(
                mc.model.name, size, mc.reason
            ))
        }
        appendLine()

        // Delegates
        appendLine("\u2550\u2550\u2550 DELEGATES \u2550\u2550\u2550")
        for ((name, ds) in report.delegateSupport) {
            val icon = if (ds.supported) "\u2705" else "\u274c"
            appendLine("$icon $name \u2014 ${ds.detail}")
        }
        appendLine()

        // Warnings
        appendLine("\u2550\u2550\u2550 WARNINGS \u2550\u2550\u2550")
        if (report.warnings.isEmpty()) {
            appendLine("None \u2014 ready to benchmark!")
        } else {
            for (w in report.warnings) {
                appendLine("\u26a0 $w")
            }
        }
    }
}
