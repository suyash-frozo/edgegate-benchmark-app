package ai.frozo.edgegate.benchmark

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import java.io.File

/**
 * Detects device hardware — chip vendor, model, GPU, NPU capabilities.
 * Matches the same device info that Qualcomm AI Hub reports.
 */
data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val chip: String,
    val chipVendor: String,
    val gpu: String,
    val npu: String,
    val androidVersion: String,
    val sdkVersion: Int,
    val abi: String,
    val ramGb: Float,
    val board: String,
)

object DeviceDetector {

    fun detect(context: Context): DeviceInfo {
        val chip = detectChip()
        val chipVendor = detectChipVendor(chip)
        val gpu = detectGpu()
        val npu = detectNpu(chipVendor, chip)
        val ram = detectRam(context)

        return DeviceInfo(
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            chip = chip,
            chipVendor = chipVendor,
            gpu = gpu,
            npu = npu,
            androidVersion = Build.VERSION.RELEASE,
            sdkVersion = Build.VERSION.SDK_INT,
            abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            ramGb = ram,
            board = Build.BOARD,
        )
    }

    private fun detectChip(): String {
        // Try SOC model first (most reliable on newer Android)
        val socModel = getSystemProp("ro.soc.model")
        if (socModel.isNotEmpty()) return socModel

        // Fallback to hardware
        val hardware = getSystemProp("ro.hardware")
        val board = Build.BOARD

        return when {
            hardware.isNotEmpty() -> hardware
            board.isNotEmpty() -> board
            else -> Build.HARDWARE
        }
    }

    private fun detectChipVendor(chip: String): String {
        val lower = chip.lowercase() + " " + Build.HARDWARE.lowercase() + " " + Build.BOARD.lowercase()
        return when {
            "qcom" in lower || "sm" in lower || "msm" in lower || "sdm" in lower -> "qualcomm"
            "mt" in lower || "mediatek" in lower -> "mediatek"
            "exynos" in lower || "samsung" in lower && "slsi" in lower -> "samsung"
            "tensor" in lower || "gs" in lower -> "google"
            "kirin" in lower || "hi" in lower -> "huawei"
            "unisoc" in lower || "sc" in lower -> "unisoc"
            else -> "unknown"
        }
    }

    private fun detectGpu(): String {
        // GPU info is best detected at GL context creation time.
        // Return the board platform as a proxy.
        val board = Build.BOARD.lowercase()
        return when {
            "qcom" in board || "sm" in board -> "Qualcomm Adreno"
            "mt" in board || "mediatek" in board -> "ARM Mali (MediaTek)"
            "exynos" in board -> "ARM Mali (Samsung)"
            "tensor" in board || "gs" in board -> "ARM Mali (Google)"
            else -> "GPU (${Build.BOARD})"
        }
    }

    private fun detectNpu(vendor: String, chip: String): String {
        return when (vendor) {
            "qualcomm" -> "Qualcomm Hexagon DSP/NPU"
            "mediatek" -> "MediaTek APU"
            "samsung" -> "Samsung NPU"
            "google" -> "Google Edge TPU"
            "huawei" -> "Huawei Da Vinci NPU"
            else -> "NNAPI (generic)"
        }
    }

    private fun detectRam(context: Context): Float {
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memInfo)
        return memInfo.totalMem / (1024f * 1024f * 1024f)
    }

    private fun getSystemProp(key: String): String {
        return try {
            val process = Runtime.getRuntime().exec(arrayOf("getprop", key))
            process.inputStream.bufferedReader().readLine()?.trim() ?: ""
        } catch (e: Exception) {
            ""
        }
    }

    fun formatDeviceInfo(info: DeviceInfo): String {
        return buildString {
            appendLine("${info.manufacturer} ${info.model}")
            appendLine("Chip:    ${info.chip} (${info.chipVendor})")
            appendLine("GPU:     ${info.gpu}")
            appendLine("NPU:     ${info.npu}")
            appendLine("Android: ${info.androidVersion} (SDK ${info.sdkVersion})")
            appendLine("RAM:     ${"%.1f".format(info.ramGb)} GB")
            appendLine("ABI:     ${info.abi}")
        }
    }
}
