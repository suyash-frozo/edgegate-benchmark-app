package ai.frozo.edgegate.benchmark

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * HTTP client for EdgeGate API.
 *
 * Fetches models from EdgeGate, downloads artifacts, and reports
 * benchmark results back — completing the loop that Qualcomm AI Hub
 * does but on any device.
 */
class EdgeGateClient(
    private val baseUrl: String,
    private val apiKey: String,
    private val workspaceId: String,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val jsonMediaType = "application/json".toMediaType()

    data class ModelInfo(
        val id: String,
        val kind: String,
        val original_filename: String?,
        val sha256: String,
        val size_bytes: Long,
    )

    data class PipelineInfo(
        val id: String,
        val name: String,
    )

    /**
     * Test connection to EdgeGate API.
     */
    fun testConnection(): Result<String> {
        val request = Request.Builder()
            .url("$baseUrl/v1/workspaces/$workspaceId")
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val body = response.body?.string() ?: "{}"
                val map = gson.fromJson(body, Map::class.java)
                Result.success(map["name"]?.toString() ?: "Connected")
            } else {
                Result.failure(IOException("HTTP ${response.code}: ${response.body?.string()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * List model artifacts in the workspace.
     */
    fun listModels(): Result<List<ModelInfo>> {
        val request = Request.Builder()
            .url("$baseUrl/v1/workspaces/$workspaceId/artifacts?kind=model")
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                val body = response.body?.string() ?: "[]"
                val type = object : TypeToken<List<ModelInfo>>() {}.type
                val models: List<ModelInfo> = gson.fromJson(body, type)
                Result.success(models)
            } else {
                Result.failure(IOException("HTTP ${response.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Download a model artifact to local storage.
     */
    fun downloadModel(artifactId: String, destFile: File): Result<File> {
        // First try to get presigned URL
        val urlRequest = Request.Builder()
            .url("$baseUrl/v1/workspaces/$workspaceId/bundles/$artifactId/artifact-url")
            .addHeader("Authorization", "Bearer $apiKey")
            .build()

        return try {
            val urlResponse = client.newCall(urlRequest).execute()
            val downloadUrl = if (urlResponse.isSuccessful) {
                val body = gson.fromJson(urlResponse.body?.string(), Map::class.java)
                body["url"]?.toString() ?: "$baseUrl/v1/workspaces/$workspaceId/bundles/$artifactId/artifact"
            } else {
                // Fallback to direct artifact download
                "$baseUrl/v1/workspaces/$workspaceId/artifacts/$artifactId"
            }

            // Download
            val dlRequest = Request.Builder()
                .url(downloadUrl)
                .addHeader("Authorization", "Bearer $apiKey")
                .build()

            val dlResponse = client.newCall(dlRequest).execute()
            if (dlResponse.isSuccessful) {
                dlResponse.body?.byteStream()?.use { input ->
                    FileOutputStream(destFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Result.success(destFile)
            } else {
                Result.failure(IOException("Download failed: HTTP ${dlResponse.code}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Report benchmark results back to EdgeGate.
     *
     * This is equivalent to what Qualcomm AI Hub reports after a profile job —
     * inference_time_ms, peak_memory_mb, compute_unit_breakdown.
     */
    fun reportBenchmark(
        device: DeviceInfo,
        modelName: String,
        results: List<DelegateResult>,
    ): Result<String> {
        val best = results.filter { it.success }.minByOrNull { it.inferenceTimeMs }
        val cpuResult = results.find { it.delegate == "cpu" && it.success }

        // Build payload matching EdgeGate's normalized metrics format
        val payload = mapOf(
            "device" to mapOf(
                "manufacturer" to device.manufacturer,
                "model" to device.model,
                "chip" to device.chip,
                "chip_vendor" to device.chipVendor,
                "gpu" to device.gpu,
                "npu" to device.npu,
                "android_version" to device.androidVersion,
                "sdk_version" to device.sdkVersion,
                "abi" to device.abi,
                "ram_gb" to device.ramGb,
            ),
            "model_name" to modelName,
            "benchmark_results" to results.map { r ->
                mapOf(
                    "delegate" to r.delegate,
                    "inference_time_ms" to r.inferenceTimeMs,
                    "p50_ms" to r.p50Ms,
                    "p95_ms" to r.p95Ms,
                    "p99_ms" to r.p99Ms,
                    "peak_memory_mb" to r.peakMemoryMb,
                    "init_time_ms" to r.initTimeMs,
                    "warmup_runs" to r.warmupRuns,
                    "measurement_runs" to r.measurementRuns,
                    "success" to r.success,
                    "error" to r.error,
                )
            },
            // Normalized metrics (same format as AI Hub) — use the best delegate
            "normalized_metrics" to if (best != null) mapOf(
                "inference_time_ms" to best.inferenceTimeMs,
                "peak_memory_mb" to best.peakMemoryMb,
                "throughput_fps" to if (best.inferenceTimeMs > 0) 1000f / best.inferenceTimeMs else 0f,
                "best_delegate" to best.delegate,
                "compute_unit_breakdown" to mapOf(
                    "npu" to if (best.delegate == "nnapi") 100 else 0,
                    "gpu" to if (best.delegate == "gpu") 100 else 0,
                    "cpu" to if (best.delegate == "cpu") 100 else 0,
                ),
                "speedup_vs_cpu" to if (cpuResult != null && best.inferenceTimeMs > 0)
                    cpuResult.inferenceTimeMs / best.inferenceTimeMs else 1f,
            ) else emptyMap<String, Any>(),
        )

        val json = gson.toJson(payload)
        val request = Request.Builder()
            .url("$baseUrl/v1/workspaces/$workspaceId/device-benchmarks")
            .addHeader("Authorization", "Bearer $apiKey")
            .addHeader("Content-Type", "application/json")
            .post(json.toRequestBody(jsonMediaType))
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                Result.success("Reported to EdgeGate")
            } else {
                // Even if the endpoint doesn't exist yet, save locally
                Result.failure(IOException("HTTP ${response.code}: ${response.body?.string()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
