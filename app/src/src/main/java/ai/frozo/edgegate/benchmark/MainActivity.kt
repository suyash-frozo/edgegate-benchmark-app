package ai.frozo.edgegate.benchmark

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

/**
 * EdgeGate Benchmark — Main Activity.
 *
 * Flow:
 * 1. Auto-detect device hardware (chip, GPU, NPU)
 * 2. Connect to EdgeGate API → list available models
 * 3. Select model → download TFLite artifact
 * 4. Run benchmark on CPU, GPU, NNAPI (real NPU/DSP)
 * 5. Show results (same metrics as Qualcomm AI Hub)
 * 6. Report back to EdgeGate
 */
class MainActivity : AppCompatActivity(), BenchmarkEngine.ProgressCallback {

    companion object {
        // Hardcoded for internal testing — not for distribution
        const val DEFAULT_SERVER_URL = "https://edgegateapi.frozo.ai"
        const val DEFAULT_WORKSPACE_ID = "7479eff9-985a-49c0-b3a0-5e5b3885a5a7"
    }

    private lateinit var deviceInfo: DeviceInfo
    private lateinit var engine: BenchmarkEngine
    private var client: EdgeGateClient? = null
    private var benchmarkResults: List<DelegateResult>? = null
    private var currentModelName: String = ""

    // Views
    private lateinit var tvDeviceInfo: TextView
    private lateinit var etServerUrl: EditText
    private lateinit var etApiKey: EditText
    private lateinit var etWorkspaceId: EditText
    private lateinit var btnConnect: Button
    private lateinit var cardModels: LinearLayout
    private lateinit var spinnerModels: Spinner
    private lateinit var cbCpu: CheckBox
    private lateinit var cbGpu: CheckBox
    private lateinit var cbNnapi: CheckBox
    private lateinit var btnBenchmark: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvStatus: TextView
    private lateinit var cardResults: LinearLayout
    private lateinit var tvResults: TextView
    private lateinit var btnReport: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        engine = BenchmarkEngine(this)

        // Detect device immediately
        deviceInfo = DeviceDetector.detect(this)
        tvDeviceInfo.text = DeviceDetector.formatDeviceInfo(deviceInfo)

        // Pre-fill EdgeGate values for internal testing
        etServerUrl.setText(DEFAULT_SERVER_URL)
        etWorkspaceId.setText(DEFAULT_WORKSPACE_ID)

        // Connect button
        btnConnect.setOnClickListener { connectToEdgeGate() }

        // Benchmark button
        btnBenchmark.setOnClickListener { runBenchmark() }

        // Report button
        btnReport.setOnClickListener { reportToEdgeGate() }

        // Show model selection immediately with test model option
        // (user only needs to enter API key to connect to EdgeGate)
        showModelCard(emptyList())
    }

    private fun bindViews() {
        tvDeviceInfo = findViewById(R.id.tvDeviceInfo)
        etServerUrl = findViewById(R.id.etServerUrl)
        etApiKey = findViewById(R.id.etApiKey)
        etWorkspaceId = findViewById(R.id.etWorkspaceId)
        btnConnect = findViewById(R.id.btnConnect)
        cardModels = findViewById(R.id.cardModels)
        spinnerModels = findViewById(R.id.spinnerModels)
        cbCpu = findViewById(R.id.cbCpu)
        cbGpu = findViewById(R.id.cbGpu)
        cbNnapi = findViewById(R.id.cbNnapi)
        btnBenchmark = findViewById(R.id.btnBenchmark)
        progressBar = findViewById(R.id.progressBar)
        tvStatus = findViewById(R.id.tvStatus)
        cardResults = findViewById(R.id.cardResults)
        tvResults = findViewById(R.id.tvResults)
        btnReport = findViewById(R.id.btnReport)
    }

    private fun connectToEdgeGate() {
        val url = etServerUrl.text.toString().trimEnd('/')
        val key = etApiKey.text.toString()
        val wsId = etWorkspaceId.text.toString()

        if (url.isEmpty() || key.isEmpty() || wsId.isEmpty()) {
            Toast.makeText(this, "Fill in all fields", Toast.LENGTH_SHORT).show()
            return
        }

        showProgress("Connecting to EdgeGate...")
        client = EdgeGateClient(url, key, wsId)

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) { client!!.testConnection() }

            result.onSuccess { wsName ->
                showStatus("Connected to: $wsName")

                // Fetch models
                val modelsResult = withContext(Dispatchers.IO) { client!!.listModels() }
                modelsResult.onSuccess { models ->
                    if (models.isEmpty()) {
                        showStatus("Connected but no models found. Upload a .tflite model to your workspace.")
                        // Still show the model card with option to load from URL
                        showModelCard(emptyList())
                    } else {
                        showModelCard(models)
                    }
                }.onFailure { e ->
                    showStatus("Connected but failed to list models: ${e.message}")
                    showModelCard(emptyList())
                }
            }.onFailure { e ->
                showStatus("Connection failed: ${e.message}")
            }

            hideProgress()
        }
    }

    private fun showModelCard(models: List<EdgeGateClient.ModelInfo>) {
        cardModels.visibility = View.VISIBLE

        val names = models.map { it.original_filename ?: it.id }.toMutableList()
        // Always add option to download a standard test model
        names.add(0, "⬇ Download MobileNet v1 (test)")

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, names)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerModels.adapter = adapter
    }

    private fun runBenchmark() {
        val selectedIndex = spinnerModels.selectedItemPosition
        val delegates = mutableListOf<String>()
        if (cbCpu.isChecked) delegates.add("cpu")
        if (cbGpu.isChecked) delegates.add("gpu")
        if (cbNnapi.isChecked) delegates.add("nnapi")

        if (delegates.isEmpty()) {
            Toast.makeText(this, "Select at least one delegate", Toast.LENGTH_SHORT).show()
            return
        }

        btnBenchmark.isEnabled = false
        showProgress("Preparing model...")

        lifecycleScope.launch {
            try {
                // Get model file
                val modelFile = withContext(Dispatchers.IO) {
                    if (selectedIndex == 0) {
                        // Download MobileNet test model
                        downloadTestModel()
                    } else {
                        // Download from EdgeGate
                        val models = client?.listModels()?.getOrNull() ?: emptyList()
                        val model = models[selectedIndex - 1]
                        val dest = File(cacheDir, model.original_filename ?: "model.tflite")
                        client?.downloadModel(model.id, dest)?.getOrThrow() ?: dest
                    }
                }

                currentModelName = modelFile.name
                showStatus("Running benchmark on ${delegates.joinToString(", ") { it.uppercase() }}...")

                // Run benchmark
                val config = BenchmarkConfig(
                    warmupRuns = 5,
                    measurementRuns = 50,
                    delegates = delegates,
                )

                val results = withContext(Dispatchers.Default) {
                    engine.benchmark(modelFile, config, this@MainActivity)
                }

                benchmarkResults = results

                // Show results
                val formatted = BenchmarkEngine.formatResults(
                    results, currentModelName, deviceInfo.chip
                )
                showResults(formatted)

            } catch (e: Exception) {
                showStatus("Benchmark failed: ${e.message}")
            } finally {
                hideProgress()
                btnBenchmark.isEnabled = true
            }
        }
    }

    private fun downloadTestModel(): File {
        val dest = File(cacheDir, "mobilenet_v1_1.0_224_quant.tflite")
        if (dest.exists() && dest.length() > 1000) return dest

        val url = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model_plus_flex"
        // Use a known small TFLite model
        val modelUrl = "https://raw.githubusercontent.com/google-coral/test_data/master/mobilenet_v1_1.0_224_quant.tflite"

        URL(modelUrl).openStream().use { input ->
            FileOutputStream(dest).use { output ->
                input.copyTo(output)
            }
        }
        return dest
    }

    private fun showResults(text: String) {
        runOnUiThread {
            cardResults.visibility = View.VISIBLE
            tvResults.text = text
            btnReport.visibility = View.VISIBLE
            tvStatus.visibility = View.GONE
        }
    }

    private fun reportToEdgeGate() {
        val results = benchmarkResults ?: return
        val edgeGateClient = client ?: return

        showProgress("Reporting to EdgeGate...")
        btnReport.isEnabled = false

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                edgeGateClient.reportBenchmark(deviceInfo, currentModelName, results)
            }

            result.onSuccess {
                Toast.makeText(this@MainActivity, "Reported to EdgeGate!", Toast.LENGTH_SHORT).show()
            }.onFailure { e ->
                // Save results locally even if EdgeGate endpoint not ready
                val file = File(getExternalFilesDir(null),
                    "benchmark_${deviceInfo.chipVendor}_${System.currentTimeMillis()}.json")
                val gson = com.google.gson.Gson()
                file.writeText(gson.toJson(mapOf(
                    "device" to deviceInfo,
                    "model" to currentModelName,
                    "results" to results,
                )))
                Toast.makeText(this@MainActivity,
                    "Saved locally: ${file.name}\n(EdgeGate: ${e.message})",
                    Toast.LENGTH_LONG).show()
            }

            hideProgress()
            btnReport.isEnabled = true
        }
    }

    // BenchmarkEngine.ProgressCallback
    override fun onProgress(delegate: String, status: String) {
        runOnUiThread {
            tvStatus.text = status
        }
    }

    private fun showProgress(msg: String) {
        runOnUiThread {
            progressBar.visibility = View.VISIBLE
            tvStatus.visibility = View.VISIBLE
            tvStatus.text = msg
        }
    }

    private fun hideProgress() {
        runOnUiThread {
            progressBar.visibility = View.GONE
        }
    }

    private fun showStatus(msg: String) {
        runOnUiThread {
            tvStatus.visibility = View.VISIBLE
            tvStatus.text = msg
        }
    }
}
