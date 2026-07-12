package ai.frozo.edgegate.benchmark

import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.text.InputType
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

class MainActivity : AppCompatActivity(), BenchmarkEngine.ProgressCallback {

    companion object {
        const val DEFAULT_SERVER_URL = "https://edgegateapi.frozo.ai"
        const val DEFAULT_WORKSPACE_ID = "7479eff9-985a-49c0-b3a0-5e5b3885a5a7"

        private val GATE_PRESETS = listOf("Vision Model", "LLM", "Custom")
    }

    private lateinit var deviceInfo: DeviceInfo
    private lateinit var engine: BenchmarkEngine
    private var npuProfiler: NpuProfiler? = null
    private var client: EdgeGateClient? = null
    private var benchmarkResults: List<DelegateResult>? = null
    private var currentModelName: String = ""
    private var localModelFile: File? = null
    private var deviceReport: DeviceReport? = null
    private var profileExpanded = false

    // Quality gates state
    private var selectedPresetIndex = 0  // 0=Vision, 1=LLM, 2=Custom
    private val customGateRows = mutableListOf<GateRowViews>()

    // Views
    private lateinit var tvDeviceInfo: TextView
    private lateinit var tvDeviceProfileSummary: TextView
    private lateinit var tvDeviceProfileFull: TextView
    private lateinit var tvToggleProfile: TextView
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
    private lateinit var btnPickFile: Button
    private lateinit var tvSelectedFile: TextView

    // LLM views
    private lateinit var btnPickGguf: Button
    private lateinit var tvSelectedGguf: TextView
    private lateinit var etLlmPrompt: EditText
    private lateinit var etMaxTokens: EditText
    private lateinit var etLlmThreads: EditText
    private lateinit var btnRunLlm: Button
    private lateinit var cardLlmResults: LinearLayout
    private lateinit var tvLlmResults: TextView
    private var llmModelFile: File? = null

    // Quality gates views
    private lateinit var spinnerGatePreset: Spinner
    private lateinit var layoutCustomGates: LinearLayout
    private lateinit var gateRowsContainer: LinearLayout
    private lateinit var btnAddGate: Button
    private lateinit var btnRemoveGate: Button
    private lateinit var cardGateResults: LinearLayout
    private lateinit var tvGateResults: TextView

    // File pickers
    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { handlePickedFile(it) }
    }

    private val ggufPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        uri?.let { handlePickedGguf(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        engine = BenchmarkEngine(this)

        deviceInfo = DeviceDetector.detect(this)
        tvDeviceInfo.text = DeviceDetector.formatDeviceInfo(deviceInfo)

        etServerUrl.setText(DEFAULT_SERVER_URL)
        etWorkspaceId.setText(DEFAULT_WORKSPACE_ID)

        btnConnect.setOnClickListener { connectToEdgeGate() }
        btnBenchmark.setOnClickListener { runBenchmark() }
        btnReport.setOnClickListener { reportToEdgeGate() }
        btnPickFile.setOnClickListener {
            filePickerLauncher.launch(arrayOf("*/*"))
        }

        // LLM benchmark
        btnPickGguf.setOnClickListener { ggufPickerLauncher.launch(arrayOf("*/*")) }
        btnRunLlm.setOnClickListener { runLlmBenchmark() }

        // Quality gates setup
        setupGatePresetSpinner()
        btnAddGate.setOnClickListener { addCustomGateRow() }
        btnRemoveGate.setOnClickListener { removeLastCustomGateRow() }

        // Run device compatibility analysis
        analyzeDevice()

        // Toggle expand/collapse for device profile
        tvToggleProfile.setOnClickListener { toggleDeviceProfile() }

        showModelCard(emptyList())
    }

    // ========================================================================
    // Quality Gates UI
    // ========================================================================

    /** Helper class to hold references to views in a single custom gate row. */
    private data class GateRowViews(
        val container: LinearLayout,
        val metricSpinner: Spinner,
        val operatorSpinner: Spinner,
        val thresholdInput: EditText,
    )

    private fun setupGatePresetSpinner() {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, GATE_PRESETS)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerGatePreset.adapter = adapter

        spinnerGatePreset.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                selectedPresetIndex = position
                if (position == 2) {
                    // Custom: show custom gate editor
                    layoutCustomGates.visibility = View.VISIBLE
                    if (customGateRows.isEmpty()) {
                        addCustomGateRow() // start with one row
                    }
                } else {
                    layoutCustomGates.visibility = View.GONE
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun addCustomGateRow() {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = 8 }
        }

        // Metric spinner
        val metricSpinner = Spinner(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 100, 2f)
            val metricAdapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                AVAILABLE_METRICS,
            )
            metricAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            adapter = metricAdapter
        }

        // Operator spinner
        val operatorSpinner = Spinner(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 100, 1f).apply {
                marginStart = 8
            }
            val opAdapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                AVAILABLE_OPERATORS,
            )
            opAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            adapter = opAdapter
            // Default to "lte"
            setSelection(AVAILABLE_OPERATORS.indexOf("lte"))
        }

        // Threshold input
        val thresholdInput = EditText(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 100, 1f).apply {
                marginStart = 8
            }
            hint = "value"
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            textSize = 13f
            setTextColor(0xFFFFFFFF.toInt())
            setHintTextColor(0xFF56697D.toInt())
            setBackgroundResource(R.drawable.input_bg)
            setPadding(12, 8, 12, 8)
            setText("50")
        }

        row.addView(metricSpinner)
        row.addView(operatorSpinner)
        row.addView(thresholdInput)

        gateRowsContainer.addView(row)
        customGateRows.add(GateRowViews(row, metricSpinner, operatorSpinner, thresholdInput))
    }

    private fun removeLastCustomGateRow() {
        if (customGateRows.isNotEmpty()) {
            val last = customGateRows.removeAt(customGateRows.size - 1)
            gateRowsContainer.removeView(last.container)
        }
    }

    /** Get the currently configured gates based on preset selection. */
    private fun getSelectedGates(): List<Gate> {
        return when (selectedPresetIndex) {
            0 -> VISION_MODEL_GATES
            1 -> LLM_MODEL_GATES
            2 -> {
                // Build from custom rows
                customGateRows.mapNotNull { row ->
                    val metric = row.metricSpinner.selectedItem?.toString() ?: return@mapNotNull null
                    val operator = row.operatorSpinner.selectedItem?.toString() ?: return@mapNotNull null
                    val threshold = row.thresholdInput.text.toString().toFloatOrNull() ?: return@mapNotNull null
                    Gate(metric, operator, threshold)
                }
            }
            else -> VISION_MODEL_GATES
        }
    }

    // ========================================================================
    // Device Analysis
    // ========================================================================

    private fun analyzeDevice() {
        lifecycleScope.launch {
            val report = withContext(Dispatchers.Default) {
                DeviceCompatibility.analyze(this@MainActivity)
            }
            deviceReport = report

            // Initialize NpuProfiler for benchmark use
            val profiler = NpuProfiler(this@MainActivity)
            npuProfiler = profiler

            // Build summary (first 3 lines)
            val summary = buildString {
                append("${report.device.manufacturer} ${report.device.model}")
                append(" | RAM: ${"%.1f".format(report.ramTotalGb)} GB")
                append(" | ${"%.1f".format(report.ramAvailableGb)} GB free")
                val compatCount = report.modelCompatibility.count { it.status == CompatStatus.COMPATIBLE }
                val total = report.modelCompatibility.size
                append("\n$compatCount/$total models compatible")
                if (!report.readyToBenchmark) append(" | Issues detected")
                // NPU accelerator count
                val npuCount = report.nnDevices.count { it.type == NpuProfiler.DEVICE_TYPE_ACCELERATOR }
                if (npuCount > 0) append(" | $npuCount NPU/DSP detected")
                if (report.rootAvailable) append("\nRoot detected \u2014 real NPU metrics will be captured during benchmark")
            }
            tvDeviceProfileSummary.text = summary
            tvDeviceProfileFull.text = DeviceCompatibility.formatReport(report)

            // Show warnings as toasts
            for (warning in report.warnings) {
                if (warning.startsWith("[CRITICAL]")) {
                    Toast.makeText(this@MainActivity, warning, Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this@MainActivity, warning, Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun toggleDeviceProfile() {
        profileExpanded = !profileExpanded
        if (profileExpanded) {
            tvDeviceProfileSummary.visibility = View.GONE
            tvDeviceProfileFull.visibility = View.VISIBLE
            tvToggleProfile.text = "Collapse"
        } else {
            tvDeviceProfileSummary.visibility = View.VISIBLE
            tvDeviceProfileFull.visibility = View.GONE
            tvToggleProfile.text = "Expand"
        }
    }

    // ========================================================================
    // View Binding
    // ========================================================================

    private fun bindViews() {
        tvDeviceInfo = findViewById(R.id.tvDeviceInfo)
        tvDeviceProfileSummary = findViewById(R.id.tvDeviceProfileSummary)
        tvDeviceProfileFull = findViewById(R.id.tvDeviceProfileFull)
        tvToggleProfile = findViewById(R.id.tvToggleProfile)
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
        btnPickFile = findViewById(R.id.btnPickFile)
        tvSelectedFile = findViewById(R.id.tvSelectedFile)

        // LLM views
        btnPickGguf = findViewById(R.id.btnPickGguf)
        tvSelectedGguf = findViewById(R.id.tvSelectedGguf)
        etLlmPrompt = findViewById(R.id.etLlmPrompt)
        etMaxTokens = findViewById(R.id.etMaxTokens)
        etLlmThreads = findViewById(R.id.etLlmThreads)
        btnRunLlm = findViewById(R.id.btnRunLlm)
        cardLlmResults = findViewById(R.id.cardLlmResults)
        tvLlmResults = findViewById(R.id.tvLlmResults)

        // Quality gates views
        spinnerGatePreset = findViewById(R.id.spinnerGatePreset)
        layoutCustomGates = findViewById(R.id.layoutCustomGates)
        gateRowsContainer = findViewById(R.id.gateRowsContainer)
        btnAddGate = findViewById(R.id.btnAddGate)
        btnRemoveGate = findViewById(R.id.btnRemoveGate)
        cardGateResults = findViewById(R.id.cardGateResults)
        tvGateResults = findViewById(R.id.tvGateResults)
    }

    // ========================================================================
    // File Handling
    // ========================================================================

    private fun handlePickedFile(uri: Uri) {
        // Copy file to cache dir so TFLite can read it
        val fileName = getFileName(uri) ?: "model_${System.currentTimeMillis()}"
        val dest = File(cacheDir, fileName)

        try {
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(dest).use { output ->
                    input.copyTo(output)
                }
            }

            localModelFile = dest
            tvSelectedFile.visibility = View.VISIBLE
            tvSelectedFile.text = "\u2713 Loaded: $fileName (${dest.length() / 1024} KB)"

            // Update spinner to show the local file
            updateSpinnerWithLocalFile(fileName)

            showStatus("Model loaded from device. Tap 'Run Benchmark' to test.")
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load file: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun getFileName(uri: Uri): String? {
        var name: String? = null
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && nameIndex >= 0) {
                name = cursor.getString(nameIndex)
            }
        }
        return name
    }

    private fun updateSpinnerWithLocalFile(fileName: String) {
        val currentItems = mutableListOf("\u2B07 Download MobileNet v1 (test)", "\uD83D\uDCC1 $fileName (local)")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, currentItems)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerModels.adapter = adapter
        spinnerModels.setSelection(1) // Select the local file
    }

    // ========================================================================
    // EdgeGate Connection
    // ========================================================================

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
                val modelsResult = withContext(Dispatchers.IO) { client!!.listModels() }
                modelsResult.onSuccess { models ->
                    showModelCard(models)
                    if (models.isEmpty()) {
                        showStatus("Connected. No models in workspace \u2014 pick a local file or use test model.")
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

        val names = mutableListOf("\u2B07 Download MobileNet v1 (test)")
        // Add local file if already picked
        if (localModelFile != null) {
            names.add("\uD83D\uDCC1 ${localModelFile!!.name} (local)")
        }
        // Add EdgeGate models
        names.addAll(models.map { "\u2601 ${it.original_filename ?: it.id}" })

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, names)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerModels.adapter = adapter
    }

    // ========================================================================
    // Benchmark Execution
    // ========================================================================

    private fun runBenchmark() {
        val selectedText = spinnerModels.selectedItem?.toString() ?: ""
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

        // Capture gates before going async
        val gates = getSelectedGates()

        lifecycleScope.launch {
            try {
                val modelFile = withContext(Dispatchers.IO) {
                    when {
                        // Local file picked from device
                        selectedText.startsWith("\uD83D\uDCC1") && localModelFile != null -> {
                            localModelFile!!
                        }
                        // Download MobileNet test
                        selectedText.startsWith("\u2B07") -> {
                            downloadTestModel()
                        }
                        // EdgeGate cloud model
                        selectedText.startsWith("\u2601") -> {
                            val models = client?.listModels()?.getOrNull() ?: emptyList()
                            val cloudIndex = selectedIndex - 1 - (if (localModelFile != null) 1 else 0)
                            if (cloudIndex >= 0 && cloudIndex < models.size) {
                                val model = models[cloudIndex]
                                val dest = File(cacheDir, model.original_filename ?: "model.tflite")
                                client?.downloadModel(model.id, dest)?.getOrThrow() ?: dest
                            } else {
                                downloadTestModel()
                            }
                        }
                        else -> downloadTestModel()
                    }
                }

                currentModelName = modelFile.name
                showStatus("Running on ${delegates.joinToString(", ") { it.uppercase() }}...")

                val config = BenchmarkConfig(
                    warmupRuns = 5,
                    measurementRuns = 50,
                    delegates = delegates,
                )

                val results = withContext(Dispatchers.Default) {
                    engine.benchmark(modelFile, config, this@MainActivity, npuProfiler)
                }

                benchmarkResults = results
                showResults(BenchmarkEngine.formatResults(results, currentModelName, deviceInfo.chip))

                // ---- Quality Gates Evaluation ----
                evaluateAndShowGates(results, gates)

            } catch (e: Exception) {
                showStatus("Benchmark failed: ${e.message}")
            } finally {
                hideProgress()
                btnBenchmark.isEnabled = true
            }
        }
    }

    // ========================================================================
    // Quality Gates Evaluation (post-benchmark)
    // ========================================================================

    private fun evaluateAndShowGates(results: List<DelegateResult>, gates: List<Gate>) {
        // Use best delegate (fastest successful)
        val best = results.filter { it.success }.minByOrNull { it.inferenceTimeMs }
        if (best == null || gates.isEmpty()) {
            runOnUiThread {
                cardGateResults.visibility = View.GONE
            }
            return
        }

        val metrics = extractMetrics(best)
        val evaluation = evaluateGates(gates, metrics, best.allLatenciesMs)
        val formattedText = formatGatesResult(evaluation)

        runOnUiThread {
            cardGateResults.visibility = View.VISIBLE
            tvGateResults.text = formattedText

            // Change result card border: green if all pass, red if any fail
            val borderDrawable = if (evaluation.passed) {
                R.drawable.card_bg_green
            } else {
                R.drawable.card_bg_red
            }
            cardResults.setBackgroundResource(borderDrawable)
            cardGateResults.setBackgroundResource(borderDrawable)
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    // ========================================================================
    // LLM Benchmark
    // ========================================================================

    private fun handlePickedGguf(uri: Uri) {
        val fileName = getFileName(uri) ?: "model.gguf"
        val dest = File(cacheDir, fileName)

        try {
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(dest).use { output ->
                    input.copyTo(output)
                }
            }
            llmModelFile = dest
            tvSelectedGguf.visibility = View.VISIBLE
            tvSelectedGguf.text = "\u2713 ${fileName} (${"%.1f".format(dest.length() / (1024f * 1024f))} MB)"
            showStatus("GGUF model loaded. Tap 'Run LLM Benchmark'.")
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun runLlmBenchmark() {
        val modelFile = llmModelFile
        if (modelFile == null) {
            Toast.makeText(this, "Pick a .gguf model first", Toast.LENGTH_SHORT).show()
            return
        }

        if (!LlmBenchmark.isAvailable()) {
            Toast.makeText(this,
                "LLM engine not available. App needs to be built with NDK (llama.cpp).",
                Toast.LENGTH_LONG).show()
            return
        }

        val prompt = etLlmPrompt.text.toString()
        val maxTokens = etMaxTokens.text.toString().toIntOrNull() ?: 64
        val threads = etLlmThreads.text.toString().toIntOrNull() ?: 4

        btnRunLlm.isEnabled = false
        showProgress("Loading LLM...")

        val llmEngine = LlmBenchmark(this)
        val config = LlmBenchmarkConfig(
            prompt = prompt,
            maxTokens = maxTokens,
            threads = threads,
        )

        lifecycleScope.launch {
            try {
                val result = llmEngine.benchmark(modelFile, config,
                    object : LlmBenchmark.ProgressCallback {
                        override fun onProgress(status: String) {
                            runOnUiThread { tvStatus.text = status }
                        }
                        override fun onToken(token: String) {
                            // Could show streaming tokens
                        }
                    }
                )

                runOnUiThread {
                    cardLlmResults.visibility = View.VISIBLE
                    tvLlmResults.text = llmEngine.formatResult(result)

                    // Evaluate LLM gates if preset is LLM
                    if (selectedPresetIndex == 1 && result.success) {
                        val metrics = mapOf(
                            "ttft_ms" to result.ttftMs,
                            "tokens_per_second" to result.tokensPerSecond,
                            "peak_memory_mb" to result.peakMemoryMb,
                            "inference_time_ms" to result.totalTimeMs,
                        )
                        val gates = getSelectedGates()
                        if (gates.isNotEmpty()) {
                            val evaluation = evaluateGates(gates, metrics, emptyList())
                            cardGateResults.visibility = View.VISIBLE
                            tvGateResults.text = formatGatesResult(evaluation)
                            val bg = if (evaluation.passed) R.drawable.card_bg_green else R.drawable.card_bg_red
                            cardLlmResults.setBackgroundResource(bg)
                            cardGateResults.setBackgroundResource(bg)
                        }
                    }
                }
            } catch (e: Exception) {
                showStatus("LLM benchmark failed: ${e.message}")
            } finally {
                hideProgress()
                btnRunLlm.isEnabled = true
            }
        }
    }

    // ========================================================================
    // Vision Model Helpers
    // ========================================================================

    private fun downloadTestModel(): File {
        val dest = File(cacheDir, "mobilenet_v1_1.0_224_quant.tflite")
        if (dest.exists() && dest.length() > 1000) return dest

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
            // Reset border to default before gate evaluation updates it
            cardResults.setBackgroundResource(R.drawable.card_bg)
            tvResults.text = text
            btnReport.visibility = View.VISIBLE
            tvStatus.visibility = View.GONE
        }
    }

    private fun reportToEdgeGate() {
        val results = benchmarkResults ?: return

        if (client == null) {
            // Save locally if not connected
            saveResultsLocally(results)
            return
        }

        showProgress("Reporting to EdgeGate...")
        btnReport.isEnabled = false

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                client!!.reportBenchmark(deviceInfo, currentModelName, results)
            }

            result.onSuccess {
                Toast.makeText(this@MainActivity, "Reported to EdgeGate!", Toast.LENGTH_SHORT).show()
            }.onFailure { e ->
                saveResultsLocally(results)
            }

            hideProgress()
            btnReport.isEnabled = true
        }
    }

    private fun saveResultsLocally(results: List<DelegateResult>) {
        val file = File(
            getExternalFilesDir(null),
            "benchmark_${deviceInfo.chipVendor}_${System.currentTimeMillis()}.json"
        )
        val gson = com.google.gson.Gson()
        file.writeText(gson.toJson(mapOf(
            "device" to deviceInfo,
            "model" to currentModelName,
            "results" to results,
        )))
        Toast.makeText(
            this,
            "Saved: ${file.name}",
            Toast.LENGTH_LONG
        ).show()
    }

    override fun onProgress(delegate: String, status: String) {
        runOnUiThread { tvStatus.text = status }
    }

    private fun showProgress(msg: String) {
        runOnUiThread {
            progressBar.visibility = View.VISIBLE
            tvStatus.visibility = View.VISIBLE
            tvStatus.text = msg
        }
    }

    private fun hideProgress() {
        runOnUiThread { progressBar.visibility = View.GONE }
    }

    private fun showStatus(msg: String) {
        runOnUiThread {
            tvStatus.visibility = View.VISIBLE
            tvStatus.text = msg
        }
    }
}
