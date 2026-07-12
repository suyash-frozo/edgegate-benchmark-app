# EdgeGate Benchmark App

Android app that benchmarks AI models on real phone hardware and reports results to [EdgeGate](https://edgegate.frozo.ai).

Supports **4 runtimes** — test any model format on any Snapdragon/ARM device:

| Runtime | Model Format | Delegates |
|---|---|---|
| **TFLite** | `.tflite` | CPU, GPU (OpenCL), NNAPI (NPU/DSP) |
| **ONNX Runtime** | `.onnx` | CPU, NNAPI |
| **llama.cpp** | `.gguf` | CPU (with ARM NEON) |
| **GenieX** | GenieX catalog | NPU, GPU, CPU, Hybrid |

## What it measures

- **Inference time** (median, P50, P95, P99)
- **Peak memory** (MB)
- **TTFT** (time to first token — LLMs)
- **Decode speed** (tokens/sec — LLMs)
- **NPU/GPU/CPU delegation** — which layers ran where
- **Cold start vs warm** load times
- **Quality gates** — pass/fail against configurable thresholds

## Quick Start

### Prerequisites

- Android Studio 2024.3.1+
- Android device with USB debugging enabled
- NDK 25.1+ (for llama.cpp native build)

### Build & Run

```bash
git clone https://github.com/suyash-frozo/edgegate-benchmark-app.git
cd edgegate-benchmark-app
# Open in Android Studio → Run
```

### Connect to EdgeGate

1. Get an API key from [EdgeGate Settings](https://edgegate.frozo.ai)
2. In the app: enter Server URL, API Key, Workspace ID
3. Tap **Connect** → select model → **Run Benchmark** → **Report to EdgeGate**

## GenieX Integration

Benchmark LLMs from the GenieX catalog on the Snapdragon NPU:

```kotlin
val geniex = GenieXBenchmark(context)

// Benchmark on NPU
val result = geniex.benchmark(modelId = "qwen3-0.6b", computeUnit = "npu")

// Compare NPU vs GPU vs CPU
val comparison = geniex.compareComputeUnits(modelId = "qwen3-0.6b")

// Which models fit on this device?
val recommendations = geniex.recommendModelsForDevice(ramGb = 8.0)
```

## Project Structure

```
app/src/main/java/ai/frozo/edgegate/benchmark/
├── MainActivity.kt          # Main UI
├── BenchmarkEngine.kt       # TFLite + ONNX Runtime benchmarking
├── GenieXBenchmark.kt       # GenieX LLM benchmarking (NPU/GPU/CPU)
├── LlmBenchmark.kt          # llama.cpp JNI bridge
├── EdgeGateClient.kt        # Reports results to EdgeGate API
├── DeviceDetector.kt        # Device profiling (chip, GPU, NPU, RAM)
├── DeviceCompatibility.kt   # Model compatibility analysis
├── NpuProfiler.kt           # NPU hardware profiling
├── ModelParser.kt           # Parse TFLite/ONNX/GGUF layer counts
├── DelegateReporter.kt      # Layer delegation estimation
└── QualityGates.kt          # Pass/fail gate evaluation
```

## License

BSD-3-Clause

Built by [Frozo](https://frozo.ai) — makers of [EdgeGate](https://edgegate.frozo.ai).
