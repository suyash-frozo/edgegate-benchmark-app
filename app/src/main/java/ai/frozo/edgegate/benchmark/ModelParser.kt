package ai.frozo.edgegate.benchmark

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

object ModelParser {
    data class ModelInfo(
        val totalLayers: Int,
        val format: String,  // "tflite", "onnx", "gguf"
        val modelSize: Long,
    )

    fun parse(file: File): ModelInfo {
        val ext = file.extension.lowercase()
        return when (ext) {
            "tflite" -> parseTflite(file)
            "onnx" -> parseOnnx(file)
            "gguf" -> parseGguf(file)
            else -> ModelInfo(0, "unknown", file.length())
        }
    }

    private fun parseTflite(file: File): ModelInfo {
        // TFLite uses FlatBuffer format
        // The operator count can be extracted from the schema
        try {
            val bytes = file.readBytes()
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            // FlatBuffer root table offset is at position 4
            val rootOffset = buf.getInt(0)
            // SubGraph table has operators vector
            // Simplified: count unique operator code entries
            // For a rough count, search for operator patterns in the buffer
            var opCount = 0
            // Count "operator_codes" entries - simplified heuristic
            // Real parsing would use TFLite FlatBuffer schema
            val content = String(bytes, Charsets.ISO_8859_1)
            // Count ADD, CONV_2D, etc operator entries by looking for patterns
            // Alternative: use the buffer structure
            // The number of operators is typically 10-100 for mobile models
            opCount = estimateLayerCount(file.length(), "tflite")
            return ModelInfo(opCount, "tflite", file.length())
        } catch (e: Exception) {
            return ModelInfo(estimateLayerCount(file.length(), "tflite"), "tflite", file.length())
        }
    }

    private fun parseOnnx(file: File): ModelInfo {
        // ONNX uses protobuf format
        // Count graph nodes
        try {
            val raf = RandomAccessFile(file, "r")
            val bytes = ByteArray(minOf(file.length(), 1024 * 1024).toInt()) // Read first 1MB
            raf.readFully(bytes)
            raf.close()
            // Count "op_type" field tags in protobuf (field tag for op_type in NodeProto is field 4)
            // Simplified: count occurrences of common op names
            val content = String(bytes, Charsets.ISO_8859_1)
            var count = 0
            val ops = listOf("Conv", "Relu", "BatchNorm", "Add", "MatMul", "Gemm", "Pool",
                "Reshape", "Transpose", "Softmax", "Sigmoid", "Concat", "Flatten",
                "GlobalAveragePool", "MaxPool", "AveragePool", "Clip", "Mul")
            for (op in ops) {
                var idx = 0
                while (true) {
                    idx = content.indexOf(op, idx)
                    if (idx == -1) break
                    count++
                    idx += op.length
                }
            }
            // Deduplicate - each op name appears once in op_type field
            val layerCount = if (count > 0) count / 2 else estimateLayerCount(file.length(), "onnx")
            return ModelInfo(maxOf(layerCount, 1), "onnx", file.length())
        } catch (e: Exception) {
            return ModelInfo(estimateLayerCount(file.length(), "onnx"), "onnx", file.length())
        }
    }

    private fun parseGguf(file: File): ModelInfo {
        // GGUF header contains tensor count
        try {
            val raf = RandomAccessFile(file, "r")
            // GGUF magic: 0x46475547 ("GGUF")
            val magic = ByteArray(4)
            raf.readFully(magic)
            if (String(magic) != "GGUF") {
                raf.close()
                return ModelInfo(0, "gguf", file.length())
            }
            // Version (uint32)
            val versionBytes = ByteArray(4)
            raf.readFully(versionBytes)
            // Tensor count (uint64)
            val tensorCountBytes = ByteArray(8)
            raf.readFully(tensorCountBytes)
            val buf = ByteBuffer.wrap(tensorCountBytes).order(ByteOrder.LITTLE_ENDIAN)
            val tensorCount = buf.getLong().toInt()
            raf.close()
            return ModelInfo(tensorCount, "gguf", file.length())
        } catch (e: Exception) {
            return ModelInfo(estimateLayerCount(file.length(), "gguf"), "gguf", file.length())
        }
    }

    // Rough estimate based on model size when parsing fails
    private fun estimateLayerCount(sizeBytes: Long, format: String): Int {
        val sizeMb = sizeBytes / (1024f * 1024f)
        return when (format) {
            "tflite" -> (sizeMb * 5).toInt().coerceIn(10, 200)
            "onnx" -> (sizeMb * 4).toInt().coerceIn(10, 200)
            "gguf" -> (sizeMb * 0.05).toInt().coerceIn(20, 100) // LLMs have fewer but larger layers
            else -> 20
        }
    }
}
