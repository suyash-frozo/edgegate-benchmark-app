plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.frozo.edgegate.benchmark"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.frozo.edgegate.benchmark"
        minSdk = 27  // GenieX SDK (com.qualcomm.qti:geniex-android) requires minSdk 27
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        ndk {
            abiFilters += listOf("arm64-v8a")  // 64-bit ARM only (all modern phones)
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // Native C++ build for llama.cpp
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    ndkVersion = "25.1.8937393"
}

dependencies {
    // TFLite — core + GPU delegate API + NNAPI
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // ONNX Runtime — for .onnx models (CPU + NNAPI)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")

    // GenieX — on-device LLM inference via Qualcomm NPU/GPU/CPU
    implementation("com.qualcomm.qti:geniex-android:0.3.5")

    // Networking
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.code.gson:gson:2.11.0")

    // Android
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
