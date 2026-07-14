cmake_minimum_required(VERSION 3.22)
project(llama_jni)

set(CMAKE_CXX_STANDARD 17)

# Build llama.cpp — disable what we don't need
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Build llama.cpp/ggml as STATIC and link them privately into libllama_jni.so.
# By default llama.cpp emits shared libllama.so + libggml*.so — the SAME filenames
# the GenieX AAR ships. When both are packaged, ours win and shadow GenieX's, so
# GenieX's llama_cpp plugin can't resolve libllama.so ("Failed to load plugin_id
# symbol", registerPlugin rc -100000 -> geniex_llm_create "plugin not found" -100302).
# Static-linking means we emit ONLY libllama_jni.so, leaving GenieX's runtime intact.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Enable ARM NEON optimizations
if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
endif()

add_subdirectory(llama.cpp)

# Build our JNI bridge — only depends on llama (not common)
add_library(llama_jni SHARED llama_jni.cpp)

target_include_directories(llama_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include
)

target_link_libraries(llama_jni
    llama
    log
)

# Hide all symbols pulled in from the static llama/ggml archives so libllama_jni.so
# exports only its JNI entry points. Otherwise those global symbols would still be
# visible and could interpose GenieX's own libllama.so/libggml*.so at runtime.
target_link_options(llama_jni PRIVATE "-Wl,--exclude-libs,ALL")
