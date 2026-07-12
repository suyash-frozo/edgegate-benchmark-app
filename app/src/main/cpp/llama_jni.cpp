/**
 * JNI bridge for llama.cpp — connects Kotlin LlmBenchmark to native inference.
 *
 * Provides:
 *   - Model loading (GGUF files)
 *   - Text generation with timing metrics
 *   - Memory-safe cleanup
 *
 * Returns all metrics as JSON so the Kotlin side can parse them.
 */

#include <jni.h>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>

#include "llama.cpp/include/llama.h"

using namespace std::chrono;

extern "C" {

/**
 * Load a GGUF model file into memory.
 * Returns a pointer to the model context (cast to jlong), or 0 on failure.
 */
JNIEXPORT jlong JNICALL
Java_ai_frozo_edgegate_benchmark_LlmBenchmark_nativeLoadModel(
    JNIEnv *env, jobject thiz,
    jstring model_path, jint threads, jint gpu_layers
) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;

    llama_model *model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (!model) return 0;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = threads;
    ctx_params.n_threads_batch = threads;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        return 0;
    }

    // Pack model + context into a struct we can pass around
    struct LlamaState {
        llama_model *model;
        llama_context *ctx;
    };

    auto *state = new LlamaState{model, ctx};
    return reinterpret_cast<jlong>(state);
}

/**
 * Generate text from a prompt. Returns JSON with all timing metrics.
 */
JNIEXPORT jstring JNICALL
Java_ai_frozo_edgegate_benchmark_LlmBenchmark_nativeGenerate(
    JNIEnv *env, jobject thiz,
    jlong model_ptr, jstring prompt_str,
    jint max_tokens, jfloat temperature, jfloat top_p
) {
    struct LlamaState {
        llama_model *model;
        llama_context *ctx;
    };

    auto *state = reinterpret_cast<LlamaState *>(model_ptr);
    if (!state || !state->model || !state->ctx) {
        return env->NewStringUTF("{\"error\":true,\"error_message\":\"Invalid model pointer\"}");
    }

    const char *prompt = env->GetStringUTFChars(prompt_str, nullptr);

    // Tokenize the prompt
    const llama_vocab *vocab = llama_model_get_vocab(state->model);
    int n_prompt_max = strlen(prompt) * 2 + 128;
    std::vector<llama_token> tokens(n_prompt_max);
    int n_prompt_tokens = llama_tokenize(vocab, prompt, strlen(prompt),
                                          tokens.data(), n_prompt_max, true, true);
    env->ReleaseStringUTFChars(prompt_str, prompt);

    if (n_prompt_tokens < 0) {
        return env->NewStringUTF("{\"error\":true,\"error_message\":\"Tokenization failed\"}");
    }
    tokens.resize(n_prompt_tokens);

    // Clear KV cache (latest llama.cpp uses llama_memory_clear)
    llama_memory_clear(llama_get_memory(state->ctx), true);

    // Prompt evaluation — measure TTFT
    auto total_start = high_resolution_clock::now();
    auto prompt_start = high_resolution_clock::now();

    // Create batch for prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt_tokens);
    if (llama_decode(state->ctx, batch) != 0) {
        return env->NewStringUTF("{\"error\":true,\"error_message\":\"Prompt decode failed\"}");
    }

    auto prompt_end = high_resolution_clock::now();
    float prompt_eval_ms = duration_cast<microseconds>(prompt_end - prompt_start).count() / 1000.0f;

    // Token generation
    auto gen_start = high_resolution_clock::now();
    float ttft_ms = 0;

    std::string generated_text;
    int n_generated = 0;
    llama_token new_token;

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

    for (int i = 0; i < max_tokens; i++) {
        new_token = llama_sampler_sample(sampler, state->ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Record TTFT on first token
        if (i == 0) {
            auto first_token_time = high_resolution_clock::now();
            ttft_ms = duration_cast<microseconds>(first_token_time - total_start).count() / 1000.0f;
        }

        // Convert token to text
        char buf[256];
        int len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (len > 0) {
            generated_text.append(buf, len);
        }

        n_generated++;

        // Prepare next batch (single token)
        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(state->ctx, next_batch) != 0) break;
    }

    llama_sampler_free(sampler);

    auto gen_end = high_resolution_clock::now();
    float generation_ms = duration_cast<microseconds>(gen_end - gen_start).count() / 1000.0f;
    float total_ms = duration_cast<microseconds>(gen_end - total_start).count() / 1000.0f;

    float tokens_per_sec = (generation_ms > 0 && n_generated > 0)
        ? (n_generated / (generation_ms / 1000.0f))
        : 0;

    // Build JSON result
    std::ostringstream json;
    json << "{";
    json << "\"error\":false,";
    json << "\"text\":\"";
    // Escape the generated text for JSON
    for (char c : generated_text) {
        switch (c) {
            case '"': json << "\\\""; break;
            case '\\': json << "\\\\"; break;
            case '\n': json << "\\n"; break;
            case '\r': json << "\\r"; break;
            case '\t': json << "\\t"; break;
            default: json << c;
        }
    }
    json << "\",";
    json << "\"total_tokens\":" << (n_prompt_tokens + n_generated) << ",";
    json << "\"prompt_tokens\":" << n_prompt_tokens << ",";
    json << "\"generation_tokens\":" << n_generated << ",";
    json << "\"ttft_ms\":" << ttft_ms << ",";
    json << "\"tokens_per_second\":" << tokens_per_sec << ",";
    json << "\"total_time_ms\":" << total_ms << ",";
    json << "\"prompt_eval_time_ms\":" << prompt_eval_ms << ",";
    json << "\"generation_time_ms\":" << generation_ms;
    json << "}";

    return env->NewStringUTF(json.str().c_str());
}

/**
 * Unload model and free all resources.
 */
JNIEXPORT void JNICALL
Java_ai_frozo_edgegate_benchmark_LlmBenchmark_nativeUnloadModel(
    JNIEnv *env, jobject thiz, jlong model_ptr
) {
    struct LlamaState {
        llama_model *model;
        llama_context *ctx;
    };

    auto *state = reinterpret_cast<LlamaState *>(model_ptr);
    if (state) {
        if (state->ctx) llama_free(state->ctx);
        if (state->model) llama_model_free(state->model);
        delete state;
    }
}

/**
 * Get system info from llama.cpp (useful for debugging).
 */
JNIEXPORT jstring JNICALL
Java_ai_frozo_edgegate_benchmark_LlmBenchmark_nativeGetSystemInfo(
    JNIEnv *env, jobject thiz
) {
    return env->NewStringUTF(llama_print_system_info());
}

} // extern "C"
