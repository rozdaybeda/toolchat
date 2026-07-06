#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "tools.h"

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace toolchat {

struct AgentConfig {
    std::string model_path;
    int n_ctx      = 4096;
    int n_threads  = [] { unsigned n = std::thread::hardware_concurrency(); return n > 0 ? (int)n : 4; }();
    int n_gpu_layers = 0;
    int main_gpu     = 0;
    int max_tokens   = 1024;   // generation cap per model turn
    float temperature = 0.3f;  // low → deterministic, reliable tool calls
    float top_p       = 0.8f;
    float min_p       = 0.0f;  // 0 = disabled (some models, e.g. GLM, want 0.01)
    int top_k         = 20;
    // MoE models only: keep the expert FFN tensors of the first n layers in
    // system RAM instead of VRAM (llama.cpp's --n-cpu-moe). 0 = fully offload.
    int n_cpu_moe     = 0;
    int max_tool_iters = 6;    // safety cap on tool-call rounds per query
    uint32_t seed     = 42;    // sampler RNG seed (vary for benchmarking)
    // Compute device to offload to, as an opaque ggml_backend_dev_t (from
    // enumerate_devices().handle). nullptr = CPU / llama's default placement.
    // When set, n_gpu_layers should be > 0 (e.g. 99 for all layers).
    void* device = nullptr;
    // Non-thinking prefill (`<think></think>`) is a Qwen3/3.5 convention and is
    // wrong for Qwen2.5 (no thinking mode). -1 = auto-detect from architecture,
    // 0 = off, 1 = on.
    int think_prefill = -1;
};

// One tool invocation observed during a run, surfaced to the CLI for tracing.
struct ToolTrace {
    std::string name;
    nlohmann::json arguments;
    nlohmann::json result;
};

// Callback invoked as each tool call is executed (for live CLI output).
using TraceCallback = std::function<void(const ToolTrace&)>;

// A compute device ggml discovered (its backend registry). Always includes the
// CPU; GPUs/iGPUs appear only in a GPU-enabled build with a working driver.
struct DeviceInfo {
    void*       handle;       // ggml_backend_dev_t (opaque here); use as AgentConfig::device
    std::string name;         // short backend name, e.g. "Vulkan0"
    std::string description;  // human name, e.g. "NVIDIA GeForce RTX 5060 Ti"
    int         type;         // 0 = CPU, 1 = GPU, 2 = iGPU, 3 = other
};

// Enumerate the compute devices available for offloading.
std::vector<DeviceInfo> enumerate_devices();

// Which chat template / tool-call convention to use, chosen from the model's
// architecture. ChatML = Qwen (`<|im_start|>`, system role, `<tool_call>` JSON);
// Gemma = `<start_of_turn>` turns, no system role, native-tolerant tool parsing.
enum class PromptFormat { ChatML, Gemma };

class Agent {
public:
    Agent();
    ~Agent();

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;
    Agent(Agent&&) noexcept;
    Agent& operator=(Agent&&) noexcept;

    bool init(const AgentConfig& config, ToolRegistry registry);

    // Runs a single user query to completion: generate → execute any tool calls
    // → feed results back → repeat until the model returns a plain answer or the
    // iteration cap is hit. Returns the final assistant text.
    std::string run(const std::string& query, const TraceCallback& on_tool = {});

    bool is_ready() const { return initialized_; }
    const std::string& get_error() const { return last_error_; }
    int get_last_token_count() const { return last_token_count_; }
    const ToolRegistry& registry() const { return registry_; }

private:
    // Tool schemas + calling instructions, with no role/turn markers. ChatML
    // wraps this in a system turn; Gemma folds it into the first user turn.
    std::string build_tools_instructions() const;
    std::string generate(const std::string& prompt);            // raw model turn
    // Extract every tool call from model output — `<tool_call>{json}</tool_call>`
    // (Qwen) or the native `call:fn{...}` micro-syntax (Gemma).
    std::vector<nlohmann::json> parse_tool_calls(const std::string& text) const;
    void cleanup();

    llama_model*   model_   = nullptr;
    llama_context* ctx_     = nullptr;
    llama_sampler* sampler_ = nullptr;
    PromptFormat   format_  = PromptFormat::ChatML;
    bool           use_think_prefill_ = false;
    AgentConfig    config_;
    ToolRegistry   registry_;
    std::string    last_error_;
    bool           initialized_     = false;
    int            last_token_count_ = 0;
};

} // namespace toolchat
