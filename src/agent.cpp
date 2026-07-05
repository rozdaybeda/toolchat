#include "agent.h"
#include "llama.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace toolchat {

// Enumerate every device in ggml's backend registry (CPU always present; GPUs /
// iGPUs only in a GPU-enabled build with a working driver).
std::vector<DeviceInfo> enumerate_devices() {
    llama_backend_init();  // idempotent; registers the static backends
    std::vector<DeviceInfo> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        int t;
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:  t = 0; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:  t = 1; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: t = 2; break;
            default:                            t = 3; break;
        }
        const char* name = ggml_backend_dev_name(d);
        const char* desc = ggml_backend_dev_description(d);
        out.push_back({ (void*)d, name ? name : "", desc ? desc : "", t });
    }
    return out;
}

} // namespace toolchat

namespace toolchat {

using nlohmann::json;

// --- small string helpers ---------------------------------------------------

static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// Qwen3.x may emit a <think>...</think> reasoning block even in non-thinking
// mode. Strip it before parsing tool calls / displaying the answer.
static std::string strip_think(const std::string& s) {
    std::string out;
    size_t pos = 0;
    for (;;) {
        size_t open = s.find("<think>", pos);
        if (open == std::string::npos) { out += s.substr(pos); break; }
        out += s.substr(pos, open - pos);
        size_t close = s.find("</think>", open);
        if (close == std::string::npos) break;           // unterminated → drop rest
        pos = close + std::string("</think>").size();
    }
    return out;
}

// Gemma sometimes renders its turn markers as literal text instead of stopping on
// them (e.g. a trailing "<end_of_turn>" leaking into the answer). Cut the output
// at the first turn marker so it doesn't show up in answers or tool parsing.
static std::string strip_gemma_markers(const std::string& s) {
    std::string out = s;
    for (const char* m : {"<end_of_turn>", "<start_of_turn>"}) {
        size_t p = out.find(m);
        if (p != std::string::npos) out = out.substr(0, p);
    }
    return out;
}

// --- lifecycle --------------------------------------------------------------

Agent::Agent() = default;

Agent::~Agent() { cleanup(); }

Agent::Agent(Agent&& o) noexcept
    : model_(o.model_), ctx_(o.ctx_), sampler_(o.sampler_),
      format_(o.format_), use_think_prefill_(o.use_think_prefill_),
      config_(std::move(o.config_)),
      registry_(std::move(o.registry_)), last_error_(std::move(o.last_error_)),
      initialized_(o.initialized_), last_token_count_(o.last_token_count_) {
    o.model_ = nullptr; o.ctx_ = nullptr; o.sampler_ = nullptr; o.initialized_ = false;
}

Agent& Agent::operator=(Agent&& o) noexcept {
    if (this != &o) {
        cleanup();
        model_ = o.model_; ctx_ = o.ctx_; sampler_ = o.sampler_;
        format_ = o.format_; use_think_prefill_ = o.use_think_prefill_;
        config_ = std::move(o.config_); registry_ = std::move(o.registry_);
        last_error_ = std::move(o.last_error_);
        initialized_ = o.initialized_; last_token_count_ = o.last_token_count_;
        o.model_ = nullptr; o.ctx_ = nullptr; o.sampler_ = nullptr; o.initialized_ = false;
    }
    return *this;
}

void Agent::cleanup() {
    if (sampler_) { llama_sampler_free(sampler_); sampler_ = nullptr; }
    if (ctx_)     { llama_free(ctx_);             ctx_     = nullptr; }
    if (model_)   { llama_model_free(model_);     model_   = nullptr; }
    initialized_ = false;
}

bool Agent::init(const AgentConfig& config, ToolRegistry registry) {
    cleanup();
    config_   = config;
    registry_ = std::move(registry);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    model_params.main_gpu     = config.main_gpu;

    // If a specific device was chosen (e.g. the dedicated GPU vs the iGPU), pin
    // offloading to exactly that device. The list is read during the load call
    // below, so a local (null-terminated) array is fine.
    ggml_backend_dev_t dev_list[2] = { (ggml_backend_dev_t)config.device, nullptr };
    if (config.device)
        model_params.devices = dev_list;

    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        last_error_ = "Failed to load model from: " + config.model_path;
        return false;
    }

    // Read the architecture once — drives both the thinking prefill and the
    // Flash Attention decision below.
    char arch[64] = {};
    llama_model_meta_val_str(model_, "general.architecture", arch, sizeof(arch));
    const std::string a(arch);
    const bool is_qwen35 = (a == "qwen35" || a.rfind("qwen35", 0) == 0);  // hybrid Gated-DeltaNet
    const bool is_gemma  = (a.rfind("gemma", 0) == 0);                    // gemma2/gemma3/gemma4

    // Chat template family: Gemma uses <start_of_turn> turns with no system role;
    // everything else uses Qwen-style ChatML.
    format_ = is_gemma ? PromptFormat::Gemma : PromptFormat::ChatML;

    // Non-thinking prefill: on for qwen3/qwen3.5, off for qwen2/Qwen2.5 and Gemma
    // (Gemma has no thinking mode / no <think> convention).
    if (config.think_prefill == -1)
        use_think_prefill_ = (a.rfind("qwen3", 0) == 0);  // "qwen3", "qwen35", ...
    else
        use_think_prefill_ = (config.think_prefill != 0);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = config.n_ctx;
    // We re-decode the whole conversation in one batch each turn, so the batch
    // must be able to hold a full context (the default n_batch=2048 otherwise
    // trips GGML_ASSERT(n_tokens_all <= n_batch) on longer prompts).
    ctx_params.n_batch         = config.n_ctx;
    ctx_params.n_threads       = config.n_threads;
    ctx_params.n_threads_batch = config.n_threads;
    // Only Qwen3.5's fused Gated Delta Net path needs Flash Attention off (it
    // segfaults with FA on at this build). Standard transformers (qwen2/qwen3)
    // keep FA on auto — disabling it there just costs speed, especially on GPU.
    if (is_qwen35)
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        last_error_ = "Failed to create context";
        cleanup();
        return false;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(config.seed));

    initialized_ = true;
    return true;
}

// --- prompt construction ----------------------------------------------------

// Tool schemas + calling instructions, WITHOUT any role/turn markers. Qwen wraps
// this in a `<|im_start|>system … <|im_end|>` turn; Gemma (no system role) folds
// it into the first user turn. The <tool_call>{json}</tool_call> contract is the
// same for both — Gemma follows it by instruction, and parse_tool_calls() is also
// tolerant of Gemma's native call:fn{...} form.
std::string Agent::build_tools_instructions() const {
    std::string tools_block;
    for (const auto& schema : registry_.schemas_json()) {
        tools_block += schema.dump() + "\n";
    }
    if (!tools_block.empty()) tools_block.pop_back();  // trailing newline

    std::string instr =
        "You are a helpful assistant with access to tools. Use a tool only when it "
        "is needed to answer the user; otherwise reply directly. Do not show your "
        "reasoning. /no_think\n";

    if (!registry_.empty()) {
        instr +=
            "\n# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n" + tools_block + "\n</tools>\n\n"
            "For each function call, return a json object with function name and "
            "arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
            "</tool_call>";
    }
    return instr;
}

// --- generation -------------------------------------------------------------

std::string Agent::generate(const std::string& prompt) {
    if (!initialized_) { last_error_ = "Agent not initialized"; return ""; }

    llama_memory_clear(llama_get_memory(ctx_), true);

    const llama_vocab* vocab = llama_model_get_vocab(model_);

    std::vector<llama_token> tokens(config_.n_ctx);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  tokens.data(), (int32_t)tokens.size(), false, true);
    if (n_tokens < 0) {
        last_error_ = "Tokenization failed (prompt exceeds context of "
                    + std::to_string(config_.n_ctx) + " tokens)";
        return "";
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(ctx_, batch) != 0) {
        last_error_ = "Failed to decode prompt";
        return "";
    }

    std::string result;
    int token_count = 0;
    for (int i = 0; i < config_.max_tokens; ++i) {
        llama_token tok = llama_sampler_sample(sampler_, ctx_, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;   // <|im_end|> / EOS

        ++token_count;
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) result.append(buf, n);

        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx_, batch) != 0) {
            last_error_ = "Failed to decode token";
            break;
        }
    }

    last_token_count_ = token_count;
    return result;
}

// --- tool-call parsing ------------------------------------------------------

// Parse Gemma's native function-call micro-syntax as a fallback. Both documented
// variants share the inner shape `call:name{key:value, key:<delim>str<delim>}`
// (delim is <escape> or <|"|>); we scan for that and normalize to {name,arguments}.
static void parse_gemma_calls(const std::string& text, std::vector<json>& calls) {
    const std::string marker = "call:";
    size_t pos = 0;
    for (;;) {
        size_t c = text.find(marker, pos);
        if (c == std::string::npos) break;
        size_t brace = text.find('{', c);
        if (brace == std::string::npos) break;
        std::string name = trim(text.substr(c + marker.size(), brace - (c + marker.size())));
        // Match the balanced closing brace of the argument block.
        int depth = 1; size_t i = brace + 1;
        for (; i < text.size() && depth > 0; ++i) {
            if (text[i] == '{') ++depth;
            else if (text[i] == '}') --depth;
        }
        if (depth != 0) break;                       // unterminated
        std::string body = text.substr(brace + 1, (i - 1) - (brace + 1));
        pos = i;
        if (name.empty()) continue;

        // Split body on top-level commas into key:value pairs.
        json args = json::object();
        size_t start = 0;
        auto add_pair = [&](const std::string& piece) {
            size_t colon = piece.find(':');
            if (colon == std::string::npos) return;
            std::string key = trim(piece.substr(0, colon));
            std::string val = trim(piece.substr(colon + 1));
            if (key.empty()) return;
            // Strip Gemma string delimiters (<escape>…<escape> or <|"|>…<|"|>).
            for (const char* d : {"<escape>", "<|\"|>"}) {
                std::string ds(d);
                if (val.size() >= 2 * ds.size() && val.compare(0, ds.size(), ds) == 0 &&
                    val.compare(val.size() - ds.size(), ds.size(), ds) == 0) {
                    args[key] = val.substr(ds.size(), val.size() - 2 * ds.size());
                    return;
                }
            }
            // Bare token: number if it parses, else raw string.
            try { size_t used; double num = std::stod(val, &used);
                  if (used == val.size()) { args[key] = num; return; } } catch (...) {}
            args[key] = val;
        };
        int d2 = 0;
        for (i = 0; i <= body.size(); ++i) {
            if (i == body.size() || (body[i] == ',' && d2 == 0)) {
                add_pair(body.substr(start, i - start));
                start = i + 1;
            } else if (body[i] == '{' || body[i] == '[') ++d2;
            else if (body[i] == '}' || body[i] == ']') --d2;
        }
        calls.push_back(json{{"name", name}, {"arguments", args}});
    }
}

// Some models (e.g. Gemma 3) ignore the <tool_call> tags and just emit the JSON
// object, often inside a ```json ... ``` fence:  {"name": ..., "arguments": ...}.
// Scan for balanced, string-aware {...} groups and accept any that parse to an
// object with a string "name".
static void parse_loose_json_calls(const std::string& text, std::vector<json>& calls) {
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '{') { ++i; continue; }
        int depth = 0; bool in_str = false; char prev = 0; size_t j = i;
        for (; j < text.size(); ++j) {
            char c = text[j];
            if (in_str) { if (c == '"' && prev != '\\') in_str = false; }
            else if (c == '"') in_str = true;
            else if (c == '{') ++depth;
            else if (c == '}') { if (--depth == 0) { ++j; break; } }
            prev = c;
        }
        if (depth != 0) break;  // unbalanced → give up
        try {
            json o = json::parse(text.substr(i, j - i));
            if (o.is_object() && o.contains("name") && o["name"].is_string())
                calls.push_back(std::move(o));
        } catch (const std::exception&) {}
        i = j;
    }
}

std::vector<json> Agent::parse_tool_calls(const std::string& text) const {
    std::vector<json> calls;
    const std::string open = "<tool_call>";
    const std::string close = "</tool_call>";
    size_t pos = 0;
    for (;;) {
        size_t a = text.find(open, pos);
        if (a == std::string::npos) break;
        size_t b = text.find(close, a + open.size());
        if (b == std::string::npos) break;
        std::string body = trim(text.substr(a + open.size(), b - (a + open.size())));
        try {
            json call = json::parse(body);
            if (call.contains("name") && call["name"].is_string())
                calls.push_back(std::move(call));
        } catch (const std::exception&) {
            // Malformed tool call — skip; the model can retry on the next round.
        }
        pos = b + close.size();
    }
    // Native-tolerant: if the model (Gemma) didn't use our <tool_call> tags, try
    // its alternatives — a bare/fenced JSON object (Gemma 3) or the native
    // call:fn{...} micro-syntax (Gemma 4).
    if (calls.empty() && format_ == PromptFormat::Gemma) {
        parse_loose_json_calls(text, calls);
        if (calls.empty()) parse_gemma_calls(text, calls);
    }

    // Normalize: some models (e.g. Gemma 3) emit "arguments" as a *stringified*
    // JSON object ("{}" / "{\"path\":\"x\"}") rather than an object. Parse it back
    // so tool dispatch always gets a real object.
    for (auto& c : calls) {
        if (!c.contains("arguments")) { c["arguments"] = json::object(); continue; }
        if (c["arguments"].is_string()) {
            try { c["arguments"] = json::parse(c["arguments"].get<std::string>()); }
            catch (const std::exception&) { c["arguments"] = json::object(); }
        }
        if (!c["arguments"].is_object()) c["arguments"] = json::object();
    }
    return calls;
}

// --- the agent loop ---------------------------------------------------------

std::string Agent::run(const std::string& query, const TraceCallback& on_tool) {
    if (!initialized_) { last_error_ = "Agent not initialized"; return ""; }

    const bool gemma = (format_ == PromptFormat::Gemma);

    // Per-family turn markers. Qwen: separate system turn + ChatML roles. Gemma:
    // <bos>, <start_of_turn>role/<end_of_turn>, no system role (instructions fold
    // into the first user turn), and no <think> prefill.
    const std::string user_open  = gemma ? "<start_of_turn>user\n"  : "<|im_start|>user\n";
    const std::string turn_end   = gemma ? "<end_of_turn>\n"        : "<|im_end|>\n";
    const std::string asst_primer = gemma
        ? "<start_of_turn>model\n"
        : (use_think_prefill_ ? "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                              : "<|im_start|>assistant\n");

    std::string conv;
    if (gemma) {
        conv  = "<bos>" + user_open + build_tools_instructions() + "\n\n" + query + turn_end;
    } else {
        conv  = "<|im_start|>system\n" + build_tools_instructions() + "<|im_end|>\n";
        conv += user_open + query + turn_end;
    }

    std::string final_answer;
    int total_tokens = 0;

    for (int iter = 0; iter <= config_.max_tool_iters; ++iter) {
        conv += asst_primer;
        std::string out = strip_think(generate(conv));
        if (gemma) out = strip_gemma_markers(out);
        total_tokens += last_token_count_;
        out = trim(out);
        conv += out + turn_end;

        std::vector<json> calls = parse_tool_calls(out);
        final_answer = out;
        if (calls.empty()) break;   // plain answer → done

        // Execute every requested call and group the responses into one user turn.
        std::string tool_turn = user_open;
        for (const auto& call : calls) {
            std::string name = call.value("name", "");
            json args = call.contains("arguments") ? call["arguments"] : json::object();
            json result = registry_.dispatch(name, args);
            if (on_tool) on_tool(ToolTrace{name, args, result});
            tool_turn += "<tool_response>\n" + result.dump() + "\n</tool_response>\n";
        }
        tool_turn += turn_end;
        conv += tool_turn;
    }

    last_token_count_ = total_tokens;
    return final_answer;
}

} // namespace toolchat
