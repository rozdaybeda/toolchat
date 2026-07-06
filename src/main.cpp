#include "agent.h"
#include "catalog.h"
#include "download.h"
#include "tools.h"
#include "tui.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#endif

using namespace toolchat;
namespace fs = std::filesystem;

static const char* kTypeLabel[] = { "CPU", "GPU", "iGPU", "other" };

static void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [model_path] [options]\n"
              << "\n"
              << "A local tool-calling chat (Qwen / Gemma + llama.cpp). Ask a question;\n"
              << "the model decides which tools to call, the tools run in-process,\n"
              << "and results are fed back until it produces a final answer.\n"
              << "\n"
              << "Run with no model_path for an interactive menu: pick a model (any\n"
              << "catalog model, downloaded on demand) and a device (CPU / GPU / iGPU).\n"
              << "\n"
              << "Options:\n"
              << "  -t, --threads <n>     Number of CPU threads (default: all cores)\n"
              << "  -c, --ctx <n>         Context size (default: 4096)\n"
              << "  -g, --gpu-layers <n>  GPU layers to offload (default: 0, CPU only)\n"
              << "  -d, --device <n>      GPU device index (0 = first GPU, ...)\n"
              << "      --n-cpu-moe <n>   Keep MoE experts of first n layers on CPU (big MoE, small VRAM)\n"
              << "  -i, --iters <n>       Max tool-call rounds per query (default: 6)\n"
              << "  -s, --seed <n>        Sampler RNG seed (default: 42)\n"
              << "      --think <mode>    Non-thinking prefill: auto|on|off (default: auto by arch)\n"
              << "      --menu            Force the interactive model/device menu\n"
              << "      --fetch <key>     Download a catalog model (by key) and exit\n"
              << "      --bench           Machine-readable output: one JSON object per query\n"
              << "  -h, --help            Show this help\n"
              << "\n"
              << "In-chat commands:  /model  /device  /rerun  /tools  /help  /quit\n"
              << "\n"
              << "Example:\n"
              << "  " << program << "                 # interactive menu\n"
              << "  " << program << " models/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf -g 99 -d 0\n";
}

// --- interactive setup ------------------------------------------------------

// Read a line from stdin, stripping any trailing CR (piped input on Windows
// keeps the \r, which otherwise breaks number parsing and command matching).
static bool read_line(std::string& line) {
    if (!std::getline(std::cin, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

// Prompt for a number in [1, n]; returns 0-based index, or -1 on EOF.
static int prompt_choice(const std::string& label, int n) {
    for (;;) {
        std::cout << label << " [1-" << n << "]: " << std::flush;
        std::string line;
        if (!read_line(line)) return -1;
        try {
            int v = std::stoi(line);
            if (v >= 1 && v <= n) return v - 1;
        } catch (...) {}
        std::cout << "  Please enter a number between 1 and " << n << ".\n";
    }
}

// True when we can drive a full-screen TUI: both stdin and stdout are a real
// terminal (piped/redirected I/O falls back to the numbered prompts).
static bool interactive_terminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) && _isatty(_fileno(stdout));
#else
    return true;
#endif
}

// "Qwen3-4B  (2.5 GB)  [downloaded]" — the picker shows every catalog model with
// its download status (missing models are fetched on selection).
static std::string catalog_label(const ModelEntry& e) {
    std::string s = e.name;
    if (!e.size.empty()) s += "  (" + e.size + ")";
    s += e.downloaded ? "  [downloaded]"
                      : (e.repo.empty() ? "  [missing]" : "  [not downloaded]");
    return s;
}

static std::vector<std::string> catalog_labels(const std::vector<ModelEntry>& cat) {
    std::vector<std::string> out;
    for (const auto& e : cat) out.push_back(catalog_label(e));
    return out;
}

// "temp 0.7, top-p 1, min-p 0.01, ctx 8192" — the set fields of a model's
// manifest params; empty when the model has no overrides.
static std::string params_summary(const ModelParams& p) {
    std::ostringstream s;
    auto add = [&](const char* label, auto v) {
        if (v >= 0) s << (s.tellp() > 0 ? ", " : "") << label << " " << v;
    };
    add("temp",      p.temperature);
    add("top-p",     p.top_p);
    add("min-p",     p.min_p);
    add("top-k",     p.top_k);
    add("ctx",       p.n_ctx);
    add("n-cpu-moe", p.n_cpu_moe);
    return s.str();
}

// Detail blurb shown under the picker for the highlighted model: description,
// minimum hardware requirements, and any per-model llama params, one line each
// (lines with nothing to show are omitted).
static std::string catalog_detail(const ModelEntry& e) {
    std::string d = e.desc;
    auto line = [&](const std::string& text) {
        if (!text.empty()) d += (d.empty() ? "" : "\n") + text;
    };
    line(e.req.empty() ? "" : "Needs: " + e.req);
    std::string ps = params_summary(e.params);
    line(ps.empty() ? "" : "Params: " + ps);
    return d;
}

static std::vector<std::string> catalog_details(const std::vector<ModelEntry>& cat) {
    std::vector<std::string> out;
    for (const auto& e : cat) out.push_back(catalog_detail(e));
    return out;
}

// Overlay a chosen model's manifest params onto the config: tunables first reset
// to `base` (defaults + CLI flags) so switching models never inherits a previous
// model's overrides, then each set (>= 0) param is applied. Explicit CLI flags
// win: -c keeps its n_ctx, --n-cpu-moe its layer count.
static void apply_model_params(AgentConfig& config, const ModelEntry& e,
                               const AgentConfig& base,
                               bool cli_ctx_set, bool cli_moe_set) {
    config.temperature = base.temperature;
    config.top_p       = base.top_p;
    config.min_p       = base.min_p;
    config.top_k       = base.top_k;
    config.n_ctx       = base.n_ctx;
    config.n_cpu_moe   = base.n_cpu_moe;

    const ModelParams& p = e.params;
    if (p.temperature >= 0) config.temperature = p.temperature;
    if (p.top_p       >= 0) config.top_p       = p.top_p;
    if (p.min_p       >= 0) config.min_p       = p.min_p;
    if (p.top_k       >= 0) config.top_k       = p.top_k;
    if (p.n_ctx       >= 0 && !cli_ctx_set) config.n_ctx     = p.n_ctx;
    if (p.n_cpu_moe   >= 0 && !cli_moe_set) config.n_cpu_moe = p.n_cpu_moe;
}

// Default selection: the first already-downloaded model, else 0.
static int default_model_index(const std::vector<ModelEntry>& cat) {
    for (int i = 0; i < (int)cat.size(); ++i)
        if (cat[i].downloaded) return i;
    return 0;
}

// One-line download progress bar, rewritten in place (throttled to ~5/sec).
static void print_progress(uint64_t done, uint64_t total,
                           std::chrono::steady_clock::time_point t0) {
    using namespace std::chrono;
    static steady_clock::time_point last;
    auto now = steady_clock::now();
    if (done < total && duration_cast<milliseconds>(now - last).count() < 200) return;
    last = now;
    double secs  = duration<double>(now - t0).count();
    double mb    = done / 1048576.0;
    double speed = secs > 0 ? mb / secs : 0.0;
    if (total > 0) {
        int pct = (int)(100.0 * done / total);
        int width = 24, filled = pct * width / 100;
        std::string bar(filled, '#');
        bar += std::string(width - filled, '.');
        std::printf("\r  [%s] %3d%%  %.0f/%.0f MB  %.1f MB/s   ",
                    bar.c_str(), pct, mb, total / 1048576.0, speed);
    } else {
        std::printf("\r  %.0f MB  %.1f MB/s   ", mb, speed);
    }
    std::fflush(stdout);
}

// Download the model if it isn't on disk yet. Returns false on failure.
static bool ensure_downloaded(ModelEntry& e, std::ostream& log) {
    if (e.downloaded) return true;
    if (e.repo.empty()) {
        std::cerr << "Model file is missing and has no download source: "
                  << e.local_path << "\n";
        return false;
    }
    log << "Downloading " << e.name << " (" << e.size << ") from " << e.repo << " ...\n";
    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    bool ok = download_file(hf_resolve_url(e.repo, e.file), e.local_path, err,
                            [&](uint64_t d, uint64_t t) { print_progress(d, t, t0); });
    std::cout << "\n";
    if (!ok) {
        std::cerr << "Download failed: " << err << "\n";
        return false;
    }
    e.downloaded = true;
    log << "Saved to " << e.local_path << "\n";
    return true;
}

// "NVIDIA GeForce RTX 5060 Ti  [GPU]" (CPU entries also show the thread count).
static std::string device_label(const DeviceInfo& d, int n_threads) {
    std::string s = (d.description.empty() ? d.name : d.description);
    s += "  [" + std::string(kTypeLabel[d.type]) + "]";
    if (d.type == 0) s += "  (" + std::to_string(n_threads) + " threads)";
    return s;
}

static std::vector<std::string> make_device_labels(const std::vector<DeviceInfo>& devs,
                                                   int n_threads) {
    std::vector<std::string> out;
    for (const auto& d : devs) out.push_back(device_label(d, n_threads));
    return out;
}

// Apply a chosen device to the config: CPU → threads; GPU/iGPU → offload all.
static void apply_device_choice(AgentConfig& config, const DeviceInfo& d) {
    if (d.type == 0) {          // CPU
        config.device       = nullptr;
        config.n_gpu_layers = 0;
    } else {                    // GPU / iGPU
        config.device       = d.handle;
        config.n_gpu_layers = 99;
    }
}

// Default TUI selection: first GPU if present, else the first device (CPU).
static int default_device_index(const std::vector<DeviceInfo>& devs) {
    for (int i = 0; i < (int)devs.size(); ++i)
        if (devs[i].type == 1) return i;
    return 0;
}

// Numbered (non-TTY) model picker over the catalog. Returns the chosen index
// into `cat`, or -1 on cancel/EOF/empty.
static int choose_model_index(const std::vector<ModelEntry>& cat) {
    if (cat.empty()) {
        std::cerr << "No models available (need scripts/models.json or a .gguf under models/).\n";
        return -1;
    }
    std::cout << "Select a model:\n";
    for (size_t i = 0; i < cat.size(); ++i) {
        std::cout << "  " << (i + 1) << ") " << catalog_label(cat[i]) << "\n";
        if (!cat[i].desc.empty()) std::cout << "       " << cat[i].desc << "\n";
        if (!cat[i].req.empty())  std::cout << "       needs: " << cat[i].req << "\n";
        std::string ps = params_summary(cat[i].params);
        if (!ps.empty())          std::cout << "       params: " << ps << "\n";
    }
    return prompt_choice("Model", (int)cat.size());
}

// Interactive device picker. Sets config.device / n_gpu_layers and writes the
// chosen device's display label to `label`. Returns false on EOF.
static bool choose_device(AgentConfig& config, std::string& label) {
    auto devs = enumerate_devices();
    if (devs.empty()) return true;  // nothing to pick; stay on default CPU
    std::cout << "\nSelect a device:\n";
    for (size_t i = 0; i < devs.size(); ++i)
        std::cout << "  " << (i + 1) << ") " << device_label(devs[i], config.n_threads) << "\n";
    int di = prompt_choice("Device", (int)devs.size());
    if (di < 0) return false;
    apply_device_choice(config, devs[di]);
    label = device_label(devs[di], config.n_threads);
    return true;
}

// --- agent (re)loading ------------------------------------------------------

// Build a fresh registry and (re)initialize the agent from the current config.
static bool load_agent(Agent& agent, AgentConfig& config, std::ostream& log) {
    ToolRegistry registry;
    register_builtin_tools(registry);
    log << "Loading model: " << config.model_path << " ...\n";
    Agent fresh;
    if (!fresh.init(config, std::move(registry))) {
        std::cerr << "Error: " << fresh.get_error() << "\n";
        return false;
    }
    agent = std::move(fresh);
    log << "Model loaded.\n";
    return true;
}

// Device label synthesized from the config alone — used on the scriptable path
// (model + -g/-d flags) where no device was picked from the enumerated list.
static std::string config_device_label(const AgentConfig& config) {
    if (config.n_gpu_layers == 0)
        return "CPU  (" + std::to_string(config.n_threads) + " threads)";
    return "GPU device " + std::to_string(config.main_gpu)
         + "  (" + std::to_string(config.n_gpu_layers) + " layers offloaded)";
}

// Show the active model and device after (re)loading. `dev_label` is the picked
// device's display label; empty on the scriptable path, where we derive one.
static void print_status(const AgentConfig& config, const std::string& dev_label,
                         std::ostream& log) {
    log << "Model:  " << fs::path(config.model_path).filename().string() << "\n";
    log << "Device: " << (dev_label.empty() ? config_device_label(config) : dev_label) << "\n";
    log << "Params: temp " << config.temperature << ", top-p " << config.top_p;
    if (config.min_p > 0) log << ", min-p " << config.min_p;
    log << ", top-k " << config.top_k << ", ctx " << config.n_ctx;
    if (config.n_cpu_moe > 0) log << ", n-cpu-moe " << config.n_cpu_moe;
    log << "\n";
}

static void print_tools(const Agent& agent) {
    std::cout << "Available tools:\n";
    for (const auto& t : agent.registry().tools())
        std::cout << "  - " << t.name << " : " << t.description << "\n";
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::string model_path;
    std::string fetch_key;  // --fetch <key>: download a catalog model and exit
    AgentConfig config;
    bool bench = false;    // machine-readable JSON output, one object per query
    bool menu  = false;    // force the interactive picker
    bool cli_ctx_set = false;  // -c given: wins over a model's n_ctx override
    bool cli_moe_set = false;  // --n-cpu-moe given: wins over a model's override

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            config.n_threads = std::stoi(argv[++i]);
        } else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) {
            config.n_ctx = std::stoi(argv[++i]);
            cli_ctx_set = true;
        } else if ((arg == "-g" || arg == "--gpu-layers") && i + 1 < argc) {
            config.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--n-cpu-moe" && i + 1 < argc) {
            config.n_cpu_moe = std::stoi(argv[++i]);
            cli_moe_set = true;
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            config.main_gpu = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--iters") && i + 1 < argc) {
            config.max_tool_iters = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
            config.seed = (uint32_t)std::stoul(argv[++i]);
        } else if (arg == "--think" && i + 1 < argc) {
            std::string v = argv[++i];
            config.think_prefill = (v == "on") ? 1 : (v == "off") ? 0 : -1;  // else auto
        } else if (arg == "--menu") {
            menu = true;
        } else if (arg == "--fetch" && i + 1 < argc) {
            fetch_key = argv[++i];
        } else if (arg == "--bench") {
            bench = true;
        } else if (model_path.empty() && arg[0] != '-') {
            model_path = arg;
        }
    }

    // In bench mode, keep stdout clean (JSON only) — banner goes to stderr.
    std::ostream& log = bench ? std::cerr : std::cout;

    // --fetch <key>: download a catalog model (by key or name) and exit.
    if (!fetch_key.empty()) {
        auto catalog = load_catalog();
        auto it = std::find_if(catalog.begin(), catalog.end(), [&](const ModelEntry& e) {
            return e.key == fetch_key || e.name == fetch_key;
        });
        if (it == catalog.end()) {
            std::cerr << "Unknown model key: " << fetch_key << "\nAvailable:\n";
            for (const auto& e : catalog)
                if (!e.key.empty()) std::cerr << "  " << e.key << "  (" << e.name << ")\n";
            return 1;
        }
        return ensure_downloaded(*it, std::cout) ? 0 : 1;
    }

    // Display label of the active device (set when picked from the enumerated
    // list; empty on the scriptable path, where print_status derives one).
    std::string cur_device_label;

    // Snapshot of the tunables as set by defaults + CLI flags — the baseline that
    // apply_model_params resets to on every model switch, so one model's manifest
    // overrides never leak into the next.
    const AgentConfig base_config = config;
    bool params_applied = false;  // model params already applied (menu path)

    // Interactive setup: run when no model was given (or --menu was passed) and
    // we're not in bench mode. Scripted/bench usage always passes a model path,
    // so it skips the menu and behaves exactly as before. On a real terminal we
    // use the ftxui form; piped/redirected I/O falls back to numbered prompts.
    if (!bench && (model_path.empty() || menu)) {
        auto catalog = load_catalog();
        if (catalog.empty()) {
            std::cerr << "No models available. Add scripts/models.json or a .gguf under models/.\n";
            return 1;
        }
        auto devices = enumerate_devices();
        int mi = default_model_index(catalog);
        int di = default_device_index(devices);

        if (interactive_terminal()) {
            if (!run_setup_tui(catalog_labels(catalog), catalog_details(catalog),
                               make_device_labels(devices, config.n_threads),
                               mi, di)) {
                return 0;  // user cancelled
            }
        } else {
            mi = choose_model_index(catalog);
            if (mi < 0) return 1;
            if (!choose_device(config, cur_device_label)) return 1;
            di = -1;  // choose_device already applied the config
        }

        if (di >= 0 && di < (int)devices.size()) {
            apply_device_choice(config, devices[di]);
            cur_device_label = device_label(devices[di], config.n_threads);
        }
        // Fetch the model if it isn't on disk yet, then use its local path.
        if (!ensure_downloaded(catalog[mi], log)) return 1;
        model_path = catalog[mi].local_path;
        apply_model_params(config, catalog[mi], base_config, cli_ctx_set, cli_moe_set);
        params_applied = true;
        std::cout << "\n";
    }

    if (model_path.empty()) {
        std::cerr << "Error: Model path required\n";
        print_usage(argv[0]);
        return 1;
    }
    config.model_path = model_path;

    // Scriptable path (model given on the command line): if the path is a
    // catalog model, pick up its manifest params too (the menu path already
    // applied them, flagged by params_applied).
    if (!params_applied) {
        std::error_code ec;
        for (const auto& e : load_catalog()) {
            if (!e.downloaded) continue;
            if (fs::equivalent(fs::path(e.local_path), fs::path(model_path), ec) && !ec) {
                apply_model_params(config, e, base_config, cli_ctx_set, cli_moe_set);
                break;
            }
        }
    }

    Agent agent;
    if (!load_agent(agent, config, log)) return 1;
    print_status(config, cur_device_label, log);

    if (!bench) {
        std::cout << "\n";
        print_tools(agent);
        std::cout << "\nAsk a question. Commands: /model /device /rerun /tools /help /quit\n\n";
    }

    std::string last_query;  // remembered for /rerun

    std::string line;
    while (true) {
        if (!bench) std::cout << "> ";
        if (!read_line(line)) break;
        if (line.empty()) continue;

        // Bare exit/quit words (typed without a leading slash) also leave.
        if (!bench && (line == "exit" || line == "quit"))
            break;

        // /rerun re-runs the previous query under the current model/device — set
        // `line` and fall through to normal execution (do NOT continue).
        if (!bench && line == "/rerun") {
            if (last_query.empty()) {
                std::cout << "No previous query to re-run.\n\n";
                continue;
            }
            line = last_query;
            std::cout << line << "\n";  // echo the query being re-run
        } else if (!bench && line[0] == '/') {
            // Other in-chat commands (interactive mode only).
            if (line == "/quit" || line == "/exit") {
                break;
            } else if (line == "/help") {
                std::cout << "Commands:\n"
                             "  /model   switch to a different model (reloads)\n"
                             "  /device  switch CPU / GPU / iGPU (reloads)\n"
                             "  /rerun   re-run the previous query on the current model/device\n"
                             "  /tools   list the available tools\n"
                             "  /help    show this help\n"
                             "  /quit    exit  (or type 'exit' / 'quit')\n\n";
            } else if (line == "/tools") {
                print_tools(agent);
                std::cout << "\n";
            } else if (line == "/model") {
                auto catalog = load_catalog();
                int mi = -1;
                if (interactive_terminal()) {
                    if (catalog.empty()) std::cerr << "No models available.\n";
                    else mi = tui_menu("Select model", catalog_labels(catalog),
                                       default_model_index(catalog),
                                       catalog_details(catalog));
                } else {
                    mi = choose_model_index(catalog);
                }
                // Fetch on demand, then reload.
                if (mi >= 0 && ensure_downloaded(catalog[mi], log)) {
                    config.model_path = catalog[mi].local_path;
                    apply_model_params(config, catalog[mi], base_config,
                                       cli_ctx_set, cli_moe_set);
                    load_agent(agent, config, log);
                    print_status(config, cur_device_label, log);
                    std::cout << "\n";
                }
            } else if (line == "/device") {
                bool picked = false;
                if (interactive_terminal()) {
                    auto devices = enumerate_devices();
                    int di = tui_menu("Select device",
                                      make_device_labels(devices, config.n_threads),
                                      default_device_index(devices));
                    if (di >= 0) {
                        apply_device_choice(config, devices[di]);
                        cur_device_label = device_label(devices[di], config.n_threads);
                        picked = true;
                    }
                } else {
                    picked = choose_device(config, cur_device_label);
                }
                if (picked) {
                    load_agent(agent, config, log);
                    print_status(config, cur_device_label, log);
                    std::cout << "\n";
                }
            } else {
                std::cout << "Unknown command: " << line << "  (try /help)\n\n";
            }
            continue;
        }

        last_query = line;  // remember for /rerun

        std::vector<ToolTrace> traces;
        auto t0 = std::chrono::steady_clock::now();
        std::string answer = agent.run(line, [&](const ToolTrace& tr) {
            traces.push_back(tr);
            if (!bench) {
                std::cout << "  \xe2\x86\x92 tool_call " << tr.name << "(" << tr.arguments.dump() << ")\n";
                std::cout << "  \xe2\x86\x90 " << tr.result.dump() << "\n";
            }
        });
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        int n_tokens = agent.get_last_token_count();

        if (bench) {
            // One compact JSON object per query, on its own line.
            nlohmann::json calls = nlohmann::json::array();
            for (const auto& tr : traces)
                calls.push_back({{"name", tr.name}, {"arguments", tr.arguments}});
            nlohmann::json out = {
                {"query", line},
                {"tool_calls", calls},
                {"answer", answer},
                {"ms", ms},
                {"tok", n_tokens},
            };
            std::cout << out.dump() << std::endl;
        } else if (answer.empty() && !agent.get_error().empty()) {
            std::cerr << "Error: " << agent.get_error() << "\n";
        } else {
            double tok_per_sec = ms > 0 ? n_tokens * 1000.0 / ms : 0.0;
            std::cout << "\n" << answer << "\n";
            std::cout << "  (" << ms << " ms, " << n_tokens << " tok, "
                      << (int)tok_per_sec << " tok/s)\n\n";
        }
    }

    return 0;
}
