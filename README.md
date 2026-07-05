# toolchat

A local **tool-calling** (function-calling) agent in C++ powered by
[llama.cpp](https://github.com/ggerganov/llama.cpp) and **Qwen** (2.5 / 3 / 3.5).
Ask a question; the model decides which tools to call, the tools run **in-process
on your CPU**, results are fed back to the model, and the loop repeats until it
produces a final answer.

Its purpose is to **exercise and validate LLM tool calling locally** — no API
keys, no server, runs on an end-user CPU (optional GPU offload).

## Models

The agent is model-agnostic: it hand-builds the chat template and parses tool
calls itself, auto-detecting the family from the GGUF's architecture — **Qwen**
(ChatML) and **Gemma** (`<start_of_turn>` turns) are both supported. Any Qwen
instruct GGUF is the recommended path; Gemma 4 is wired up too (see the
[Gemma vs Qwen](#gemma-4-vs-qwen-2026-07-05-gpu-14-cases--3-seeds) comparison —
Qwen wins on tool calling). Picking one is a **speed vs. accuracy** trade — see
the [benchmark](#benchmarks) below. Short version:

| Model | Size (Q4_K_M) | Tool-call accuracy | Latency/query (CPU) | Best for |
|---|---|---|---|---|
| [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | ~2.7 GB | **100%** | 13.2 s | Newest; needs the b9871 build (below) |
| [Qwen3-4B](https://huggingface.co/unsloth/Qwen3-4B-GGUF) | ~2.5 GB | **100%** | 9.8 s | **Max accuracy** (perfect and faster than 3.5) |
| [Qwen2.5-3B](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF) | ~1.9 GB | 92.9% | 8.1 s | Skip — over-eager, no speed win |
| [Qwen2.5-1.5B](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF) | ~1.0 GB | 92.9% | **3.8 s** | **Speed-first** (2.6× faster, zero spurious calls) |
| [Qwen2.5-0.5B](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF) | ~0.4 GB | 71.4% | 2.3 s | Too weak for reliable tool use |

**Recommendations:** **Qwen3-4B** for maximum accuracy, **Qwen2.5-1.5B** when
latency matters. (Qwen2.5 has no thinking mode; the agent auto-detects the
architecture and adjusts — no flags needed.)

Qwen3.5 (Small series, Feb–Mar 2026) is the newest line. Running it required two
things, both already done in this repo:

### 1. Bumped llama.cpp (vcpkg port → b9871)

Qwen3.5's `qwen35` architecture was added to llama.cpp *after* the stock vcpkg
`llama-cpp` port (build `b7146`), which fails with
`unknown model architecture: 'qwen35'`. The vcpkg port at
`$VCPKG_ROOT/ports/llama-cpp` was bumped to **b9871** and switched to build
llama.cpp's own **bundled ggml** (so llama + ggml are from the same commit and
consistent for the new architecture). Key portfile changes: new `REF`/`SHA512`,
`-DLLAMA_USE_SYSTEM_GGML=OFF`, `-DLLAMA_BUILD_APP=OFF`, `-DGGML_OPENMP=OFF`, and a
second `vcpkg_cmake_config_fixup` for the bundled ggml. See
`docs/vcpkg-llama-cpp-b9871.md` for the exact diff to reproduce.

### 2. Flash Attention disabled

Qwen3.5-4B is a hybrid model that uses a **fused Gated Delta Net** (linear
attention). With Flash Attention auto-enabled, that CPU path segfaults at this
build, so `src/agent.cpp` sets
`ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`. (Harmless for
plain Qwen3 too.)

## Requirements

- CMake 3.21+
- Clang (LLVM) — Windows builds use clang with MSVC ABI
- [vcpkg](https://github.com/microsoft/vcpkg) with `VCPKG_ROOT` set
- Ninja
- Python 3.8+ (for model download)
- Vulkan SDK (optional, for GPU builds)

## Download a model

You don't have to download anything up front — the interactive picker
([below](#interactive-chat-tui)) lists every catalog model with a
`[downloaded]` / `[not downloaded]` marker and **fetches a missing one on the
spot** (in-process, with a progress bar). The catalog lives in
[`scripts/models.json`](scripts/models.json) (shared by the app and the Python
script).

To pre-fetch from the command line:

```bash
# in-app downloader (no Python needed), by catalog key:
.\build\clang-x64-vulkan-release\toolchat.exe --fetch qwen3-4b

# or the Python helper:
python scripts/download_model.py       # choose a model
```

Or manually:

```bash
pip install huggingface_hub
huggingface-cli download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf \
  --local-dir models/Qwen3.5-4B-Q4_K_M
```

## Build

vcpkg dependencies (`llama-cpp`, `nlohmann-json`) install automatically on first
configure.

### CPU only (the target use case)

```bash
cmake --preset clang-x64-release
cmake --build --preset clang-x64-release
```

Output: `build/clang-x64-release/toolchat.exe`

### With Vulkan GPU support

Requires the Vulkan SDK installed (`glslc` on PATH). The bumped `llama-cpp` port
has a `vulkan` feature that builds the bundled ggml with `GGML_VULKAN` against the
system SDK — the preset enables it automatically.

```bash
cmake --preset clang-x64-vulkan-release
cmake --build --preset clang-x64-vulkan-release
```

Run with `-g 99` (offload all layers) and `-d <n>` to pick the GPU:

```bash
.\build\clang-x64-vulkan-release\toolchat.exe models\Qwen3-4B-Q4_K_M\Qwen3-4B-Q4_K_M.gguf -g 99 -d 0
```

On startup it lists the Vulkan devices it found; `-d 0` is the first. Standard
models are ~13–14× faster on GPU (see [Benchmarks](#benchmarks)). **Exception:
run Qwen3.5 on CPU** — its Gated-DeltaNet path has no efficient Vulkan kernels and
is *slower* on GPU.

## Usage

### Interactive chat (TUI)

Run with **no model path** and you get a terminal UI (built with
[ftxui](https://github.com/ArthurSonzogni/FTXUI)) to pick a model and a device,
then it drops you into the chat:

```bash
.\build\clang-x64-vulkan-release\toolchat.exe
```

```
┌ toolchat ──────────────────────────────────┐
│ Model:  [ Gemma 4 E4B (2.9 GB) [not downl… ▾ ]│
│ Device:                                      │
│   ( ) NVIDIA GeForce RTX 5060 Ti             │
│   (•) AMD Radeon 780M Graphics               │
│   ( ) AMD Ryzen 7 8745H  (16 thr)            │
│              [ Start chat ]                   │
└──────────────────────────────────────────────┘
  Up/Down move · Enter select · Esc cancel
```

Use ↑/↓ to move, Enter to open the model dropdown / pick a device, and
**Start chat** to load. The **model dropdown lists every catalog model**
(from `scripts/models.json`) plus any extra `.gguf` in `models/`, each tagged
`[downloaded]` or `[not downloaded]`. If you pick one that isn't downloaded yet,
it's **fetched in-process** (WinHTTP, with a progress bar) before loading — no
need to run the Python script first. Devices are enumerated live from ggml's
backend registry, so the list reflects what's actually on the machine (the
CPU-only build shows just the CPU; the Vulkan build adds every GPU/iGPU, each
tagged `[GPU]` / `[iGPU]` / `[CPU]`). Picking a GPU/iGPU offloads all layers to
*that* device; picking CPU runs on threads.

If stdin/stdout isn't a real terminal (piped input, CI), it falls back to a
plain numbered menu automatically.

Inside the chat you can switch on the fly (these reopen the TUI picker):

```
/model    pick a different model (reloads)
/device   switch CPU / GPU / iGPU (reloads)
/rerun    re-run the previous query on the current model/device
/tools    list the available tools
/help     show commands
/quit     exit  (or type 'exit' / 'quit')
```

The active model and device are printed after every load, so `/model` or
`/device` followed by `/rerun` lets you replay the same query to compare
models or CPU-vs-GPU behaviour side by side.

### Direct (scriptable)

Pass a model path (and optional flags) to skip the menu:

```bash
.\build\clang-x64-release\toolchat.exe models\Qwen3.5-4B-Q4_K_M\Qwen3.5-4B-Q4_K_M.gguf -t 8
```

Force the menu even with flags set using `--menu`.

### Options

```
-t, --threads <n>     CPU threads (default: all cores)
-c, --ctx <n>         Context size (default: 4096)
-g, --gpu-layers <n>  GPU layers to offload (0 = CPU only, 99 = all)
-d, --device <n>      GPU device index (0 = first GPU, ...)
-i, --iters <n>       Max tool-call rounds per query (default: 6)
-s, --seed <n>        Sampler RNG seed (default: 42)
    --think <mode>    Non-thinking prefill: auto|on|off (default: auto by arch)
    --menu            Force the interactive model/device menu
    --fetch <key>     Download a catalog model (by key, e.g. qwen3-4b) and exit
    --bench           Machine-readable output: one JSON object per query
```

Run with no `model_path` to get the interactive menu (model + device picker).

### Example session

```
Available tools:
  - get_current_time : Get the current local date and time.
  - calculator : Evaluate an arithmetic expression with + - * / and parentheses.
  - list_directory : List the files and subdirectories in a directory.
  - read_file : Read the contents of a text file.

> What time is it, and what is 23 * 19 + 7?
  → tool_call get_current_time({})
  ← {"current_time":"2026-07-04 18:55:01"}
  → tool_call calculator({"expression":"23 * 19 + 7"})
  ← {"expression":"23 * 19 + 7","result":444}

It's currently 2026-07-04 18:55:01, and 23 * 19 + 7 = 444.
  (15839 ms, 86 tok, 5 tok/s)
```

(CPU inference of a 4B model is a few tok/s; smaller models are far faster — see
below.)

## Benchmarks

Tool-calling accuracy and latency, measured with `scripts/benchmark_toolcalls.py`
(14 cases × 3 seeds = 42 runs/model, CPU, 8 threads, Q4_K_M). "Overall" = correct
tool set **and** correct arguments; no-tool cases pass only if the model calls
zero tools. Full methodology and per-case breakdown in
[`docs/benchmark.md`](docs/benchmark.md).

Latency is shown for both the CPU backend (8 threads) and the Vulkan GPU backend
(RTX 5060 Ti). Accuracy is backend-independent.

| Model | Overall | Tool-required | No-tool abstain | CPU | GPU |
|---|---|---|---|---|---|
| Qwen3.5-4B | **100%** | 100% | 100% | 13.2 s | 28 s ❌ |
| Qwen3-4B | **100%** | 100% | 100% | 9.8 s | **0.7 s** |
| Qwen2.5-3B | 92.9% | 100% | 66.7% | 8.1 s | **0.6 s** |
| Qwen2.5-1.5B | 92.9% | 90.9% | 100% | 3.8 s | **0.3 s** |
| Qwen2.5-0.5B | 71.4% | 81.8% | 33.3% | 2.3 s | **0.5 s** |

Highlights:

- **Both 4B models are perfect** here; Qwen3-4B is ~25% faster than Qwen3.5-4B
  on CPU (Qwen3.5's Gated-DeltaNet path runs with Flash Attention off).
- **Qwen2.5-3B is over-eager** — never misses a needed tool but calls tools when
  it shouldn't (abstains only 67%), and it's barely faster than Qwen3-4B on CPU.
- **Qwen2.5-1.5B** is the speed/accuracy sweet spot: no spurious calls; only
  weakness is operator precedence (botches `(100-5)/5`).
- **Qwen2.5-0.5B** is too weak for reliable tool use.
- **On GPU, standard models are ~13–14× faster** (sub-second), but **Qwen3.5-4B
  regresses badly** — its Gated-DeltaNet ops have no efficient Vulkan kernels at
  this build. Run Qwen3.5 on CPU, everything else on GPU.

### Gemma vs Qwen (2026-07-05, GPU, 14 cases × 3 seeds)

The agent also supports **Gemma** (auto-detected `gemma` template — no system role,
tools fold into the first user turn). The tool-call parser is native-tolerant: it
accepts our `<tool_call>` JSON, Gemma 4's native `call:fn{…}` form, **and** a
bare/fenced ` ```json ` object with a stringified-args quirk (which is how Gemma 3
emits calls — without this it scores near zero). Head-to-head:

| Model | Overall | Tool-required | No-tool abstain | GPU latency |
|---|---|---|---|---|
| Qwen3-4B | **100%** | 100% (33/33) | 100% (9/9) | 0.7 s |
| Qwen2.5-1.5B | 90.5% | 87.9% (29/33) | 100% (9/9) | **0.3 s** |
| Gemma 4 E4B | 85.7% | 81.8% (27/33) | **100%** (9/9) | 1.0 s |
| Gemma 4 E2B | 73.8% | 66.7% (22/33) | **100%** (9/9) | 0.9 s |
| Gemma 3 4B | 71.4% | **90.9%** (30/33) | **0%** (0/9) | 0.5 s |
| Gemma 3 1B | 28.6% | 30.3% (10/33) | 22.2% (2/9) | 1.2 s |

- **Qwen wins for tool calling.** Gemma 4 E4B trails Qwen3-4B *and* the smaller,
  4× faster Qwen2.5-1.5B on overall accuracy.
- **Gemma 4 has the judgment; Gemma 3 doesn't.** Gemma 4 (both sizes) abstains
  perfectly on no-tool prompts. **Gemma 3 4B is the opposite extreme** — it's the
  *best* of any model at firing a needed tool (90.9%) but **abstains 0% of the
  time**: it calls a tool on literally every no-tool prompt (e.g.
  `calculator("Paris")` for "capital of France"). Great recall, no discipline →
  71.4% overall. This over-eagerness is the clear Gemma-3→4 generational fix.
- **Gemma 3 1B is too weak** for reliable tool use (28.6%).
- **Format matters for Gemma 3:** it wraps its call in a ` ```json ` fence with
  `"arguments"` as a *string*, so a strict `<tool_call>` parser would report ~0%.
  The measured numbers use the native-tolerant parser (see above).

Reproduce:

```bash
python scripts/benchmark_toolcalls.py \
  --exe build/clang-x64-release/toolchat.exe \
  --model "Qwen3-4B=models/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf" \
  --model "Qwen2.5-1.5B=models/Qwen2.5-1.5B-Instruct-Q4_K_M/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
  --seeds 1 2 3 --threads 8
```

The raw tallies JSON is written next to the exe (the `--exe` folder) by default;
pass `--out <path>` to write it elsewhere. Add `--gpu-layers 99 --device 0` to
benchmark on the GPU.

## Built-in tools

| Tool | Arguments | Returns |
|---|---|---|
| `get_current_time` | — | Local date/time string |
| `calculator` | `expression: string` | Result of `+ - * / ( )` arithmetic |
| `list_directory` | `path: string` | Entries (name, type, size) |
| `read_file` | `path: string`, `max_bytes?: int` | File contents (truncated) |

Add your own by pushing a `Tool{name, description, parameters, handler}` into the
`ToolRegistry` in `src/tools.cpp` (`register_builtin_tools`).

## How tool calling works here

The vcpkg `llama-cpp` port exposes only the low-level `llama` library — not
llama.cpp's `common` helpers with the built-in tool-call parser. So this project
does it explicitly, which also makes the mechanics easy to inspect:

1. **Prompt** — `src/agent.cpp` builds the chat template by hand and embeds each
   tool's JSON schema inside `<tools></tools>`. Two families are supported, chosen
   from `general.architecture`: **Qwen** ChatML (`<|im_start|>` roles, a dedicated
   `system` turn) and **Gemma** (`<start_of_turn>` turns, `<bos>`, no system role —
   the tool instructions fold into the first user turn).
2. **Generate** — the model emits
   `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
3. **Parse & dispatch** — `Agent::parse_tool_calls` extracts each block;
   `ToolRegistry::dispatch` runs the matching C++ handler. For Gemma it also
   accepts the native `call:fn{...}` micro-syntax as a fallback.
4. **Feed back** — the JSON result is appended as a `<tool_response>` user turn
   and the model is run again, looping until it returns a plain answer (capped by
   `--iters`).

The model runs in **non-thinking mode** for fast, deterministic tool calls. For
Qwen3/Qwen3.5 the agent prefills an empty `<think></think>` block (their
enable-thinking-off convention); Qwen2.5 and Gemma have no thinking mode, so it's
skipped. This is auto-selected from the model's `general.architecture` (override
with `--think auto|on|off`), and any stray `<think>...</think>` output is stripped
before parsing.

## License

- Qwen3.5: see the model card on Hugging Face
- llama.cpp: MIT
