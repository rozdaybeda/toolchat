# Tool-calling benchmark: Qwen3.5 / Qwen3 / Qwen2.5

Measured with `scripts/benchmark_toolcalls.py` (uses `toolchat --bench --seed`).
14 cases × 3 seeds = 42 runs per model, 8 threads, Q4_K_M. CPU = 8-thread CPU
backend; GPU = Vulkan on an RTX 5060 Ti (`-g 99 -d 0`).

Scoring per run:
- **tool selection** — called *exactly* the expected set of tools (penalizes both
  missing and spurious calls),
- **arguments** — when the right tool was called, args are correct (calculator
  expression evaluates to the expected value; path matches),
- **overall** — selection AND arguments both correct.
- No-tool "abstain" cases pass only if the model calls **zero** tools.

## Results

Accuracy is backend-independent (the tool-calling *decision* doesn't depend on
CPU vs GPU — only speed does), so the accuracy columns are the CPU run; GPU
accuracy matched within seed noise.

| Model | Overall | Tool-required | No-tool abstain | CPU latency | GPU latency | Size |
|---|---|---|---|---|---|---|
| Qwen3.5-4B  | **100%** | 100% (33/33) | 100% (9/9) | 13.2 s | **28 s ❌** | 2.7 GB |
| Qwen3-4B    | **100%** | 100% (33/33) | 100% (9/9) | 9.8 s  | **0.7 s** | 2.5 GB |
| Qwen2.5-3B  | 92.9% | 100% (33/33) | 66.7% (6/9) | 8.1 s  | **0.6 s** | 1.9 GB |
| Qwen2.5-1.5B| 92.9% | 90.9% (30/33) | 100% (9/9) | 3.8 s | **0.3 s** | 1.0 GB |
| Qwen2.5-0.5B| 71.4% | 81.8% (27/33) | 33.3% (3/9) | 2.3 s | **0.5 s** | 0.4 GB |

Latency is per-query wall time (incl. tool round-trips), model load excluded,
averaged over 42 runs. GPU = Vulkan on RTX 5060 Ti.

### GPU notes

- **Standard transformers get ~13–14× speedups** on the GPU (Qwen3-4B 9.8→0.7 s,
  Qwen2.5-3B 8.1→0.6 s) — sub-second per query.
- **Qwen3.5-4B regresses on GPU** (28 s, 3 tok/s — *slower* than CPU). Its fused
  Gated Delta Net (linear-attention) ops lack efficient Vulkan kernels at this
  llama.cpp build and fall back with per-token host sync. **Run Qwen3.5 on CPU.**
- Flash Attention is kept on for standard models (off only for Qwen3.5, which
  needs it off to not crash) — re-enabling it was worth a few % throughput on GPU.

## Takeaways

- **Both 4B models are perfect** on this suite (basic tool selection, args, and
  abstaining are solved). Qwen3-4B is ~25% faster than Qwen3.5-4B — the latter's
  hybrid Gated-DeltaNet path runs with Flash Attention disabled on CPU.
- **Qwen2.5-3B is over-eager**: it never misses a needed tool (100%) but calls
  tools when it shouldn't (abstains only 66.7% — e.g. runs `get_current_time` on
  "say hi in French"). And it's barely faster than Qwen3-4B, so it's a poor trade.
- **Qwen2.5-1.5B is the speed/accuracy sweet spot**: **2.6× faster** than
  Qwen3-4B (3.8 s vs 9.8 s), 92.9% overall, and it *never* makes a spurious call.
  Its only weakness is operator precedence — it botches `(100-5)/5` (emits a
  wrong expression), so complex arithmetic is a risk.
- **Qwen2.5-0.5B is too weak** for reliable tool use (71%, abstains only 33%),
  though it's the fastest at 2.3 s.

## Gemma vs Qwen (2026-07-05, GPU)

Head-to-head on the same 14-case suite (3 seeds, Vulkan GPU, Q4_K_M). Gemma uses
a different template (`<start_of_turn>` turns, no system role, no `<think>`),
auto-detected from the `gemma` architecture. The parser is **native-tolerant** —
it accepts our `<tool_call>` JSON, Gemma 4's native `call:fn{...}` form, and a
bare/fenced ` ```json ` object (Gemma 3's style), and it coerces a stringified
`"arguments"` back to an object.

| Model | Overall | Tool-required | No-tool abstain | GPU latency |
|---|---|---|---|---|
| Qwen3-4B     | **100%** | 100% (33/33)  | 100% (9/9) | 0.7 s |
| Qwen2.5-1.5B | 90.5%    | 87.9% (29/33) | 100% (9/9) | **0.3 s** |
| Gemma 4 E4B  | 85.7%    | 81.8% (27/33) | **100%** (9/9) | 1.0 s |
| Gemma 4 E2B  | 73.8%    | 66.7% (22/33) | **100%** (9/9) | 0.9 s |
| Gemma 3 4B   | 71.4%    | **90.9%** (30/33) | **0%** (0/9) | 0.5 s |
| Gemma 3 1B   | 28.6%    | 30.3% (10/33) | 22.2% (2/9) | 1.2 s |

- **Qwen wins.** Gemma 4 E4B (85.7%) trails Qwen3-4B (100%) and even the smaller,
  4× faster Qwen2.5-1.5B (90.5%). At the 4B tier Qwen is both more accurate and
  faster.
- **Gemma 4's misses are tool *selection*, not arguments** (E4B: `list_models` +
  `read_vcpkg`, both wrong tool *set*; E2B: `calc_word`, `calc_parens`,
  `list_scripts`, `read_vcpkg`). Argument correctness stays high, so the format
  handling isn't the bottleneck — it's genuine over-/mis-selection.
- **Gemma 3 4B is a study in over-eagerness.** It has the **highest
  tool-required score of any model (90.9%)** — it almost never misses a needed
  tool — but it **abstains 0% of the time**, firing a tool on every no-tool prompt
  (e.g. `get_current_time()` for "say hi in Spanish", `calculator("Paris")` for
  "capital of France"). Perfect recall, zero precision on the abstain axis →
  71.4% overall. Gemma 4's perfect abstain is the clear generational improvement.
- **Gemma 3 1B is too weak** (28.6%) for reliable tool use.
- **Format caveat (Gemma 3):** it emits calls as a ` ```json ` fenced object with
  `"arguments"` as a *stringified* JSON — a strict `<tool_call>` parser reports
  it as ~0% tool use. The numbers above use the native-tolerant parser; without
  the fenced-JSON + stringified-args handling, Gemma 3 4B measures 18% instead of
  its real 71%.
- **Integration notes:** both Gemma 3 and 4 load/run on Vulkan with no crash
  (unlike Qwen3.5's Gated-DeltaNet). Gemma occasionally renders a literal
  `<end_of_turn>` in the answer — the agent strips it. Qwen numbers are unchanged
  (Qwen uses `<tool_call>` tags, unaffected by the Gemma-only parser paths),
  confirming no regression.

## Recommendation (CPU end user)

- **Max accuracy:** Qwen3-4B (perfect, 9.8 s) — better than Qwen3.5-4B here
  because it's faster at the same accuracy.
- **Fast + still reliable:** **Qwen2.5-1.5B** (3.8 s, 92.9%, zero spurious
  calls) — the best pick when latency matters and tools aren't precision-math.
- Avoid Qwen2.5-3B (over-eager, no speed win) and Qwen2.5-0.5B (unreliable).

## Reproduce

```bash
python scripts/download_model.py    # grab the models you want to compare
python scripts/benchmark_toolcalls.py \
  --exe build/clang-x64-release/toolchat.exe \
  --model "Qwen3-4B=models/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf" \
  --model "Qwen2.5-1.5B=models/Qwen2.5-1.5B-Instruct-Q4_K_M/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
  --seeds 1 2 3 --threads 8
```

The `benchmark_results.json` tally file is written next to the exe (the `--exe`
folder) by default; override with `--out <path>`. Append
`--gpu-layers 99 --device 0` for the GPU numbers.
