#!/usr/bin/env python3
"""Benchmark tool-calling accuracy of two GGUF models with toolchat --bench.

For each model we run every case in the suite across several RNG seeds and score:
  - tool selection : did it call exactly the expected set of tools?
  - arguments      : (when the right tool was called) are the args correct?
  - overall        : selection AND arguments both correct
No-tool cases (expect == []) pass only if the model calls zero tools.

Usage:
  python scripts/benchmark_toolcalls.py \
      --exe build/clang-x64-release/toolchat.exe \
      --model "Qwen3.5-4B=models/Qwen3.5-4B-Q4_K_M/Qwen3.5-4B-Q4_K_M.gguf" \
      --model "Qwen3-4B=models/Qwen3-4B-Q4_K_M/Qwen3-4B-Q4_K_M.gguf" \
      --seeds 1 2 3 --threads 8
"""
import argparse
import ast
import json
import operator
import subprocess
import sys
from pathlib import Path

# --- safe arithmetic evaluator for checking calculator arguments -------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def safe_eval(expr):
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant):   return node.value
        if isinstance(node, ast.BinOp):      return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):    return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported")
    return ev(ast.parse(str(expr), mode="eval"))

# --- argument checkers (return True if the call's args are acceptable) -------
# Some models emit "arguments" as a stringified JSON object; coerce to a dict.
def _as_dict(args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    return args if isinstance(args, dict) else {}

def calc_equals(value):
    def check(args):
        try:
            return abs(safe_eval(_as_dict(args).get("expression", "")) - value) < 1e-6
        except Exception:
            return False
    return check

def path_contains(sub):
    def check(args):
        return sub.lower() in str(_as_dict(args).get("path", "")).lower()
    return check

# --- test suite --------------------------------------------------------------
# expect: set of tool names that must be called (and only those).
# arg_check: optional {tool_name: fn(args)->bool} validated when that tool is called.
SUITE = [
    # ---- calculator ----
    dict(id="calc_basic",   prompt="What is 23 * 19 + 7?",
         expect=["calculator"], arg_check={"calculator": calc_equals(444)}),
    dict(id="calc_parens",  prompt="Compute (100 - 5) / 5.",
         expect=["calculator"], arg_check={"calculator": calc_equals(19)}),
    dict(id="calc_word",    prompt="If I have 12 boxes with 15 items each, how many items are there in total?",
         expect=["calculator"], arg_check={"calculator": calc_equals(180)}),
    dict(id="calc_big",     prompt="What is 987 minus 654 plus 321?",
         expect=["calculator"], arg_check={"calculator": calc_equals(654)}),
    # ---- time ----
    dict(id="time_plain",   prompt="What's the current time?", expect=["get_current_time"]),
    dict(id="time_phrase",  prompt="Do you happen to know today's date and time right now?",
         expect=["get_current_time"]),
    # ---- list_directory ----
    dict(id="list_models",  prompt="What files are in the models directory?",
         expect=["list_directory"], arg_check={"list_directory": path_contains("models")}),
    dict(id="list_scripts", prompt="List the files in the scripts folder.",
         expect=["list_directory"], arg_check={"list_directory": path_contains("scripts")}),
    # ---- read_file ----
    dict(id="read_vcpkg",   prompt="Show me the contents of vcpkg.json.",
         expect=["read_file"], arg_check={"read_file": path_contains("vcpkg.json")}),
    dict(id="read_readme",  prompt="Read the first 100 bytes of README.md for me.",
         expect=["read_file"], arg_check={"read_file": path_contains("readme")}),
    # ---- multi-tool ----
    dict(id="combo_time_calc", prompt="Tell me the current time, and also what 8 * 9 is.",
         expect=["get_current_time", "calculator"], arg_check={"calculator": calc_equals(72)}),
    # ---- no tool should be used ----
    dict(id="none_greeting", prompt="Say hello in Spanish.", expect=[]),
    dict(id="none_capital",  prompt="What is the capital of France?", expect=[]),
    dict(id="none_explain",  prompt="In one sentence, what is a large language model?", expect=[]),
]


def run_model(exe, model_path, threads, seed, prompts, ctx, iters, gpu_layers, device):
    """Run all prompts through one toolchat --bench process; return list of result dicts."""
    exe = str(Path(exe).resolve())            # CreateProcess needs a real, absolute path
    model_path = str(Path(model_path).resolve())
    cmd = [exe, model_path, "-t", str(threads), "-c", str(ctx), "-i", str(iters),
           "--seed", str(seed), "--bench"]
    if gpu_layers:
        cmd += ["-g", str(gpu_layers), "-d", str(device)]
    stdin_data = "".join(p + "\n" for p in prompts)
    proc = subprocess.run(cmd, input=stdin_data.encode("utf-8"),
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    results = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return results


def score(case, result):
    """Return (selection_ok, args_ok, overall_ok)."""
    calls = result.get("tool_calls", [])
    called = [c["name"] for c in calls]
    expect = case["expect"]
    # selection: exact set match (penalizes both missing and spurious calls)
    selection_ok = sorted(set(called)) == sorted(set(expect))
    # arguments: every expected tool that was called has acceptable args
    args_ok = True
    checks = case.get("arg_check", {})
    for name, fn in checks.items():
        matching = [c["arguments"] for c in calls if c["name"] == name]
        if not matching:
            args_ok = False
        elif not any(fn(a) for a in matching):
            args_ok = False
    overall_ok = selection_ok and args_ok
    return selection_ok, args_ok, overall_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="build/clang-x64-release/toolchat.exe")
    ap.add_argument("--model", action="append", required=True,
                    help="NAME=path/to/model.gguf (repeatable)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--gpu-layers", type=int, default=0, help="GPU layers to offload (0 = CPU)")
    ap.add_argument("--device", type=int, default=0, help="GPU device index")
    ap.add_argument("--out", default=None,
                    help="Results JSON path (default: next to the exe)")
    args = ap.parse_args()

    # Default the results file to the exe's own folder (e.g. the build preset
    # dir), so it lands next to toolchat.exe rather than in the CWD. An explicit
    # --out (absolute or relative) is honored as given.
    out_path = Path(args.out) if args.out else Path(args.exe).resolve().parent / "benchmark_results.json"

    models = []
    for m in args.model:
        name, _, path = m.partition("=")
        models.append((name, path))

    prompts = [c["prompt"] for c in SUITE]
    all_results = {}

    for name, path in models:
        if not Path(path).exists():
            print(f"!! model not found: {path} (skipping {name})", file=sys.stderr)
            continue
        print(f"\n=== {name} ({path}) ===", file=sys.stderr)
        # per-case tallies across seeds
        tally = {c["id"]: dict(sel=0, arg=0, overall=0, n=0, ms=[]) for c in SUITE}
        for seed in args.seeds:
            print(f"  seed {seed} ...", file=sys.stderr, flush=True)
            results = run_model(args.exe, path, args.threads, seed, prompts, args.ctx, args.iters,
                                args.gpu_layers, args.device)
            if len(results) != len(SUITE):
                print(f"    warning: got {len(results)} results, expected {len(SUITE)}",
                      file=sys.stderr)
            for case, result in zip(SUITE, results):
                sel, arg, ov = score(case, result)
                t = tally[case["id"]]
                t["sel"] += sel; t["arg"] += arg; t["overall"] += ov; t["n"] += 1
                t["ms"].append(result.get("ms", 0))
        all_results[name] = tally

    # ---- report ----
    def pct(x, n): return f"{100.0*x/n:5.1f}%" if n else "   n/a"

    print("\n" + "=" * 78)
    print("TOOL-CALLING ACCURACY  (overall = correct tool set AND correct args)")
    print("=" * 78)
    names = list(all_results.keys())
    for name in names:
        tally = all_results[name]
        n = sum(t["n"] for t in tally.values())
        sel = sum(t["sel"] for t in tally.values())
        arg = sum(t["arg"] for t in tally.values())
        ov = sum(t["overall"] for t in tally.values())
        ms = [x for t in tally.values() for x in t["ms"]]
        tool_ids = [c["id"] for c in SUITE if c["expect"]]
        none_ids = [c["id"] for c in SUITE if not c["expect"]]
        tool_ov = sum(tally[i]["overall"] for i in tool_ids)
        tool_n = sum(tally[i]["n"] for i in tool_ids)
        none_ov = sum(tally[i]["overall"] for i in none_ids)
        none_n = sum(tally[i]["n"] for i in none_ids)
        print(f"\n{name}   ({n} runs = {len(SUITE)} cases x {len(args.seeds)} seeds)")
        print(f"  overall accuracy       : {pct(ov, n)}  ({ov}/{n})")
        print(f"  tool selection         : {pct(sel, n)}")
        print(f"  argument correctness   : {pct(arg, n)}")
        print(f"  tool-required cases    : {pct(tool_ov, tool_n)}  ({tool_ov}/{tool_n})")
        print(f"  no-tool cases (abstain): {pct(none_ov, none_n)}  ({none_ov}/{none_n})")
        print(f"  avg latency / query    : {sum(ms)/len(ms)/1000:.1f} s" if ms else "")

    # per-case breakdown
    print("\n" + "-" * 78)
    print(f"{'case':<16} " + " ".join(f"{n[:14]:>14}" for n in names))
    print("-" * 78)
    for c in SUITE:
        row = f"{c['id']:<16} "
        for name in names:
            t = all_results[name][c["id"]]
            row += f"{t['overall']}/{t['n']:<12} "
        print(row)

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw tallies written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
