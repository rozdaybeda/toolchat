#!/usr/bin/env python3
"""Download GGUF models for toolchat (tool-calling demo).

The model catalog lives in scripts/models.json — the single source of truth
shared with the C++ app (catalog.cpp), so this list and the in-app TUI stay in
sync. The agent auto-detects the chat template (Qwen ChatML / Gemma) from the
GGUF architecture; no per-model config is needed here.
"""
import json, subprocess, sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
CATALOG = Path(__file__).parent / "models.json"


def load_models():
    """Build the MODELS dict from the shared models.json manifest."""
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    out = {}
    for m in data["models"]:
        out[m["key"]] = {
            "desc": f"{m['name']} Q4_K_M — ~{m['size']}, {m['desc']}",
            "repo": m["repo"],
            "files": [m["file"]],
            "local_dir": m["dir"],
        }
    return out


MODELS = load_models()


def ensure_hf():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        import huggingface_hub
        return huggingface_hub


def resolve_filename(hf, repo, want):
    """Return a filename that actually exists in the repo: `want` if present, else
    the first *.gguf sharing its quant tag (e.g. Q4_K_M). Model file names vary
    (esp. Gemma's E/QAT variants), so don't hard-fail on a guessed leaf name."""
    try:
        files = hf.HfApi().list_repo_files(repo)
    except Exception:
        return want
    if want in files:
        return want
    tag = next((t for t in ("Q4_K_M", "Q4_K", "Q4_0") if t in want), "Q4_K_M")
    cands = [f for f in files if f.endswith(".gguf") and tag in f and "/" not in f]
    if not cands:  # last resort: any single-file gguf at the repo root
        cands = [f for f in files if f.endswith(".gguf") and "/" not in f]
    if cands and cands[0] != want:
        print(f"  (note: '{want}' not in repo; using '{cands[0]}')")
    return cands[0] if cands else want


def download(key):
    hf = ensure_hf()
    m = MODELS[key]
    local_dir = MODELS_DIR / m["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    resolved = []
    for filename in m["files"]:
        filename = resolve_filename(hf, m["repo"], filename)
        resolved.append(filename)
        dest = local_dir / filename
        if dest.exists():
            print(f"  Already downloaded: {dest}")
        else:
            print(f"  Downloading {m['repo']}/{filename} ...")
            hf.hf_hub_download(repo_id=m["repo"], filename=filename, local_dir=str(local_dir))
            print(f"  Done: {dest}")
    return local_dir / resolved[0]


def main():
    print("Available models:")
    keys = list(MODELS.keys())
    for i, key in enumerate(keys):
        print(f"  [{i+1}] {key:14}  {MODELS[key]['desc']}")
    print("  [0] All")

    choice = input(f"\nSelect model(s) to download [0-{len(keys)}]: ").strip()
    if choice == "0":
        to_download = keys
    elif choice.isdigit() and 1 <= int(choice) <= len(keys):
        to_download = [keys[int(choice) - 1]]
    else:
        to_download = []

    if not to_download:
        print("No selection, exiting.")
        return

    for key in to_download:
        print(f"\n{MODELS[key]['desc']}")
        first_file = download(key)
        print(f"  Run (CPU): toolchat.exe {first_file}")
        print(f"  Run (GPU): toolchat.exe {first_file} -g 99 -d 0")


if __name__ == "__main__":
    main()
