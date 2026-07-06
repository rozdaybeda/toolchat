#include "catalog.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

namespace toolchat {

namespace fs = std::filesystem;
using nlohmann::json;

// Directory containing the running executable (so the manifest can be found even
// when the CWD isn't the project root).
static fs::path exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {};
    DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) return fs::path(buf).parent_path();
#endif
    return {};
}

// The models directory, resolved once so every path (catalog scan, download
// target, model load) agrees no matter what the current working directory is or
// where the exe lives. Anchors to the project root — the directory holding
// scripts/models.json (or an existing models/) — checking the CWD first, then
// walking up from the exe so a binary deep in build/ still uses toolchat/models
// rather than build/<preset>/models.
static fs::path models_dir() {
    std::error_code ec;
    std::vector<fs::path> roots;
    fs::path cwd = fs::current_path(ec);
    if (!ec) roots.push_back(cwd);
    for (fs::path d = exe_dir(); !d.empty(); d = d.parent_path()) {
        roots.push_back(d);
        if (d == d.parent_path()) break;
    }
    // Anchor to the project root, identified by scripts/models.json — a repo-only
    // marker that never exists under build/<preset>/. Checked across every root
    // FIRST so a stray build/<preset>/models directory can't win over the real
    // toolchat/models (which is the whole point of resolving this centrally).
    for (const auto& r : roots)
        if (fs::exists(r / "scripts" / "models.json", ec)) return r / "models";
    // Alternate layouts: a models.json beside the exe, then any existing models/.
    for (const auto& r : roots)
        if (fs::exists(r / "models.json", ec)) return r / "models";
    for (const auto& r : roots)
        if (fs::exists(r / "models", ec)) return r / "models";
    // Standalone (no repo checkout, first run): models next to the exe, else CWD.
    fs::path ed = exe_dir();
    return ed.empty() ? fs::path("models") : ed / "models";
}

// Locate models.json: prefer the project-root layout (CWD), then next to the exe.
static fs::path find_manifest() {
    std::vector<fs::path> candidates = {
        "scripts/models.json",
        "models.json",
        exe_dir() / "models.json",
        exe_dir() / "scripts" / "models.json",
    };
    std::error_code ec;
    for (const auto& p : candidates)
        if (!p.empty() && fs::exists(p, ec)) return p;
    return {};
}

std::string hf_resolve_url(const std::string& repo, const std::string& file) {
    return "https://huggingface.co/" + repo + "/resolve/main/" + file + "?download=true";
}

std::vector<ModelEntry> load_catalog() {
    std::vector<ModelEntry> out;
    std::error_code ec;

    const fs::path mdir = models_dir();
    fs::path manifest = find_manifest();
    if (!manifest.empty()) {
        try {
            std::ifstream f(manifest);
            json data = json::parse(f);
            for (const auto& m : data.value("models", json::array())) {
                ModelEntry e;
                e.key  = m.value("key", "");
                e.name = m.value("name", e.key);
                e.repo = m.value("repo", "");
                e.file = m.value("file", "");
                e.dir  = m.value("dir", "");
                e.size = m.value("size", "");
                e.desc = m.value("desc", "");
                e.req  = m.value("req", "");
                if (auto p = m.find("params"); p != m.end() && p->is_object()) {
                    e.params.temperature = p->value("temperature", -1.0f);
                    e.params.top_p       = p->value("top_p", -1.0f);
                    e.params.min_p       = p->value("min_p", -1.0f);
                    e.params.top_k       = p->value("top_k", -1);
                    e.params.n_ctx       = p->value("n_ctx", -1);
                    e.params.n_cpu_moe   = p->value("n_cpu_moe", -1);
                }
                e.local_path = (mdir / e.dir / e.file).generic_string();
                e.downloaded = fs::exists(e.local_path, ec);
                out.push_back(std::move(e));
            }
        } catch (const std::exception&) {
            // Malformed manifest — fall through to on-disk scan only.
        }
    }

    // Append any on-disk .gguf under models/ that the manifest didn't list, so
    // manually-added models still appear in the picker.
    fs::path root = mdir;
    if (fs::exists(root, ec)) {
        for (auto it = fs::recursive_directory_iterator(root, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (it->path().extension() != ".gguf") continue;
            std::string path = it->path().generic_string();
            bool known = std::any_of(out.begin(), out.end(), [&](const ModelEntry& e) {
                return e.local_path == path;
            });
            if (known) continue;
            ModelEntry e;
            e.name = it->path().filename().string();
            e.file = e.name;
            e.local_path = path;
            e.downloaded = true;
            out.push_back(std::move(e));
        }
    }
    return out;
}

} // namespace toolchat
