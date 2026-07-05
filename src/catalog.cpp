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
                e.local_path = (fs::path("models") / e.dir / e.file).generic_string();
                e.downloaded = fs::exists(e.local_path, ec);
                out.push_back(std::move(e));
            }
        } catch (const std::exception&) {
            // Malformed manifest — fall through to on-disk scan only.
        }
    }

    // Append any on-disk .gguf under models/ that the manifest didn't list, so
    // manually-added models still appear in the picker.
    fs::path root = "models";
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
