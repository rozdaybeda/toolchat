#pragma once

#include <string>
#include <vector>

namespace toolchat {

// One entry in the model catalog. Catalog entries come from scripts/models.json
// (downloadable); on-disk .gguf files not in the manifest are appended as extras
// (repo empty → already present, can't be re-fetched).
struct ModelEntry {
    std::string key;         // catalog key (empty for on-disk-only extras)
    std::string name;        // display name, e.g. "Qwen3-4B"
    std::string repo;        // Hugging Face repo (empty if not downloadable)
    std::string file;        // gguf leaf filename
    std::string dir;         // local subdir under models/
    std::string size;        // approximate size string, e.g. "2.5 GB"
    std::string desc;        // short description
    std::string local_path;  // models/<dir>/<file> (relative to CWD)
    bool        downloaded = false;  // whether local_path exists on disk
};

// Load the model catalog: parse models.json (searched relative to the CWD and the
// exe directory), mark each entry's downloaded status, and append any on-disk
// .gguf under models/ that isn't already listed. Returns an empty list only if no
// manifest is found and no local models exist.
std::vector<ModelEntry> load_catalog();

// The Hugging Face direct-download URL for a repo file.
std::string hf_resolve_url(const std::string& repo, const std::string& file);

} // namespace toolchat
