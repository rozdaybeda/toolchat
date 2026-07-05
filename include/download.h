#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace toolchat {

// Progress callback: (bytes downloaded so far, total bytes). total is 0 when the
// server didn't send a Content-Length.
using DownloadProgress = std::function<void(uint64_t downloaded, uint64_t total)>;

// Download `url` to `dest_path` over HTTPS (WinHTTP), creating parent directories
// and following redirects. Writes to `<dest>.part` and renames on success so a
// partial download never looks complete. Returns false and sets `error` on
// failure. `progress` (optional) is called periodically as bytes arrive.
bool download_file(const std::string& url, const std::string& dest_path,
                   std::string& error, const DownloadProgress& progress = {});

} // namespace toolchat
