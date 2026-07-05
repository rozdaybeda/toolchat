#include "download.h"

#include <filesystem>
#include <fstream>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#endif

namespace toolchat {

namespace fs = std::filesystem;

#ifdef _WIN32

static std::wstring to_wide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), n);
    return w;
}

// RAII for WinHTTP handles.
struct WinHttpHandle {
    HINTERNET h = nullptr;
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET x) : h(x) {}
    ~WinHttpHandle() { if (h) WinHttpCloseHandle(h); }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    operator HINTERNET() const { return h; }
};

bool download_file(const std::string& url, const std::string& dest_path,
                   std::string& error, const DownloadProgress& progress) {
    std::wstring wurl = to_wide(url);

    // Split the URL into host / path.
    URL_COMPONENTS uc = {};
    uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host;   uc.dwHostNameLength = 255;
    uc.lpszUrlPath = path;    uc.dwUrlPathLength = 2047;
    if (!WinHttpCrackUrl(wurl.c_str(), (DWORD)wurl.size(), 0, &uc)) {
        error = "Bad URL: " + url;
        return false;
    }
    const bool secure = (uc.nScheme == INTERNET_SCHEME_HTTPS);

    WinHttpHandle session(WinHttpOpen(L"toolchat/1.0",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) { error = "WinHttpOpen failed"; return false; }

    WinHttpHandle connect(WinHttpConnect(session, host, uc.nPort, 0));
    if (!connect) { error = "WinHttpConnect failed"; return false; }

    WinHttpHandle request(WinHttpOpenRequest(connect, L"GET", path, nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
        secure ? WINHTTP_FLAG_SECURE : 0));
    if (!request) { error = "WinHttpOpenRequest failed"; return false; }

    // Follow redirects (HF resolve/main → CDN). Default disallows https→http,
    // which is what we want.
    if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        error = "WinHttpSendRequest failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }
    if (!WinHttpReceiveResponse(request, nullptr)) {
        error = "WinHttpReceiveResponse failed (" + std::to_string(GetLastError()) + ")";
        return false;
    }

    // HTTP status code.
    DWORD status = 0, len = sizeof(status);
    WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &len, WINHTTP_NO_HEADER_INDEX);
    if (status != 200) {
        error = "HTTP " + std::to_string(status) + " for " + url;
        return false;
    }

    // Content-Length (may be absent → total 0).
    uint64_t total = 0;
    wchar_t clen[64] = {};
    DWORD clen_len = sizeof(clen);
    if (WinHttpQueryHeaders(request, WINHTTP_QUERY_CONTENT_LENGTH,
                            WINHTTP_HEADER_NAME_BY_INDEX, clen, &clen_len,
                            WINHTTP_NO_HEADER_INDEX)) {
        total = _wcstoui64(clen, nullptr, 10);
    }

    // Write to <dest>.part, then rename on success.
    std::error_code ec;
    fs::path dest(dest_path);
    if (dest.has_parent_path()) fs::create_directories(dest.parent_path(), ec);
    fs::path part = dest;
    part += ".part";

    std::ofstream out(part, std::ios::binary | std::ios::trunc);
    if (!out) { error = "Cannot open for writing: " + part.string(); return false; }

    std::vector<char> buf(1 << 16);  // 64 KiB
    uint64_t downloaded = 0;
    if (progress) progress(0, total);
    for (;;) {
        DWORD got = 0;
        if (!WinHttpReadData(request, buf.data(), (DWORD)buf.size(), &got)) {
            error = "WinHttpReadData failed (" + std::to_string(GetLastError()) + ")";
            return false;
        }
        if (got == 0) break;  // done
        out.write(buf.data(), got);
        if (!out) { error = "Write failed (disk full?): " + part.string(); return false; }
        downloaded += got;
        if (progress) progress(downloaded, total);
    }
    out.close();

    fs::remove(dest, ec);                 // replace any stale file
    fs::rename(part, dest, ec);
    if (ec) { error = "Rename failed: " + ec.message(); return false; }
    return true;
}

#else  // non-Windows: no native downloader

bool download_file(const std::string&, const std::string&, std::string& error,
                   const DownloadProgress&) {
    error = "In-app download is only implemented on Windows (use scripts/download_model.py)";
    return false;
}

#endif

} // namespace toolchat
