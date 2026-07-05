#include "tools.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace toolchat {

using nlohmann::json;

void ToolRegistry::add(Tool tool) {
    tools_.push_back(std::move(tool));
}

const Tool* ToolRegistry::find(const std::string& name) const {
    for (const auto& t : tools_)
        if (t.name == name) return &t;
    return nullptr;
}

json ToolRegistry::schemas_json() const {
    json arr = json::array();
    for (const auto& t : tools_) {
        arr.push_back({
            {"type", "function"},
            {"function", {
                {"name", t.name},
                {"description", t.description},
                {"parameters", t.parameters},
            }},
        });
    }
    return arr;
}

json ToolRegistry::dispatch(const std::string& name, const json& args) const {
    const Tool* tool = find(name);
    if (!tool) {
        return json{{"error", "unknown tool: " + name}};
    }
    try {
        return tool->handler(args.is_null() ? json::object() : args);
    } catch (const std::exception& e) {
        return json{{"error", std::string("tool failed: ") + e.what()}};
    }
}

// ---------------------------------------------------------------------------
// Built-in tool implementations
// ---------------------------------------------------------------------------

// get_current_time -----------------------------------------------------------
static json tool_get_current_time(const json&) {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return json{{"current_time", buf}};
}

// calculator -----------------------------------------------------------------
// Minimal recursive-descent evaluator for + - * / ( ) and decimal numbers.
// Self-contained (no eval library) so the arithmetic is verifiably correct.
namespace {
struct ExprParser {
    const std::string& s;
    size_t i = 0;

    explicit ExprParser(const std::string& str) : s(str) {}

    void skip_ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }

    double parse() {
        double v = expr();
        skip_ws();
        if (i != s.size())
            throw std::runtime_error("unexpected character in expression at position " + std::to_string(i));
        return v;
    }

    // expr := term (('+' | '-') term)*
    double expr() {
        double v = term();
        for (;;) {
            skip_ws();
            if (i < s.size() && s[i] == '+') { ++i; v += term(); }
            else if (i < s.size() && s[i] == '-') { ++i; v -= term(); }
            else break;
        }
        return v;
    }

    // term := factor (('*' | '/') factor)*
    double term() {
        double v = factor();
        for (;;) {
            skip_ws();
            if (i < s.size() && s[i] == '*') { ++i; v *= factor(); }
            else if (i < s.size() && s[i] == '/') {
                ++i;
                double d = factor();
                if (d == 0.0) throw std::runtime_error("division by zero");
                v /= d;
            } else break;
        }
        return v;
    }

    // factor := ('+' | '-') factor | '(' expr ')' | number
    double factor() {
        skip_ws();
        if (i >= s.size()) throw std::runtime_error("unexpected end of expression");
        if (s[i] == '+') { ++i; return factor(); }
        if (s[i] == '-') { ++i; return -factor(); }
        if (s[i] == '(') {
            ++i;
            double v = expr();
            skip_ws();
            if (i >= s.size() || s[i] != ')') throw std::runtime_error("missing closing parenthesis");
            ++i;
            return v;
        }
        return number();
    }

    double number() {
        skip_ws();
        size_t start = i;
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) || s[i] == '.')) ++i;
        if (i == start) throw std::runtime_error("expected a number at position " + std::to_string(i));
        return std::stod(s.substr(start, i - start));
    }
};
} // namespace

static json tool_calculator(const json& args) {
    if (!args.contains("expression") || !args["expression"].is_string())
        return json{{"error", "expected string argument 'expression'"}};
    const std::string expr = args["expression"].get<std::string>();
    ExprParser parser(expr);
    double result = parser.parse();
    // Return an integer when the value is whole, otherwise a double.
    if (std::floor(result) == result && std::abs(result) < 1e15)
        return json{{"expression", expr}, {"result", (long long)result}};
    return json{{"expression", expr}, {"result", result}};
}

// list_directory -------------------------------------------------------------
static json tool_list_directory(const json& args) {
    namespace fs = std::filesystem;
    std::string path = args.value("path", ".");
    std::error_code ec;
    if (!fs::exists(path, ec))
        return json{{"error", "path does not exist: " + path}};
    if (!fs::is_directory(path, ec))
        return json{{"error", "not a directory: " + path}};

    json entries = json::array();
    for (const auto& e : fs::directory_iterator(path, ec)) {
        json item;
        item["name"] = e.path().filename().string();
        item["type"] = e.is_directory(ec) ? "dir" : "file";
        if (e.is_regular_file(ec)) {
            std::error_code sz_ec;
            auto sz = fs::file_size(e.path(), sz_ec);
            if (!sz_ec) item["size_bytes"] = (long long)sz;
        }
        entries.push_back(item);
    }
    return json{{"path", path}, {"entries", entries}};
}

// read_file ------------------------------------------------------------------
static json tool_read_file(const json& args) {
    namespace fs = std::filesystem;
    if (!args.contains("path") || !args["path"].is_string())
        return json{{"error", "expected string argument 'path'"}};
    std::string path = args["path"].get<std::string>();
    long long max_bytes = args.value("max_bytes", (long long)4096);
    if (max_bytes <= 0) max_bytes = 4096;

    std::error_code ec;
    if (!fs::exists(path, ec)) return json{{"error", "file does not exist: " + path}};
    if (fs::is_directory(path, ec)) return json{{"error", "path is a directory: " + path}};

    std::ifstream f(path, std::ios::binary);
    if (!f) return json{{"error", "could not open file: " + path}};

    std::string content(static_cast<size_t>(max_bytes), '\0');
    f.read(content.data(), max_bytes);
    content.resize(static_cast<size_t>(f.gcount()));

    bool truncated = !f.eof();
    return json{{"path", path}, {"content", content}, {"truncated", truncated}};
}

// ---------------------------------------------------------------------------

void register_builtin_tools(ToolRegistry& registry) {
    registry.add(Tool{
        "get_current_time",
        "Get the current local date and time.",
        json{{"type", "object"}, {"properties", json::object()}},
        tool_get_current_time,
    });

    registry.add(Tool{
        "calculator",
        "Evaluate an arithmetic expression with + - * / and parentheses. "
        "Example expression: \"23 * 19 + 7\".",
        json{
            {"type", "object"},
            {"properties", {
                {"expression", {
                    {"type", "string"},
                    {"description", "The arithmetic expression to evaluate."},
                }},
            }},
            {"required", json::array({"expression"})},
        },
        tool_calculator,
    });

    registry.add(Tool{
        "list_directory",
        "List the files and subdirectories in a directory on the local machine.",
        json{
            {"type", "object"},
            {"properties", {
                {"path", {
                    {"type", "string"},
                    {"description", "Directory path to list. Defaults to the current directory."},
                }},
            }},
        },
        tool_list_directory,
    });

    registry.add(Tool{
        "read_file",
        "Read the contents of a text file on the local machine (truncated to max_bytes).",
        json{
            {"type", "object"},
            {"properties", {
                {"path", {
                    {"type", "string"},
                    {"description", "Path to the file to read."},
                }},
                {"max_bytes", {
                    {"type", "integer"},
                    {"description", "Maximum number of bytes to read (default 4096)."},
                }},
            }},
            {"required", json::array({"path"})},
        },
        tool_read_file,
    });
}

} // namespace toolchat
