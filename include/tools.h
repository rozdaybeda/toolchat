#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace toolchat {

// A single callable tool exposed to the model.
//   - `parameters` is a JSON-Schema object describing the arguments.
//   - `handler` receives the parsed arguments and returns a JSON result that is
//     fed back to the model inside a <tool_response> block.
struct Tool {
    std::string name;
    std::string description;
    nlohmann::json parameters;                                   // JSON Schema (object)
    std::function<nlohmann::json(const nlohmann::json& args)> handler;
};

// Holds the set of tools available to an Agent and knows how to (a) render them
// for the system prompt and (b) dispatch a call coming back from the model.
class ToolRegistry {
public:
    void add(Tool tool);
    const Tool* find(const std::string& name) const;
    const std::vector<Tool>& tools() const { return tools_; }
    bool empty() const { return tools_.empty(); }

    // Array of {"type":"function","function":{name,description,parameters}} — the
    // exact shape Qwen expects inside the <tools></tools> block.
    nlohmann::json schemas_json() const;

    // Runs the named tool. Never throws: unknown tool names and handler
    // exceptions are turned into {"error": "..."} so the result can be handed
    // back to the model to recover from.
    nlohmann::json dispatch(const std::string& name, const nlohmann::json& args) const;

private:
    std::vector<Tool> tools_;
};

// Registers the demo tools: get_current_time, calculator, list_directory, read_file.
void register_builtin_tools(ToolRegistry& registry);

} // namespace toolchat
