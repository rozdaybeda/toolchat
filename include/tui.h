#pragma once

#include <string>
#include <vector>

namespace toolchat {

// Combined initial setup form: a model dropdown, a device radio list, and a
// "Start chat" button. model_idx / device_idx are in/out — the passed-in values
// are pre-selected, and the user's choice is written back. Returns false if the
// user cancelled (Esc / q / Ctrl-C).
bool run_setup_tui(const std::vector<std::string>& model_labels,
                   const std::vector<std::string>& device_labels,
                   int& model_idx, int& device_idx);

// Single-list picker, reused by the in-chat /model and /device commands. Returns
// the chosen index, or -1 if cancelled.
int tui_menu(const std::string& title, const std::vector<std::string>& items,
             int selected = 0);

} // namespace toolchat
