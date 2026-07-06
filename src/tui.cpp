#include "tui.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/component_base.hpp>
#include <ftxui/component/component_options.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

namespace toolchat {

using namespace ftxui;

// Fixed width of the right-hand details pane (long lines word-wrap inside it).
static constexpr int kDetailPaneWidth = 34;

// The highlighted entry's detail blurb as a word-wrapping column, one paragraph
// per '\n'-separated line (empty detail → empty pane).
static Element detail_pane(const std::vector<std::string>& details, int sel) {
    Elements rows;
    if (sel >= 0 && sel < (int)details.size()) {
        const std::string& d = details[sel];
        for (size_t pos = 0; pos < d.size();) {
            size_t nl = d.find('\n', pos);
            if (nl == std::string::npos) nl = d.size();
            if (nl > pos) rows.push_back(paragraph(d.substr(pos, nl - pos)) | dim);
            pos = nl + 1;
        }
    }
    return vbox(std::move(rows)) | size(WIDTH, EQUAL, kDetailPaneWidth);
}

bool run_setup_tui(const std::vector<std::string>& model_labels,
                   const std::vector<std::string>& model_details,
                   const std::vector<std::string>& device_labels,
                   int& model_idx, int& device_idx) {
    if (model_labels.empty() || device_labels.empty()) return false;

    // Local copies kept alive for the whole screen loop (ftxui holds references).
    std::vector<std::string> models  = model_labels;
    std::vector<std::string> details = model_details;
    std::vector<std::string> devices = device_labels;

    auto screen = ScreenInteractive::TerminalOutput();

    int  model_sel = (model_idx  >= 0 && model_idx  < (int)models.size())  ? model_idx  : 0;
    int  dev_sel   = (device_idx >= 0 && device_idx < (int)devices.size()) ? device_idx : 0;
    bool started   = false;

    auto model_dd  = Dropdown(&models, &model_sel);
    auto device_rb = Radiobox(&devices, &dev_sel);
    auto start_btn = Button("Start chat", [&] { started = true; screen.Exit(); },
                            ButtonOption::Ascii());

    auto layout = Container::Vertical({ model_dd, device_rb, start_btn });

    auto renderer = Renderer(layout, [&] {
        Element pickers = vbox({
            hbox(text("Model:  "), model_dd->Render()),
            separatorEmpty(),
            text("Device:"),
            hbox(text("  "), device_rb->Render()),
        }) | flex;
        // Details of the highlighted model in a right-hand pane, behind a
        // vertical separator (skipped entirely when there is nothing to show).
        bool has_details = model_sel >= 0 && model_sel < (int)details.size()
                        && !details[model_sel].empty();
        Element body = has_details
            ? hbox({ pickers, separator(), detail_pane(details, model_sel) })
            : pickers;
        return window(text(" toolchat "),
            vbox({
                body,
                separatorEmpty(),
                hbox(filler(), start_btn->Render(), filler()),
                separator(),
                text("Up/Down move  -  Enter select  -  Esc cancel") | dim,
            }) | size(WIDTH, GREATER_THAN, has_details ? 72 : 44)
        );
    });

    auto with_keys = CatchEvent(renderer, [&](Event e) {
        if (e == Event::Escape || e == Event::Character('q')) {
            started = false;
            screen.Exit();
            return true;
        }
        return false;
    });

    screen.Loop(with_keys);

    if (!started) return false;
    model_idx  = model_sel;
    device_idx = dev_sel;
    return true;
}

int tui_menu(const std::string& title, const std::vector<std::string>& items,
             int selected, const std::vector<std::string>& details) {
    if (items.empty()) return -1;

    std::vector<std::string> entries = items;
    std::vector<std::string> blurbs  = details;
    auto screen = ScreenInteractive::TerminalOutput();

    int  sel = (selected >= 0 && selected < (int)entries.size()) ? selected : 0;
    bool ok  = false;

    auto menu = Menu(&entries, &sel);

    auto renderer = Renderer(menu, [&] {
        bool has_details = sel >= 0 && sel < (int)blurbs.size() && !blurbs[sel].empty();
        Element body = has_details
            ? hbox({ menu->Render() | flex, separator(), detail_pane(blurbs, sel) })
            : menu->Render();
        return window(text(" " + title + " "),
            vbox({
                body,
                separator(),
                text("Up/Down move  -  Enter select  -  Esc cancel") | dim,
            }) | size(WIDTH, GREATER_THAN, blurbs.empty() ? 44 : 72)
        );
    });

    auto with_keys = CatchEvent(renderer, [&](Event e) {
        if (e == Event::Return) { ok = true;  screen.Exit(); return true; }
        if (e == Event::Escape || e == Event::Character('q')) {
            ok = false; screen.Exit(); return true;
        }
        return false;
    });

    screen.Loop(with_keys);
    return ok ? sel : -1;
}

} // namespace toolchat
