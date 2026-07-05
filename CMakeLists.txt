cmake_minimum_required(VERSION 3.21)
project(toolchat VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

find_package(llama CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(ftxui CONFIG REQUIRED)

# ftxui is built by vcpkg with MSVC, so its exported targets bake in the MSVC-only
# flag "/utf-8" as an interface compile option. That flag propagates to every
# source of any target linking ftxui and is rejected by our clang++ (GNU) driver
# ("no such file or directory: '/utf-8'"). clang treats source as UTF-8 by
# default, so strip the flag from the imported ftxui targets.
foreach(_ftxui_t ftxui::screen ftxui::dom ftxui::component)
    if(TARGET ${_ftxui_t})
        get_target_property(_ftxui_opts ${_ftxui_t} INTERFACE_COMPILE_OPTIONS)
        if(_ftxui_opts)
            list(REMOVE_ITEM _ftxui_opts "/utf-8")
            set_property(TARGET ${_ftxui_t} PROPERTY INTERFACE_COMPILE_OPTIONS "${_ftxui_opts}")
        endif()
    endif()
endforeach()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/agent.cpp
    src/tools.cpp
    src/tui.cpp
    src/catalog.cpp
    src/download.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    llama
    nlohmann_json::nlohmann_json
    ftxui::component
    ftxui::dom
    ftxui::screen
)

# WinHTTP for the in-app model downloader (src/download.cpp).
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE winhttp)
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/download_model.py
    ${CMAKE_BINARY_DIR}/download_model.py
    COPYONLY
)

# Copy the model catalog next to the exe so the in-app picker finds it even when
# the working directory isn't the project root.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/models.json
    ${CMAKE_BINARY_DIR}/models.json
    COPYONLY
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
