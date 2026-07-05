# Bumping the vcpkg `llama-cpp` port to b9871 (Qwen3.5 support)

The stock vcpkg `llama-cpp` port (build **b7146**) does not know the `qwen35`
architecture and fails to load Qwen3.5 GGUFs:

```
error loading model architecture: unknown model architecture: 'qwen35'
```

Qwen3.5 support landed in llama.cpp after b7146. These are the exact changes made
to `$VCPKG_ROOT/ports/llama-cpp` to bump it to **b9871** and build with llama's
own bundled ggml (so llama + ggml come from the same commit — important because
the stale system `ggml` vcpkg port, 2025-11-17, predates Qwen3.5's Gated-DeltaNet
ops). Re-apply these if the vcpkg registry is reset/updated.

> Pick the current latest build instead of b9871 if you like:
> `curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`
> then get its tarball SHA512 with
> `curl -sL https://github.com/ggml-org/llama.cpp/archive/b<N>.tar.gz | sha512sum`.

## `ports/llama-cpp/vcpkg.json`

- `"version"`: `"7146"` → `"9871"`
- Remove the `"ggml"` dependency (we build bundled ggml instead), keeping the
  `vcpkg-cmake` / `vcpkg-cmake-config` host deps.
- Add a `vulkan` feature (no deps — uses the system Vulkan SDK) so the bundled
  ggml can be built with the GPU backend:
  ```json
  "vulkan": {
    "description": "Vulkan GPU backend in the bundled ggml (uses the system Vulkan SDK)"
  }
  ```
  The project's own `vcpkg.json` `vulkan` feature then depends on
  `llama-cpp[vulkan]` (instead of the old separate `ggml[vulkan]`).

## `ports/llama-cpp/portfile.cmake`

Replace the top of the file (through `vcpkg_fixup_pkgconfig()`) with:

```cmake
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO ggml-org/llama.cpp
    REF b${VERSION}
    SHA512 bb5c757aa2537f2745f49b5d4833c63f9b66cc2a986d78a70c92f584a4e28fdec7383d2aa30646d5c7d3eba98f58b8fd17e84eb5778bb22041c4b61e48ce5380
    HEAD_REF master
)
# (patches dropped: they targeted the system-ggml layout and don't apply to b9871)

vcpkg_check_features(OUT_FEATURE_OPTIONS options
    FEATURES
        download    LLAMA_CURL
        tools       LLAMA_BUILD_TOOLS
        vulkan      GGML_VULKAN   # bundled ggml with the Vulkan GPU backend
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${options}
        -DGGML_CCACHE=OFF
        -DLLAMA_ALL_WARNINGS=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_SERVER=OFF
        -DLLAMA_BUILD_APP=OFF        # new unified binary needs a generated build-info.h
        -DLLAMA_USE_SYSTEM_GGML=OFF  # build llama's own bundled ggml
        -DGGML_OPENMP=OFF            # else ggml-config forces consumers to FindOpenMP (fails for clang)
        -DVCPKG_LOCK_FIND_PACKAGE_Git=OFF
)

vcpkg_cmake_install()
# llama and ggml share the lib/cmake parent; DO_NOT_DELETE_PARENT_CONFIG_PATH on
# the first call keeps ggml's config from being wiped before its own fixup runs.
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/llama" DO_NOT_DELETE_PARENT_CONFIG_PATH)
vcpkg_cmake_config_fixup(PACKAGE_NAME ggml CONFIG_PATH "lib/cmake/ggml")
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()
```

Also guard the `convert_hf_to_gguf.py` rename so a good build isn't wasted if the
install path shifts between versions:

```cmake
if(EXISTS "${CURRENT_PACKAGES_DIR}/bin/convert_hf_to_gguf.py")
    file(RENAME "${CURRENT_PACKAGES_DIR}/bin/convert_hf_to_gguf.py" "${CURRENT_PACKAGES_DIR}/tools/${PORT}/convert-hf-to-gguf.py")
endif()
```

## Rebuild

Any reconfigure of this project triggers the vcpkg rebuild (the port's ABI hash
changed), which removes the old `ggml` port and rebuilds `llama-cpp@9871`:

```bash
cmake --preset clang-x64-release
cmake --build --preset clang-x64-release
```

## Gotchas hit while doing this (in order)

1. **`unknown model architecture: 'qwen35'`** — the reason for the bump.
2. **`Cannot open include file: 'build-info.h'`** — the new `app/` target;
   fixed with `-DLLAMA_BUILD_APP=OFF`.
3. **`debug/lib/cmake/ggml does not exist`** — the first `config_fixup` deletes
   the shared `lib/cmake` parent; fixed with `DO_NOT_DELETE_PARENT_CONFIG_PATH`.
4. **`Could NOT find OpenMP_CXX`** at *consumer* configure — bundled ggml built
   with OpenMP makes `ggml-config.cmake` require it; fixed with `-DGGML_OPENMP=OFF`.
5. **Segfault (0xC0000005) during inference** — Qwen3.5-4B's fused Gated Delta
   Net path crashes with Flash Attention auto-enabled on CPU; fixed in
   `src/agent.cpp` with `flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`.
