add_library(zerofg STATIC
  src/zerofg.cc
)

target_include_directories(zerofg
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# While ZeroFG lives inside XenDroid, reuse Xenia's vendored Vulkan headers.
# When ZeroFG becomes a standalone repository, this will be replaced by a
# normal Vulkan-Headers dependency without changing the public API.
target_include_directories(zerofg
  PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/Vulkan-Headers/include
)

target_compile_features(zerofg PUBLIC cxx_std_20)
