cmake_minimum_required(VERSION 3.12)
project(glb_bai_hybrid CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# For macOS clang/gcc
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(glb_bai
  src/main.cpp
)

target_include_directories(glb_bai PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Reasonable warnings
if (MSVC)
  target_compile_options(glb_bai PRIVATE /W4)
else()
  target_compile_options(glb_bai PRIVATE -Wall -Wextra -Wpedantic)
endif()
