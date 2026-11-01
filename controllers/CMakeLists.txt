# ASTRUM C++ controllers — one CMake project for every C++ component:
#
#   common/cpp     astrum::core (wire codec, CLI options, control blocks), astrum::client
#                  (Zenoh session + latest-value telemetry + leased commands), the
#                  cross-language wire-fixture tool, and their tests
#   cpp            astrum-controller: the hold / autopilot example
#   optimal/cpp    astrum-optimal-controller: LQR / MPC laws on the same client
#
# Build + test (inside the dev image):
#   cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/controllers -j
#   ctest --test-dir build/controllers --output-on-failure
cmake_minimum_required(VERSION 3.22)
# C is enabled only because CapnProtoConfig.cmake probes libatomic with a C test program.
project(astrum_controllers C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CapnProto CONFIG REQUIRED)
find_package(zenohc CONFIG REQUIRED)
find_package(zenohcxx CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(Threads REQUIRED)

enable_testing()
include(GoogleTest)

set(ASTRUM_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Warnings as errors for hand-written sources (generated capnp code is compiled
# in its own target without this).
function(astrum_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
endfunction()

# A gtest executable registered with ctest (one ctest entry per TEST).
function(astrum_test name)
  add_executable(${name} ${ARGN})
  astrum_warnings(${name})
  target_link_libraries(${name} PRIVATE GTest::gtest_main)
  gtest_discover_tests(${name})
endfunction()

add_subdirectory(common/cpp)
add_subdirectory(cpp)
add_subdirectory(optimal/cpp)
