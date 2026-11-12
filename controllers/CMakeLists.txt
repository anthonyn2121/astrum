# ASTRUM C++ controllers — one CMake project for every C++ component:
#
#   common/cpp     the controller SDK (controllers/README.md):
#                    astrum::wire     plain types + Cap'n Proto codec (generated code private)
#                    astrum::control  SaturatedPi / OuterLoops / math helpers
#                    astrum::cli      Options + the shared-flag helpers
#                    astrum::client   Zenoh transport + the control loop
#                    astrum::runtime  Law / LawEntry / run_main
#                  plus the cross-language wire-fixture tool and the tests
#   cpp            astrum-controller: the hold / autopilot laws
#   optimal/cpp    astrum-optimal-controller: the LQR / MPC laws (ASTRUM_BUILD_OPTIMAL)
#   minsnap        astrum-minsnap-controller: the min-snap race controller (multirotor)
#
# Build + test (inside the dev image):
#   cmake -S controllers -B build/controllers -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/controllers -j
#   ctest --test-dir build/controllers --output-on-failure
# Install the SDK for out-of-tree controllers (find_package(astrum CONFIG REQUIRED),
# see examples/out_of_tree):
#   cmake --install build/controllers --prefix <prefix>
cmake_minimum_required(VERSION 3.22)
# C is enabled only because CapnProtoConfig.cmake probes libatomic with a C test program.
project(astrum_controllers VERSION 3.0.0 LANGUAGES C CXX)

option(ASTRUM_BUILD_TESTS "Build the unit tests and the wire-fixture tool (GTest, nlohmann_json)" ON)
option(ASTRUM_BUILD_OPTIMAL "Build astrum-optimal-controller (Eigen)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CapnProto CONFIG REQUIRED)
find_package(zenohc CONFIG REQUIRED)
find_package(zenohcxx CONFIG REQUIRED)
find_package(Threads REQUIRED)
if(ASTRUM_BUILD_OPTIMAL)
  find_package(Eigen3 CONFIG REQUIRED)
endif()
if(ASTRUM_BUILD_TESTS)
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(GTest CONFIG REQUIRED)
  enable_testing()
  include(GoogleTest)
endif()

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
add_subdirectory(minsnap)
if(ASTRUM_BUILD_OPTIMAL)
  add_subdirectory(optimal/cpp)
endif()
