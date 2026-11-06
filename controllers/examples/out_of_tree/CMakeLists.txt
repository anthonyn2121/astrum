# An out-of-tree ASTRUM controller: one law on the installed C++ SDK.
#
#   cmake --install build/controllers --prefix <prefix>
#   cmake -S controllers/examples/out_of_tree -B <build> -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build <build>
#   <build>/hold-controller --id 1 --heading 90
cmake_minimum_required(VERSION 3.22)
# C as well as CXX: CapnProto's package config probes libatomic with a C program.
project(astrum_out_of_tree_example LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(astrum 3 CONFIG REQUIRED)

add_executable(hold-controller main.cpp)
target_link_libraries(hold-controller PRIVATE astrum::runtime)
