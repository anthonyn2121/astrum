cmake_minimum_required(VERSION 3.16)
project(astrum_optimal_controller CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# NOTE: we deliberately do NOT use find_package(CapnProto). Its CMake config on
# this image (capnp 1.0.1) has a spuriously fatal libatomic probe — 8-byte
# atomics are lock-free compiler builtins on x86-64, so no libatomic is needed.
# pkg-config gives us the same capnp/kj libs without that broken check.
find_package(PkgConfig REQUIRED)
pkg_check_modules(CAPNP REQUIRED capnp) # Requires: kj → pulls in libkj too
pkg_check_modules(EIGEN3 REQUIRED eigen3) # header-only; libeigen3-dev in the image
find_package(Threads REQUIRED)

# Zenoh transport: the zenoh-cpp headers (header-only, /usr/include/zenoh.hxx)
# wrap the zenoh-c runtime (libzenohc.so). Both ship as debs installed in the
# image (see docker/Dockerfile). ZENOHCXX_ZENOHC selects the zenoh-c backend.
# The shipped zenohc.pc has a stale /usr/local prefix, so we don't pkg-config it;
# headers and libzenohc.so are in the default paths.

# Generate C++ from the ONE shared schema the Rust side uses, guaranteeing the
# controller and sim are wire-compatible.
set(SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../crates/astrum-proto/schema)
set(SCHEMA ${SCHEMA_DIR}/astrum.capnp)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${GEN_DIR})
add_custom_command(
  OUTPUT ${GEN_DIR}/astrum.capnp.c++ ${GEN_DIR}/astrum.capnp.h
  COMMAND capnp compile --src-prefix=${SCHEMA_DIR} -oc++:${GEN_DIR} ${SCHEMA}
  DEPENDS ${SCHEMA}
  COMMENT "Generating C++ bindings from astrum.capnp"
  VERBATIM)

# One binary; the law is chosen at runtime via `--law`. Add a laws/<name>.cpp
# here (and its factory to laws.hpp / make_law) to grow the controller.
add_executable(astrum-optimal-controller
  src/runtime.cpp
  laws/lqr.cpp
  laws/mpc.cpp
  ${GEN_DIR}/astrum.capnp.c++)
target_compile_definitions(astrum-optimal-controller PRIVATE ZENOHCXX_ZENOHC)
target_include_directories(astrum-optimal-controller PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GEN_DIR} ${CAPNP_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
target_link_directories(astrum-optimal-controller PRIVATE ${CAPNP_LIBRARY_DIRS})
target_link_libraries(astrum-optimal-controller PRIVATE
  ${CAPNP_LIBRARIES} zenohc Threads::Threads)
