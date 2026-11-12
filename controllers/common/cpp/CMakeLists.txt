# The C++ controller SDK (controllers/README.md):
#
#   astrum::wire     astrum/types.hpp + astrum/wire.hpp: plain types and the Cap'n Proto
#                    codec; the schema-generated code (namespace astrum::wire) is private
#   astrum::control  astrum/control.hpp + astrum/math.hpp: SaturatedPi, OuterLoops
#   astrum::cli      astrum/options.hpp + astrum/cli.hpp: Options and the shared flags
#   astrum::client   astrum/transport.hpp + astrum/loop.hpp + astrum/log.hpp: the Zenoh
#                    transport and the control loop (no CLI)
#   astrum::runtime  astrum/runtime.hpp: LawEntry, LawContext, run_main
#
# Installed with an export set (lib/cmake/astrum) for find_package(astrum CONFIG).

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ASTRUM_INCLUDE_DIRS $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# --- Schema codegen: the ONE astrum.capnp the Rust side uses -----------------
set(ASTRUM_SCHEMA_DIR ${ASTRUM_REPO_ROOT}/crates/astrum-proto/schema)
set(CAPNPC_SRC_PREFIX ${ASTRUM_SCHEMA_DIR})
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})
capnp_generate_cpp(ASTRUM_CAPNP_SRCS ASTRUM_CAPNP_HDRS ${ASTRUM_SCHEMA_DIR}/astrum.capnp)

# Generated code: compiled without our -Werror; its headers are SYSTEM includes and are
# not installed (astrum::wire links it privately; it is exported link-only).
add_library(astrum_wire_gen STATIC ${ASTRUM_CAPNP_SRCS})
set_target_properties(astrum_wire_gen PROPERTIES EXPORT_NAME wire_gen
                                                 POSITION_INDEPENDENT_CODE ON)
target_include_directories(astrum_wire_gen SYSTEM PUBLIC $<BUILD_INTERFACE:${CAPNPC_OUTPUT_DIR}>)
target_link_libraries(astrum_wire_gen PUBLIC CapnProto::capnp)

# astrum_sdk_library(<name> <sources...>): astrum_<name> exported as astrum::<name>.
function(astrum_sdk_library name)
  add_library(astrum_${name} STATIC ${ARGN})
  add_library(astrum::${name} ALIAS astrum_${name})
  set_target_properties(astrum_${name} PROPERTIES EXPORT_NAME ${name}
                                                  POSITION_INDEPENDENT_CODE ON)
  astrum_warnings(astrum_${name})
  target_include_directories(astrum_${name} PUBLIC ${ASTRUM_INCLUDE_DIRS})
  target_compile_features(astrum_${name} PUBLIC cxx_std_20)
endfunction()

astrum_sdk_library(wire src/wire.cpp)
target_link_libraries(astrum_wire PRIVATE astrum_wire_gen)

astrum_sdk_library(control src/control.cpp)
target_link_libraries(astrum_control PUBLIC astrum_wire)

astrum_sdk_library(cli src/options.cpp src/cli.cpp)
target_link_libraries(astrum_cli PUBLIC astrum_wire)

astrum_sdk_library(client src/log.cpp src/loop.cpp src/transport.cpp)
target_link_libraries(astrum_client PUBLIC astrum_wire PRIVATE zenohcxx::zenohc Threads::Threads)

astrum_sdk_library(runtime src/runtime.cpp)
target_link_libraries(astrum_runtime PUBLIC astrum_client astrum_cli astrum_control)

install(TARGETS astrum_wire_gen astrum_wire astrum_control astrum_cli astrum_client astrum_runtime
        EXPORT astrumTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/astrum DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(ASTRUM_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/astrum)
install(EXPORT astrumTargets NAMESPACE astrum:: DESTINATION ${ASTRUM_CMAKE_DIR})
configure_package_config_file(cmake/astrumConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/astrumConfig.cmake
                              INSTALL_DESTINATION ${ASTRUM_CMAKE_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/astrumConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/astrumConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/astrumConfigVersion.cmake
        DESTINATION ${ASTRUM_CMAKE_DIR})

if(NOT ASTRUM_BUILD_TESTS)
  return()
endif()

# --- Cross-language wire fixtures (crates/astrum-proto/tests/fixtures) -------
#   astrum-wire-fixture write  <dir>   encode values.json -> <kind>.cpp.bin
#   astrum-wire-fixture verify <dir>   decode every <kind>.<lang>.bin, compare to values.json
set(ASTRUM_FIXTURE_DIR ${ASTRUM_REPO_ROOT}/crates/astrum-proto/tests/fixtures)
add_executable(astrum-wire-fixture tools/wire_fixture.cpp)
astrum_warnings(astrum-wire-fixture)
target_link_libraries(astrum-wire-fixture PRIVATE astrum::wire nlohmann_json::nlohmann_json)
add_test(NAME wire_fixtures COMMAND astrum-wire-fixture verify ${ASTRUM_FIXTURE_DIR})

# --- Unit tests ---------------------------------------------------------------
astrum_test(astrum_common_tests tests/math_test.cpp tests/wire_test.cpp tests/control_test.cpp
            tests/options_test.cpp tests/loop_test.cpp tests/runtime_test.cpp)
target_link_libraries(astrum_common_tests PRIVATE astrum::runtime)
