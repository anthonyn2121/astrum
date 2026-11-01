# Shared C++ controller code.
#
#   astrum::core    wire codec (schema-generated types live in namespace astrum::wire),
#                   plain VehicleState/Command types, CLI options, control blocks.
#                   No Zenoh — used by the fixture tool and unit tests too.
#   astrum::client  one Zenoh session with the transport contract (peer, connect-out,
#                   latest-value telemetry, DROP/express/INTERACTIVE_HIGH commands,
#                   seq + telemetryTick, startup/telemetry timeouts).

# --- Schema codegen: the ONE astrum.capnp the Rust side uses -----------------
set(ASTRUM_SCHEMA_DIR ${ASTRUM_REPO_ROOT}/crates/astrum-proto/schema)
set(CAPNPC_SRC_PREFIX ${ASTRUM_SCHEMA_DIR})
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})
capnp_generate_cpp(ASTRUM_CAPNP_SRCS ASTRUM_CAPNP_HDRS ${ASTRUM_SCHEMA_DIR}/astrum.capnp)

# Generated code: compiled without our -Werror; its headers are SYSTEM includes.
add_library(astrum_wire_gen STATIC ${ASTRUM_CAPNP_SRCS})
target_include_directories(astrum_wire_gen SYSTEM PUBLIC ${CAPNPC_OUTPUT_DIR})
target_link_libraries(astrum_wire_gen PUBLIC CapnProto::capnp)

add_library(astrum_core STATIC src/wire.cpp src/options.cpp src/control.cpp)
add_library(astrum::core ALIAS astrum_core)
astrum_warnings(astrum_core)
target_include_directories(astrum_core PUBLIC include)
target_link_libraries(astrum_core PRIVATE astrum_wire_gen)

add_library(astrum_client STATIC src/client.cpp)
add_library(astrum::client ALIAS astrum_client)
astrum_warnings(astrum_client)
target_link_libraries(astrum_client PUBLIC astrum_core PRIVATE zenohcxx::zenohc Threads::Threads)

# --- Cross-language wire fixtures (crates/astrum-proto/tests/fixtures) -------
#   astrum-wire-fixture write  <dir>   encode values.json -> <kind>.cpp.bin
#   astrum-wire-fixture verify <dir>   decode every <kind>.<lang>.bin, compare to values.json
set(ASTRUM_FIXTURE_DIR ${ASTRUM_REPO_ROOT}/crates/astrum-proto/tests/fixtures)
add_executable(astrum-wire-fixture tools/wire_fixture.cpp)
astrum_warnings(astrum-wire-fixture)
target_link_libraries(astrum-wire-fixture PRIVATE astrum_core nlohmann_json::nlohmann_json)
add_test(NAME wire_fixtures COMMAND astrum-wire-fixture verify ${ASTRUM_FIXTURE_DIR})

# --- Unit tests ---------------------------------------------------------------
astrum_test(astrum_common_tests tests/math_test.cpp tests/wire_test.cpp tests/control_test.cpp
            tests/options_test.cpp tests/client_test.cpp)
target_link_libraries(astrum_common_tests PRIVATE astrum::client)
