# astrum-controller: the example hold / autopilot controller (built from controllers/).
add_library(astrum_hold_law STATIC src/hold_law.cpp)
astrum_warnings(astrum_hold_law)
target_include_directories(astrum_hold_law PUBLIC src)
target_link_libraries(astrum_hold_law PUBLIC astrum::runtime)

add_executable(astrum-controller src/main.cpp)
astrum_warnings(astrum-controller)
target_link_libraries(astrum-controller PRIVATE astrum_hold_law astrum::runtime)

if(ASTRUM_BUILD_TESTS)
  astrum_test(astrum_controller_tests tests/hold_law_test.cpp)
  target_link_libraries(astrum_controller_tests PRIVATE astrum_hold_law)
endif()
