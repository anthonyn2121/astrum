# astrum-minsnap-controller: minimum-snap race controller for the F450 (built from controllers/).
# The planning / tracking modules are pure (no sockets); the executable is the Law on the SDK runtime.
add_library(astrum_minsnap STATIC
  src/astar.cpp src/course.cpp src/flatness.cpp src/frames.cpp src/linalg.cpp src/minsnap.cpp
  src/planner.cpp src/race_law.cpp src/tracking.cpp)
astrum_warnings(astrum_minsnap)
target_include_directories(astrum_minsnap PUBLIC src)
target_link_libraries(astrum_minsnap PUBLIC astrum::runtime)

add_executable(astrum-minsnap-controller src/main.cpp)
astrum_warnings(astrum-minsnap-controller)
target_link_libraries(astrum-minsnap-controller PRIVATE astrum_minsnap astrum::runtime)

if(ASTRUM_BUILD_TESTS)
  astrum_test(astrum_minsnap_tests
    tests/linalg_test.cpp tests/minsnap_test.cpp tests/astar_test.cpp tests/flatness_test.cpp
    tests/tracking_test.cpp tests/planner_test.cpp tests/race_law_test.cpp tests/closed_loop_test.cpp)
  target_link_libraries(astrum_minsnap_tests PRIVATE astrum_minsnap)
endif()
