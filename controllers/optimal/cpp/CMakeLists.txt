# astrum-optimal-controller: LQR / MPC laws on the shared client (built from controllers/).
# Add a laws/<name>.cpp to astrum_laws (and its factory to laws.hpp / make_law()).
add_library(astrum_laws STATIC laws/lqr.cpp laws/mpc.cpp)
astrum_warnings(astrum_laws)
target_include_directories(astrum_laws PUBLIC include)
target_link_libraries(astrum_laws PUBLIC astrum::core Eigen3::Eigen)

add_executable(astrum-optimal-controller src/runtime.cpp)
astrum_warnings(astrum-optimal-controller)
target_link_libraries(astrum-optimal-controller PRIVATE astrum_laws astrum::client)

astrum_test(astrum_optimal_tests tests/optimal_math_test.cpp tests/laws_test.cpp)
target_link_libraries(astrum_optimal_tests PRIVATE astrum_laws)
