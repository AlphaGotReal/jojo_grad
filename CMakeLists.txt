cmake_minimum_required(VERSION 3.14)

project(JojoGrad VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Library ----

add_library(jojograd src/variable.cpp)
target_include_directories(jojograd PUBLIC include)

# ---- Examples ----

add_executable(example_basic_ops   examples/basic_ops.cpp)
add_executable(example_chain_rule  examples/chain_rule.cpp)

target_link_libraries(example_basic_ops  PRIVATE jojograd)
target_link_libraries(example_chain_rule PRIVATE jojograd)

# ---- Tests ----

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
# Prevent GTest from overriding our compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tests tests/test_variable.cpp)
target_link_libraries(tests PRIVATE jojograd GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(tests)
