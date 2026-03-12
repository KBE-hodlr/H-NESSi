set(TEST_SOURCES
  my_tests.cpp
#  linalg.cpp
)

# Define test executable(s)
add_executable(my_tests ${TEST_SOURCES})

# Link against gtest and your main project libraries/executable
target_link_libraries(my_tests PRIVATE HighFive GTest::gtest_main hodlr)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(my_tests)

