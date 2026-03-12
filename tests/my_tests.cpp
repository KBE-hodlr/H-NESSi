#include <gtest/gtest.h>

int add(int a, int b) {
  return a+b;
}

TEST(MathTests, Addition) {
    EXPECT_EQ(add(2, 2), 4);
    EXPECT_EQ(add(-1, 5), 4);
    EXPECT_EQ(add(0, 0), 0);
}

TEST(MathTests, AdditionFailureExample) {
    EXPECT_NE(add(2, 2), 5); // This passes, just to illustrate EXPECT_NE
}
