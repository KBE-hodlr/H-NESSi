// read_inputfile.hpp parser (doc 30).  Single translation unit (the two-TU link test for B19 is the separate
// target test_read_inputfile_2tu, see link/).  B15: empty boolean value must throw like the other overloads.
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "common/test_util.hpp"
#include "h_nessi/read_inputfile.hpp"

namespace {
std::string write_input(const std::string &name, const std::string &content) {
  std::ofstream f(name);
  f << content;
  return name;
}
}  // namespace

TEST(ReadInputfile, ParsesTypedValues) {
  std::string fn = write_input("ri_values.inp",
                               "__nt=  200\n__h=  0.03\n__flag=  true\n__flag2= false\n__z=  1.5 -2.25\n__vec= 1 2 3.5\n# comment\n");
  int nt = 0;
  double h = 0;
  bool f1 = false, f2 = true;
  cplx z;
  std::vector<double> v;
  find_param((char *)fn.c_str(), "__nt=", nt);
  find_param((char *)fn.c_str(), "__h=", h);
  find_param((char *)fn.c_str(), "__flag=", f1);
  find_param((char *)fn.c_str(), "__flag2=", f2);
  find_param((char *)fn.c_str(), "__z=", z);
  find_param((char *)fn.c_str(), "__vec=", v, 3);
  EXPECT_EQ(nt, 200);
  EXPECT_DOUBLE_EQ(h, 0.03);
  EXPECT_TRUE(f1);
  EXPECT_FALSE(f2);
  EXPECT_EQ(z, cplx(1.5, -2.25));
  ASSERT_EQ(v.size(), 3u);
  EXPECT_DOUBLE_EQ(v[2], 3.5);
}

TEST(ReadInputfile, MissingIntValueThrows) {
  std::string fn = write_input("ri_empty_int.inp", "__nt=\n");
  int nt = 0;
  EXPECT_ANY_THROW(find_param((char *)fn.c_str(), "__nt=", nt));
}

TEST(ReadInputfile, MissingFlagThrows) {
  std::string fn = write_input("ri_noflag.inp", "__other=  1\n");
  int nt = 0;
  EXPECT_ANY_THROW(find_param((char *)fn.c_str(), "__nt=", nt));
}

TEST(ReadInputfile, KnownBug_B15_EmptyBoolThrows) {
  hn_test::run_isolated([] {
    std::string fn = write_input("ri_empty_bool.inp", "__checkpoint=\n");
    bool b = false;
    EXPECT_ANY_THROW(find_param((char *)fn.c_str(), "__checkpoint=", b));
  });
}
