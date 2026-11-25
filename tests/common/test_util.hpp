// Shared helpers for the H-NESSi regression suite.
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace hn_test {

// Observed convergence rate between two errors obtained with step h and h/2.
inline double rate(double e_coarse, double e_fine) { return std::log2(e_coarse / e_fine); }

// Run `body` in a forked child process and report the outcome as an ordinary gtest result in the parent.
//
// Used for every known-bug test (and every test that may crash): a segfault or a failed assert() in the
// child becomes a normal test failure, so that ctest's WILL_FAIL property works (WILL_FAIL does not
// invert crashes).  gtest assertions inside `body` print their messages from the child; the child exits
// with status 1 if any of them failed.
//
// NOTE: fork() is safe here because ctest runs every test in its own process. Do not run OpenMP regions
// in the parent before calling this helper (libgomp is not fork-safe once its thread pool exists).
inline void run_isolated(const std::function<void()> &body) {
  std::fflush(stdout);
  std::fflush(stderr);
  std::cout.flush();
  pid_t pid = fork();
  ASSERT_GE(pid, 0) << "fork failed";
  if (pid == 0) {
    int code = 0;
    try {
      body();
      code = ::testing::Test::HasFailure() ? 1 : 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[child] uncaught std::exception: %s\n", e.what());
      code = 2;
    } catch (...) {
      std::fprintf(stderr, "[child] uncaught non-std exception\n");
      code = 3;
    }
    std::fflush(stdout);
    std::fflush(stderr);
    std::cout.flush();
    _exit(code);
  }
  int st = 0;
  waitpid(pid, &st, 0);
  if (WIFSIGNALED(st)) {
    FAIL() << "child process killed by signal " << WTERMSIG(st) << " (crash / failed assert)";
  } else if (WIFEXITED(st) && WEXITSTATUS(st) != 0) {
    FAIL() << "child process reported failure (exit code " << WEXITSTATUS(st) << ")";
  }
}

}  // namespace hn_test
