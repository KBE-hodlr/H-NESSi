// Failure path of the H-NESSi checks (include/h_nessi/checks.hpp): message formatting and the OpenMP error policy.
#include "h_nessi/checks.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace h_nessi {
namespace detail {

int &library_region_depth() noexcept {
  thread_local int depth = 0;
  return depth;
}

namespace {

// MPI rank of this process, or -1 if MPI is not (or no longer) initialised.
int mpi_rank_or_minus_one() {
  int init = 0, fin = 0;
  MPI_Initialized(&init);
  if (!init) return -1;
  MPI_Finalized(&fin);
  if (fin) return -1;
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

bool in_user_parallel_region() {
#ifdef _OPENMP
  return omp_in_parallel() && library_region_depth() == 0;
#else
  return false;
#endif
}

// Prints the full message with MPI rank, OpenMP thread and file:line, then aborts.
[[noreturn]] void print_and_abort(const char *what_kind, const std::string &message, const char *file, int line) {
  std::string where;
  int rank = mpi_rank_or_minus_one();
  if (rank >= 0) where += "MPI rank " + std::to_string(rank) + ", ";
#ifdef _OPENMP
  where += "OpenMP thread " + std::to_string(omp_get_thread_num()) + "/" + std::to_string(omp_get_num_threads()) +
           (omp_in_parallel() ? " (inside a parallel region)" : "") + ", ";
#endif
  where += std::string(file ? file : "?") + ":" + std::to_string(line);
  std::fprintf(stderr, "h_nessi %s: %s\n  [%s]\n  aborting (an exception cannot leave an OpenMP parallel region; "
                       "see the OpenMP error policy in h_nessi/checks.hpp)\n",
               what_kind, message.c_str(), where.c_str());
  std::fflush(stderr);
  std::abort();
}

}  // namespace

std::string format_message(const char *where, const char *cond, const std::pair<std::string, std::string> &items) {
  std::string m = std::string("h_nessi::") + where + ": " + cond + " violated";
  if (!items.first.empty()) m += " (" + items.first + ")";
  if (!items.second.empty()) m += "; " + items.second;
  return m;
}

void fail(error_kind kind, const std::string &message, const char *file, int line) {
  if (kind == error_kind::assertion) print_and_abort("assertion failed", message, file, line);
  if (in_user_parallel_region()) print_and_abort("check failed inside a user OpenMP parallel region", message, file, line);
  switch (kind) {
    case error_kind::state: throw state_error(message);
    case error_kind::numerical: throw numerical_error(message);
    default: throw precondition_error(message);
  }
}

void warn(const std::string &message) {
  int rank = mpi_rank_or_minus_one();
  std::string prefix = rank >= 0 ? "h_nessi warning (MPI rank " + std::to_string(rank) + "): " : "h_nessi warning: ";
  std::fprintf(stderr, "%s%s\n", prefix.c_str(), message.c_str());
  std::fflush(stderr);
}

bool single_thread_owner::acquire(const char *where) {
  // a per-thread token: the address of a thread_local (non-zero, distinct for concurrently running threads)
  const std::uintptr_t me = reinterpret_cast<std::uintptr_t>(&library_region_depth());
  if (owner_.load(std::memory_order_acquire) == me) return false;  // nested call on the owning thread
  std::uintptr_t expected = 0;
  if (owner_.compare_exchange_strong(expected, me, std::memory_order_acq_rel)) return true;
  fail(error_kind::state,
       format_message(where, "the object is not in use by another thread", {"", "two threads use the same object "
                      "concurrently; use one object per thread (ISSUES H3)"}),
       __FILE__, __LINE__);
}

void omp_exception_collector::rethrow() {
  std::exception_ptr p;
  {
    std::lock_guard<std::mutex> lock(m_);
    p = ptr_;
    ptr_ = nullptr;
    failed_.store(false, std::memory_order_relaxed);
  }
  if (!p) return;
  if (in_user_parallel_region()) {
    std::string what = "non-standard exception";
    try {
      std::rethrow_exception(p);
    } catch (const std::exception &e) {
      what = e.what();
    } catch (...) {
    }
    print_and_abort("error inside a user OpenMP parallel region", what, __FILE__, __LINE__);
  }
  std::rethrow_exception(p);
}

}  // namespace detail
}  // namespace h_nessi
