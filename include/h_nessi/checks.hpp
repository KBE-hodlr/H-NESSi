/**
 * @file checks.hpp
 * @brief Precondition / state / post-condition checks of H-NESSi: error types, check macros and the OpenMP error policy.
 *
 * ## Error types
 * All runtime errors raised by H-NESSi checks derive from a standard exception, so callers can catch either the
 * h_nessi type or its std base:
 *  - h_nessi::precondition_error : std::invalid_argument -- a bad argument or an inconsistent pair of objects
 *    (geometry mismatch, out-of-range parameter, non-Hermitian h(t), ...);
 *  - h_nessi::state_error        : std::logic_error      -- a call in the wrong order / on an object in the wrong
 *    state (step not synchronised with update_blocks, stale convolution tensor, no saved G0, ...);
 *  - h_nessi::numerical_error    : std::runtime_error    -- a numerical post-condition failed (non-finite result).
 *
 * ## Tiers (maintainer decision D-6)
 *  - Tier 1, HNESSI_ASSERT(cond, where, values...): hot-path / internal invariants. Compiled out with NDEBUG
 *    (Release); active in Debug and in the RelWithAsserts build type. On failure it prints the message and aborts
 *    (like assert).
 *  - Tier 2, always on: HNESSI_REQUIRE (precondition_error), HNESSI_REQUIRE_STATE (state_error) and HNESSI_ENSURE
 *    (numerical_error). They are evaluated once per library call, never inside a library loop, and cost O(1) (or a
 *    negligible fraction of the call they guard).
 *
 * Message format (the exception's what()):
 *   h_nessi::<Class>::<function>: <condition> violated (<name>=<value>, ...)[; <note>]
 * e.g.
 *   h_nessi::dyson::dyson_timestep: tstp == Sigma.tstpmk()+k_ violated (tstp=57, Sigma.tstpmk()=55, k_=2); call
 *   Sigma.update_blocks(I) exactly once per time step
 * The values are formatted only on the failure path; they are passed as HNESSI_KV(expr) items, and an optional
 * h_nessi::detail::note{"..."} item adds the trailing hint.
 *
 * ## OpenMP error policy
 * An exception that leaves an OpenMP parallel region calls std::terminate, and user programs (e.g. mpi_example) call
 * solver routines inside their own `#pragma omp parallel`. Therefore:
 *  1. A failed tier-2 check **outside** any parallel region throws the h_nessi exception.
 *  2. A failed tier-2 check inside a **user-owned** parallel region (omp_in_parallel() and not inside an H-NESSi
 *     region) prints the full message, the MPI rank (if MPI is initialised), the OpenMP thread and file:line to
 *     stderr and calls std::abort(): a library cannot unwind a user's region safely; the batch launcher then ends
 *     the other ranks. Catch h_nessi exceptions only around code whose parallel regions you own.
 *  3. **Library-owned** parallel regions (the mpi_comm `_spawn` routines) run their loop bodies through
 *     h_nessi::detail::omp_exception_collector: every thread stores the first exception (also exceptions thrown by
 *     user lambdas), the region completes, and the exception is rethrown on the calling thread, so a library call
 *     made outside a user region stays catchable. If the calling thread is itself inside a user region, rule 2
 *     applies to the rethrow.
 *  4. `_nospawn` routines run inside a user region by contract; their checks depend only on shared inputs, so every
 *     thread fails identically (consistent abort, no thread left waiting at a barrier).
 */
#ifndef HNESSI_CHECKS_HPP
#define HNESSI_CHECKS_HPP

#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace h_nessi {

/** Largest supported integration order (Integrator accepts 0..KMAX; solver objects 1..KMAX). */
constexpr int KMAX = 6;

/** A bad argument or an inconsistent pair of objects passed to an H-NESSi routine. */
class precondition_error : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

/** An H-NESSi routine was called in the wrong order or on an object in the wrong state. */
class state_error : public std::logic_error {
public:
  using std::logic_error::logic_error;
};

/** A numerical post-condition failed (non-finite result, ...). */
class numerical_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

namespace detail {

enum class error_kind { precondition, state, numerical, assertion };

/** Trailing hint appended to a check message as "; <text>". */
struct note {
  std::string text;
};

/** "name=value" (formatted on the failure path only). */
template <class T>
inline std::string kv(const char *name, const T &value) {
  std::ostringstream s;
  s.precision(10);
  s << name << "=" << value;
  return s.str();
}

inline void append_item(std::string &vals, std::string & /*nt*/, const std::string &item) {
  if (!vals.empty()) vals += ", ";
  vals += item;
}
inline void append_item(std::string & /*vals*/, std::string &nt, const note &n) {
  if (!nt.empty()) nt += "; ";
  nt += n.text;
}

/** Joins the HNESSI_KV items and the optional note of a check into (values, note). */
template <class... Args>
inline std::pair<std::string, std::string> join_items(const Args &...args) {
  std::string vals, nt;
  (append_item(vals, nt, args), ...);
  return {vals, nt};
}

/** Builds "h_nessi::<where>: <cond> violated (<values>)[; <note>]". */
std::string format_message(const char *where, const char *cond, const std::pair<std::string, std::string> &items);

/**
 * Failure path of every check: applies the OpenMP error policy (see the file comment) and either throws the exception
 * of `kind` or prints the message and aborts. Never returns.
 */
[[noreturn]] void fail(error_kind kind, const std::string &message, const char *file, int line);

/** Prints "h_nessi warning: <message>" (with the MPI rank when MPI is initialised) to stderr. */
void warn(const std::string &message);

/** Depth of H-NESSi-owned parallel regions on the calling thread (0: none). */
int &library_region_depth() noexcept;

/** RAII marker: the calling thread is executing inside an H-NESSi-owned parallel region. */
struct library_region_scope {
  library_region_scope() noexcept { ++library_region_depth(); }
  ~library_region_scope() { --library_region_depth(); }
  library_region_scope(const library_region_scope &) = delete;
  library_region_scope &operator=(const library_region_scope &) = delete;
};

/**
 * Capture-and-rethrow helper for H-NESSi-owned OpenMP regions (policy item 3):
 * @code
 *   detail::omp_exception_collector errs;
 *   #pragma omp parallel for
 *   for(int i = 0; i < n; i++) errs.run([&] { body(i); });
 *   errs.rethrow();   // on the calling thread, after the region
 * @endcode
 * After the first failure the remaining iterations are skipped (their results would be discarded anyway).
 */
class omp_exception_collector {
public:
  bool failed() const noexcept { return failed_.load(std::memory_order_relaxed); }

  template <class F>
  void run(F &&f) noexcept {
    if (failed()) return;
    library_region_scope scope;
    try {
      f();
    } catch (...) {
      capture();
    }
  }

  void capture() noexcept {
    std::lock_guard<std::mutex> lock(m_);
    if (!ptr_) ptr_ = std::current_exception();
    failed_.store(true, std::memory_order_relaxed);
  }

  /** Rethrows the first captured exception (no-op if none). Inside a user-owned region: print and abort instead. */
  void rethrow();

private:
  std::exception_ptr ptr_;
  std::atomic<bool> failed_{false};
  std::mutex m_;
};

/**
 * ST-10: records which thread is using an object (dyson, dlr_info), so that concurrent use of one object by two
 * threads is detected (ISSUES H3: one dyson and one dlr_info per thread). Copies and assignments start free, so the
 * owning class stays copyable. Cost: one atomic load and one compare-exchange per outermost call.
 */
class single_thread_owner {
public:
  single_thread_owner() = default;
  single_thread_owner(const single_thread_owner &) noexcept {}
  single_thread_owner &operator=(const single_thread_owner &) noexcept { return *this; }
  /**
   * Acquires the object for the calling thread. Returns true if this call acquired it (outermost call), false if the
   * calling thread already holds it (nested call). If another thread holds it, fails with h_nessi::state_error
   * (OpenMP policy: inside a user parallel region the message is printed and the program aborts).
   */
  bool acquire(const char *where);
  void release() noexcept { owner_.store(0, std::memory_order_release); }
  /** true if some thread holds the object (for tests). */
  bool held() const noexcept { return owner_.load(std::memory_order_acquire) != 0; }

private:
  std::atomic<std::uintptr_t> owner_{0};
};

/** RAII: acquire a single_thread_owner for the scope (release only if this scope acquired it). */
class owner_scope {
public:
  owner_scope(single_thread_owner &o, const char *where) : o_(o), outer_(o.acquire(where)) {}
  ~owner_scope() {
    if (outer_) o_.release();
  }
  owner_scope(const owner_scope &) = delete;
  owner_scope &operator=(const owner_scope &) = delete;
  bool outer() const noexcept { return outer_; }

private:
  single_thread_owner &o_;
  bool outer_;
};

}  // namespace detail
}  // namespace h_nessi

#if defined(__GNUC__) || defined(__clang__)
#define HNESSI_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HNESSI_UNLIKELY(x) (x)
#endif

/** One "name=value" item of a check message. */
#define HNESSI_KV(x) ::h_nessi::detail::kv(#x, (x))

#define HNESSI_CHECK_IMPL_(kind, cond, where, ...)                                                                 \
  do {                                                                                                             \
    if (HNESSI_UNLIKELY(!(cond))) {                                                                                \
      ::h_nessi::detail::fail(kind,                                                                                \
                              ::h_nessi::detail::format_message(where, #cond,                                      \
                                                                ::h_nessi::detail::join_items(__VA_ARGS__)),       \
                              __FILE__, __LINE__);                                                                 \
    }                                                                                                              \
  } while (0)

/** Tier 2: argument precondition, throws h_nessi::precondition_error (OpenMP policy applies). */
#define HNESSI_REQUIRE(cond, where, ...) \
  HNESSI_CHECK_IMPL_(::h_nessi::detail::error_kind::precondition, cond, where, __VA_ARGS__)
/** Tier 2: call-order / object-state precondition, throws h_nessi::state_error. */
#define HNESSI_REQUIRE_STATE(cond, where, ...) \
  HNESSI_CHECK_IMPL_(::h_nessi::detail::error_kind::state, cond, where, __VA_ARGS__)
/** Tier 2: numerical post-condition, throws h_nessi::numerical_error. */
#define HNESSI_ENSURE(cond, where, ...) \
  HNESSI_CHECK_IMPL_(::h_nessi::detail::error_kind::numerical, cond, where, __VA_ARGS__)

/** Tier 1: hot-path / internal invariant; compiled out with NDEBUG, otherwise prints the message and aborts. */
#ifdef NDEBUG
#define HNESSI_ASSERT(cond, where, ...) ((void)0)
#else
#define HNESSI_ASSERT(cond, where, ...) \
  HNESSI_CHECK_IMPL_(::h_nessi::detail::error_kind::assertion, cond, where, __VA_ARGS__)
#endif

#endif  // HNESSI_CHECKS_HPP
