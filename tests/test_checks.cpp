// WP6a: precondition / state / post-condition checks (include/h_nessi/checks.hpp, agent_docs/61_precondition_audit.md).
//  * the check macros: exception types, message format, tier-1 behaviour, OpenMP error policy (abort inside a user
//    parallel region, capture-and-rethrow in library-owned regions);
//  * constructor validation of herm_matrix_hodlr (HM-C*, B23, D2), Integrator (KMAX), dlr_info (H5, DL-*), dyson (DY-C*);
//  * the dyson entry checks: geometry bundle (SOLV), step synchronisation (DY-T1), bootstrap window (DY-S1),
//    green_from_H* tmax/window (H11), dlr pairing, finiteness post-conditions (L9, DY-P1);
//  * state checks ST-2 (G^M convolution tensor validity + hash, pre-v2 bosonic checkpoints) and ST-3 (saved G0);
//  * D-11: Hermiticity of h(t) (in-place symmetrisation, diagnostic, warn/throw tolerances, opt-out), complex h(-1);
//  * dlr_info concurrent construction (H3).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "common/test_util.hpp"
#include "h_nessi/checks.hpp"
#include "h_nessi/dlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/integration.hpp"

using namespace h_nessi;
using C = std::complex<double>;

namespace {
const C II(0, 1);

// expects `stmt` to throw exactly type E (or a subclass) and the message to contain `sub`
#define EXPECT_THROW_MSG(stmt, E, sub)                                                          \
  do {                                                                                          \
    try {                                                                                       \
      stmt;                                                                                     \
      ADD_FAILURE() << "no exception from: " #stmt;                                             \
    } catch (const E &e_) {                                                                     \
      EXPECT_NE(std::string(e_.what()).find(sub), std::string::npos) << "what(): " << e_.what(); \
    } catch (const std::exception &e_) {                                                        \
      ADD_FAILURE() << "wrong exception type from " #stmt ": " << e_.what();                    \
    }                                                                                           \
  } while (0)

// Small working system: nao = 2, Sigma = 0 (TTI), constant Hermitian h, after dyson_mat -> initGMConvTensor ->
// dyson_start, i.e. ready for dyson_timestep(k+1).
struct Sys {
  int nt, nao = 2, k, nlvl = 2, xi, r = 0;
  double beta = 4., mu = 0.2, h = 0.05;
  dlr_info dlr;
  Integration::Integrator I;
  herm_matrix_hodlr G, S;
  function H;
  dyson D;
  explicit Sys(int nt_ = 16, int k_ = 2, int xi_ = -1, bool rho_version = true)
      : nt(nt_), k(k_), xi(xi_), dlr(r, 40., 1e-12, beta, 2, xi_), I(k_), G(nt_, r, 2, 1e-12, 2, 2, xi_, k_),
        S(nt_, r, 2, 1e-12, 2, 2, xi_, k_), H(nt_, 2, 2), D(nt_, 2, k_, dlr, rho_version, false) {
    if (xi == 1) mu = -1.5;
    ZMatrix h0(2, 2);
    h0 << 0.3, C(0.2, 0.1), C(0.2, -0.1), -0.4;
    for (int t = -1; t < nt; t++) H.get_map(t) = h0;
    H.get_map(-1) = h0.real().cast<C>();  // h(-1) must be real (G^M is real)
    for (int t = 0; t <= k; t++) H.get_map(t) = H.get_map(-1);  // TTI bootstrap
    D.dyson_mat(G, mu, H, S, false, 1.0);
    G.initGMConvTensor(dlr);
    D.dyson_start(G, mu, H, S, I, h);
  }
  double step(int T) {
    G.update_blocks(I);
    S.update_blocks(I);
    return D.dyson_timestep(T, G, mu, H, S, I, h);
  }
};
}  // namespace

// ================================================================================================ checks.hpp itself
TEST(ChecksMacros, ExceptionTypesAndMessageFormat) {
  int tstp = 57, k_ = 2;
  auto f = [&](int which) {
    if (which == 0) HNESSI_REQUIRE(tstp < 10, "dyson::example", HNESSI_KV(tstp), HNESSI_KV(k_));
    if (which == 1) HNESSI_REQUIRE_STATE(tstp == k_, "dyson::example", HNESSI_KV(tstp), detail::note{"a hint"});
    if (which == 2) HNESSI_ENSURE(tstp < 0, "dyson::example");
  };
  try {
    f(0);
    FAIL();
  } catch (const precondition_error &e) {
    EXPECT_EQ(std::string(e.what()), "h_nessi::dyson::example: tstp < 10 violated (tstp=57, k_=2)");
  }
  try {
    f(1);
    FAIL();
  } catch (const state_error &e) {
    EXPECT_EQ(std::string(e.what()), "h_nessi::dyson::example: tstp == k_ violated (tstp=57); a hint");
  }
  try {
    f(2);
    FAIL();
  } catch (const numerical_error &e) {
    EXPECT_EQ(std::string(e.what()), "h_nessi::dyson::example: tstp < 0 violated");
  }
  // the std base classes catch them too
  EXPECT_THROW(f(0), std::invalid_argument);
  EXPECT_THROW(f(1), std::logic_error);
  EXPECT_THROW(f(2), std::runtime_error);
}

TEST(ChecksMacros, ValuesAreFormattedOnlyOnFailure) {
  int evaluated = 0;
  auto value = [&] {
    evaluated++;
    return 1;
  };
  HNESSI_REQUIRE(true, "x::y", HNESSI_KV(value()));
  EXPECT_EQ(evaluated, 0);
  EXPECT_THROW(HNESSI_REQUIRE(false, "x::y", HNESSI_KV(value())), precondition_error);
  EXPECT_EQ(evaluated, 1);
}

TEST(ChecksMacrosDeathTest, Tier1AssertAbortsOnlyWithoutNDEBUG) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  int evaluated = 0;
  auto cond = [&] {
    evaluated++;
    return false;
  };
  (void)cond;
#ifndef NDEBUG
  EXPECT_DEATH(HNESSI_ASSERT(cond(), "x::tier1", HNESSI_KV(evaluated)), "h_nessi assertion failed: h_nessi::x::tier1: cond\\(\\) violated");
#else
  HNESSI_ASSERT(cond(), "x::tier1", HNESSI_KV(evaluated));  // compiled out: the condition is not evaluated
  EXPECT_EQ(evaluated, 0);
#endif
}

// OpenMP policy item 2: a failed tier-2 check inside a user-owned parallel region prints and aborts.
// (The parallel regions are in helper functions: a #pragma cannot appear inside the EXPECT_DEATH macro argument.)
namespace {
void user_region_failing_check() {
#pragma omp parallel num_threads(2)
  { HNESSI_REQUIRE(omp_get_thread_num() < 0, "dyson::dyson_timestep", HNESSI_KV(omp_get_thread_num())); }
}
void user_region_unsynced_step() {
  Sys s;
#pragma omp parallel num_threads(2)
  {
#pragma omp master
    s.D.dyson_timestep(s.k + 1, s.G, s.mu, s.H, s.S, s.I, s.h);  // no update_blocks: DY-T1 fails
  }
}
void library_region_in_user_region() {
#pragma omp parallel num_threads(2)
  {
    detail::omp_exception_collector errs;
    errs.run([&] { HNESSI_REQUIRE(false, "mpi_comm::nested"); });
    errs.rethrow();
  }
}
}  // namespace

TEST(ChecksOpenMPDeathTest, UserRegionAbortsWithFullMessage) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(user_region_failing_check(),
               "check failed inside a user OpenMP parallel region: h_nessi::dyson::dyson_timestep: "
               "omp_get_thread_num\\(\\) < 0 violated(.|\n)*OpenMP thread [01]/2 \\(inside a parallel region\\), "
               ".*test_checks.cpp:[0-9]+");
  // the same through a real solver call: step out of sync inside the user's region
  EXPECT_DEATH(user_region_unsynced_step(),
               "inside a user OpenMP parallel region: h_nessi::dyson::dyson_timestep: tstp == G.tstpmk\\(\\)\\+k_ violated");
}

// OpenMP policy item 3: library-owned regions capture the first exception and rethrow it on the calling thread.
TEST(ChecksOpenMP, LibraryRegionCapturesAndRethrows) {
  detail::omp_exception_collector errs;
  std::vector<int> done(64, 0);
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 64; i++) {
    errs.run([&] {
      HNESSI_REQUIRE(i != 17, "mpi_comm::example", HNESSI_KV(i));  // throws (library region), not abort
      done[i] = 1;
    });
  }
  EXPECT_THROW_MSG(errs.rethrow(), precondition_error, "h_nessi::mpi_comm::example: i != 17 violated (i=17)");
  errs.rethrow();  // the collector is reset: no second throw
  // user exceptions (e.g. from a lambda) are captured as well
  detail::omp_exception_collector errs2;
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 8; i++) errs2.run([&] { if (i == 3) throw std::out_of_range("user lambda"); });
  EXPECT_THROW(errs2.rethrow(), std::out_of_range);
  EXPECT_EQ(detail::library_region_depth(), 0);
}

// A library region nested in a user region: the rethrow happens inside the user's region -> abort (policy item 2).
TEST(ChecksOpenMPDeathTest, LibraryRegionInsideUserRegionAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(library_region_in_user_region(),
               "error inside a user OpenMP parallel region: h_nessi::mpi_comm::nested: false violated");
}

// ================================================================================================ constructors
TEST(CtorChecks, HermMatrixHodlrGeometry) {
  // valid
  EXPECT_NO_THROW(herm_matrix_hodlr(16, 5, 2, 1e-10, 2, 2, -1, 2));
  EXPECT_NO_THROW(herm_matrix_hodlr(4, 5, 2, 1e-10, 1, 1, +1, 3));   // nt = k+1, 2^nlvl = nt
  EXPECT_NO_THROW(herm_matrix_hodlr(64, 5, 3, 1e-10, 1, 1, -1, 6));  // KMAX
  // HM-C7
  EXPECT_THROW_MSG(herm_matrix_hodlr(16, 5, 2, 1e-10, 2, 2, -1, 0), precondition_error,
                   "h_nessi::herm_matrix_hodlr::herm_matrix_hodlr: 1 <= k && k <= KMAX violated (k=0, KMAX=6)");
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 2, 1e-10, 2, 2, -1, 7), precondition_error);
  // HM-C1
  EXPECT_THROW_MSG(herm_matrix_hodlr(2, 5, 1, 1e-10, 2, 2, -1, 2), precondition_error, "nt >= k+1 violated (nt=2, k=2)");
  // HM-C2 (B23)
  EXPECT_THROW_MSG(herm_matrix_hodlr(16, 5, 0, 1e-10, 2, 2, -1, 2), precondition_error, "(nlvl=0, nt=16)");
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 5, 1e-10, 2, 2, -1, 2), precondition_error);
  // HM-C3
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 2, 0., 2, 2, -1, 2), precondition_error);
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 2, std::nan(""), 2, 2, -1, 2), precondition_error);
  // HM-C4
  EXPECT_THROW(herm_matrix_hodlr(16, 0, 2, 1e-10, 2, 2, -1, 2), precondition_error);
  // HM-C5 (D2 guard)
  EXPECT_THROW_MSG(herm_matrix_hodlr(16, 5, 2, 1e-10, 2, 3, -1, 2), precondition_error,
                   "size1 >= 1 && size1 == size2 violated (size1=2, size2=3); only square orbital matrices");
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 2, 1e-10, 0, 0, -1, 2), precondition_error);
  // HM-C6
  EXPECT_THROW(herm_matrix_hodlr(16, 5, 2, 1e-10, 2, 2, 0, 2), precondition_error);
}

TEST(CtorChecks, IntegratorOrderRange) {
  for (int k = 0; k <= KMAX; k++) EXPECT_NO_THROW(Integration::Integrator I(k)) << k;
  EXPECT_THROW_MSG(Integration::Integrator(-1), precondition_error,
                   "h_nessi::Integration::Integrator::Integrator: 0 <= k && k <= h_nessi::KMAX violated (k=-1");
  EXPECT_THROW(Integration::Integrator(KMAX + 1), precondition_error);
}

TEST(CtorChecks, DlrInfoParameters) {
  int r = 0;
  EXPECT_NO_THROW(dlr_info(r, 50., 1e-15, 4., 1, -1));  // the smallest accepted eps
  EXPECT_LT(r, 500);
  EXPECT_THROW_MSG(dlr_info(r, 50., 1e-16, 4., 1, -1), precondition_error, "eps >= 1e-15 && eps < 1 violated (eps=1e-16)");
  EXPECT_THROW(dlr_info(r, 50., 0., 4., 1, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, 50., 1., 4., 1, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, 0., 1e-10, 4., 1, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, std::numeric_limits<double>::infinity(), 1e-10, 4., 1, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, 50., 1e-10, -1., 1, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, 50., 1e-10, 4., 0, -1), precondition_error);
  EXPECT_THROW(dlr_info(r, 50., 1e-10, 4., 1, 0), precondition_error);
  EXPECT_THROW(dlr_info(r, 50., 1e-10, 4., 1, -1, 0), precondition_error);
  // H5: ntaumax below the DLR rank used to overflow dlrrf_/dlrit_; now the buffers hold libdlr's 500 and the rank is
  // checked against ntaumax after the call
  int r_full = 0;
  dlr_info full(r_full, 50., 1e-12, 4., 1, -1);
  EXPECT_THROW_MSG(dlr_info(r, 50., 1e-12, 4., 1, -1, r_full - 1), precondition_error, "input_r <= ntaumax violated");
  int r_exact = 0;
  dlr_info exact(r_exact, 50., 1e-12, 4., 1, -1, r_full);  // ntaumax == rank is fine
  EXPECT_EQ(r_exact, r_full);
  for (int i = 0; i < r_full; i++) EXPECT_EQ(exact.it0B(i), full.it0B(i));
}

// H3: dlr_it_build is now inside the critical section, so concurrent construction gives the serial result
TEST(DlrInfoThreads, ConcurrentConstructionMatchesSerial) {
  int r0 = 0;
  dlr_info ref(r0, 80., 1e-12, 6., 2, -1);
  const int N = 8;
  std::vector<int> rr(N, 0);
  std::vector<double> maxdiff(N, 1.);
#pragma omp parallel for num_threads(4) schedule(static, 1)
  for (int n = 0; n < N; n++) {
    int r = 0;
    dlr_info d(r, 80., 1e-12, 6., 2, -1);
    rr[n] = r;
    double m = 0;
    if (r == r0) {
      for (int i = 0; i < r; i++) m = std::max(m, std::abs(d.it0B(i) - ref.it0B(i)) + std::abs(d.dlrrf()[i] - ref.dlrrf()[i]));
      for (int i = 0; i < r * r * r; i++) m = std::max(m, std::abs(d.phi()[i] - ref.phi()[i]));
    }
    maxdiff[n] = m;
  }
  for (int n = 0; n < N; n++) {
    EXPECT_EQ(rr[n], r0);
    EXPECT_EQ(maxdiff[n], 0.0);
  }
}

TEST(CtorChecks, Dyson) {
  int r = 0;
  dlr_info dlr(r, 40., 1e-12, 4., 2, -1);
  EXPECT_NO_THROW(dyson(8, 2, 2, dlr));
  EXPECT_THROW(dyson(8, 2, 0, dlr), precondition_error);
  EXPECT_THROW(dyson(8, 2, 7, dlr), precondition_error);
  EXPECT_THROW(dyson(2, 2, 2, dlr), precondition_error);
  EXPECT_THROW(dyson(8, 0, 2, dlr), precondition_error);
  EXPECT_THROW_MSG(dyson(8, 3, 2, dlr), precondition_error, "dlr.size1() == nao && dlr.size2() == nao violated");
}

// ================================================================================================ dyson entry checks
TEST(DysonEntry, GeometryBundle) {
  Sys s;
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  // Sigma with another nlvl / nt / k / sig / r
  herm_matrix_hodlr S_nlvl(s.nt, s.r, 3, 1e-12, 2, 2, -1, s.k);
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, S_nlvl, s.I, s.h), precondition_error,
                   "G and Sigma must have the same geometry");
  herm_matrix_hodlr S_nt(s.nt - 1, s.r, 2, 1e-12, 2, 2, -1, s.k);
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, S_nt, s.I, s.h), precondition_error);
  herm_matrix_hodlr S_sig(s.nt, s.r, 2, 1e-12, 2, 2, +1, s.k);
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, S_sig, s.I, s.h), precondition_error);
  herm_matrix_hodlr S_k(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k + 1);
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, S_k, s.I, s.h), precondition_error);
  // Integrator of another order (H1c)
  Integration::Integrator I3(s.k + 1);
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, I3, s.h), precondition_error,
                   "G.k() == k_ && I.k() == k_ violated");
  // h, mu
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, 0.), precondition_error);
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, std::nan(""), s.H, s.S, s.I, s.h), precondition_error);
  // H too short / wrong size (DY-E9)
  function Hshort(T, 2, 2);
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, Hshort, s.S, s.I, s.h), precondition_error, "H.nt() >= tlast+1");
  // G longer than the dyson object (DY-E7)
  int r = 0;
  dlr_info dlr(r, 40., 1e-12, s.beta, 2, -1);
  dyson Dsmall(s.nt - 4, 2, s.k, dlr);
  EXPECT_THROW_MSG(Dsmall.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error, "G.nt() <= nt_ violated");
  // G of the other statistics than the dyson's dlr (dlr pairing)
  int rb = 0;
  dlr_info dlrb(rb, 40., 1e-12, s.beta, 2, +1);
  dyson Db(s.nt, 2, s.k, dlrb);
  EXPECT_THROW_MSG(Db.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error, "G.sig() == xi_ && G.r() == r_");
  // the public components carry the same checks
  EXPECT_THROW(s.D.dyson_timestep_ret(T, s.G, s.mu, s.H.ptr(0), S_nlvl, s.I, s.h), precondition_error);
  EXPECT_THROW(s.D.dyson_timestep_les(T, s.G, s.mu, s.H.ptr(0), s.S, I3, s.h), precondition_error);
  // and everything still works with the right arguments
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
}

TEST(DysonEntry, StepSynchronisation) {
  Sys s;
  int T = s.k + 1;
  // DY-T1: neither object advanced
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), state_error,
                   "h_nessi::dyson::dyson_timestep: tstp == G.tstpmk()+k_ violated (tstp=3, G.tstpmk()=0, k_=2)");
  // Sigma.update_blocks forgotten (H1e)
  s.G.update_blocks(s.I);
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), state_error,
                   "tstp == Sigma.tstpmk()+k_ violated (tstp=3, Sigma.tstpmk()=0, k_=2); call Sigma.update_blocks(I)");
  s.S.update_blocks(s.I);
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  // the same step can be solved again (corrector iterations)
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  // Sigma advanced twice (H1f)
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  s.S.update_blocks(s.I);
  EXPECT_THROW(s.D.dyson_timestep(T + 1, s.G, s.mu, s.H, s.S, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_timestep_tv(T + 1, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  // DY-T2: a bootstrap row through dyson_timestep
  EXPECT_THROW_MSG(s.D.dyson_timestep(s.k, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error, "k_ < tstp && tstp < G.nt()");
}

TEST(DysonEntry, BootstrapWindow) {
  Sys s;
  EXPECT_NO_THROW(s.D.dyson_start(s.G, s.mu, s.H, s.S, s.I, s.h));  // repeated bootstrap iterations are fine
  s.G.update_blocks(s.I);
  EXPECT_THROW_MSG(s.D.dyson_start(s.G, s.mu, s.H, s.S, s.I, s.h), state_error,
                   "h_nessi::dyson::dyson_start: G.tstpmk() == 0 && Sigma.tstpmk() == 0 violated (G.tstpmk()=1, Sigma.tstpmk()=0)");
  EXPECT_THROW(s.D.dyson_start_ntti(s.G, s.mu, s.H, s.S, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_start_2leg(s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_start_les(s.G, s.mu, s.H, s.S, s.I, s.h), state_error);
}

TEST(DysonEntry, GreenFromHTmaxAndWindow) {
  Sys s;
  ZMatrix h0 = s.H.get_map(0);
  herm_matrix_hodlr G(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  EXPECT_NO_THROW(s.D.green_from_H(G, s.mu, h0, s.h, s.k, true));
  // H11
  EXPECT_THROW_MSG(s.D.green_from_H(G, s.mu, h0, s.h, s.k + 1), precondition_error,
                   "0 <= tmax && tmax <= k_ && tmax < G.nt() violated (tmax=3, k_=2, G.nt()=16)");
  EXPECT_THROW(s.D.green_from_H(G, s.mu, s.H, s.h, s.nt - 1), precondition_error);
  ZMatrix rho = ZMatrix::Identity(2, 2) * 0.5;
  EXPECT_THROW(s.D.green_from_H_dm(G, s.mu, h0, rho, s.h, s.k + 1), precondition_error);
  G.update_blocks(s.I);
  EXPECT_THROW_MSG(s.D.green_from_H(G, s.mu, h0, s.h, s.k), state_error, "G.tstpmk() == 0 violated (G.tstpmk()=1)");
  // DY-F1, DY-F2 (now in the h_nessi message format, LX-23: nothing printed to cout)
  herm_matrix_hodlr Gb(s.nt, s.r, 2, 1e-12, 2, 2, +1, s.k);
  EXPECT_THROW_MSG(s.D.green_from_H(Gb, s.mu, h0, s.h, s.k), precondition_error, "G.sig() == xi_ violated (G.sig()=1, xi_=-1)");
  ZMatrix rho3 = ZMatrix::Identity(3, 3);
  herm_matrix_hodlr G2(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  EXPECT_THROW(s.D.green_from_H_dm(G2, s.mu, h0, rho3, s.h, s.k), precondition_error);
}

TEST(DysonEntry, DlrPairing) {
  Sys s;
  int r2 = 0;
  dlr_info other(r2, 40., 1e-6, s.beta, 2, -1);  // other rank
  ASSERT_NE(r2, s.r);
  DMatrix M(2, 2);
  EXPECT_THROW_MSG(s.G.get_mat_tau(0.5, other, M), precondition_error,
                   "h_nessi::herm_matrix_hodlr::get_mat_tau: dlr.r() == r_ && dlr.size1() == size1_ && dlr.size2() == size2_ violated");
  ZMatrix Z(2, 2);
  EXPECT_THROW(s.G.get_tv_tau(0, 0.5, other, Z), precondition_error);
  EXPECT_THROW(s.G.density_matrix(-1, other, M), precondition_error);
  std::vector<double> buf(s.r * 4 + r2 * 4);
  EXPECT_THROW(s.G.get_mat_reversed(other, buf.data()), precondition_error);
  int r1 = 0;
  dlr_info one(r1, 40., 1e-12, s.beta, 1, -1);  // other orbital size
  EXPECT_THROW(s.G.get_mat_tau(0.5, one, M), precondition_error);
  EXPECT_THROW(s.G.initGMConvTensor(one), precondition_error);
  EXPECT_NO_THROW(s.G.density_matrix(-1, s.dlr, M));
}

TEST(DysonEntry, ExtrapolateGeometry) {
  Sys s;
  s.G.update_blocks(s.I);
  Integration::Integrator I3(s.k + 1);
  EXPECT_THROW(s.D.extrapolate(s.G, I3), precondition_error);
  EXPECT_NO_THROW(s.D.extrapolate(s.G, s.I));
}

// ================================================================================================ L9, DY-P1
TEST(Finiteness, DysonMatNonFiniteThrowsAndLeavesG) {
  Sys s;
  std::vector<double> before(s.G.mat(), s.G.mat() + s.r * 4);
  herm_matrix_hodlr Sbad(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  for (int i = 0; i < s.r * 4; i++) Sbad.mat()[i] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW_MSG(s.D.dyson_mat(s.G, s.mu, s.H, Sbad), numerical_error, "returned a non-finite G^M; G is unchanged");
  for (int i = 0; i < s.r * 4; i++) EXPECT_EQ(s.G.mat()[i], before[i]);
}

TEST(Finiteness, DysonTimestepNonFiniteThrows) {
  Sys s;
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  ZMatrix nanm = ZMatrix::Constant(2, 2, C(std::nan(""), 0.));
  s.S.set_ret_curr(T, 0, nanm);
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), numerical_error,
                   "h_nessi::dyson::dyson_timestep: std::isfinite(err) violated");
}

// ================================================================================================ ST-2
TEST(ConvTensorState, StaleTensorAfterGMChangeIsRefused) {
  Sys s;
  EXPECT_TRUE(s.G.gmconv_current());
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  // a later write to G^M (here set_mat; the same for dyson_mat, map_mat, green_from_H(inc_mat), the MPI set)
  DMatrix m(2, 2);
  s.G.get_mat(0, m);
  m(0, 0) += 1e-3;
  s.G.set_mat(0, m);
  EXPECT_FALSE(s.G.gmconv_current());
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), state_error,
                   "G^M changed after the last initGMConvTensor");
  EXPECT_THROW(s.D.dyson_timestep_tv(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  std::vector<C> res(s.r * 4);
  EXPECT_THROW(s.D.tv_it_conv(T, s.S, s.G, res.data()), state_error);
  // _2leg does not use the tensor
  EXPECT_NO_THROW(s.D.dyson_timestep_2leg(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  s.G.initGMConvTensor(s.dlr);
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_NO_THROW(s.D.tv_it_conv(T, s.S, s.G, res.data()));
}

TEST(ConvTensorState, DysonMatAfterInitIsRefusedAtBootstrap) {
  Sys s;
  s.D.dyson_mat(s.G, s.mu + 0.1, s.H, s.S, false, 0.5);  // G^M changes (another mu)
  EXPECT_THROW_MSG(s.D.dyson_start(s.G, s.mu, s.H, s.S, s.I, s.h), state_error, "initGMConvTensor");
  EXPECT_THROW(s.D.dyson_start_tv(s.G, s.mu, s.H, s.S, s.I, s.h), state_error);
  EXPECT_NO_THROW(s.D.dyson_start_2leg(s.G, s.mu, s.H, s.S, s.I, s.h));  // does not use the tensor
  // a fresh object never had its tensor built
  herm_matrix_hodlr G2(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  EXPECT_FALSE(G2.gmconv_current());
  EXPECT_THROW_MSG(s.D.dyson_start(G2, s.mu, s.H, s.S, s.I, s.h), state_error, "gmconv_valid_ violated");
  s.G.initGMConvTensor(s.dlr);
  EXPECT_NO_THROW(s.D.dyson_start(s.G, s.mu, s.H, s.S, s.I, s.h));
}

namespace {
// write G to a checkpoint, optionally strip the WP3 format_version and the WP6 gmconv_valid (a pre-WP3 file), reload
herm_matrix_hodlr ckpt_roundtrip(herm_matrix_hodlr &G, const std::string &fn, bool pre_v2) {
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    G.write_checkpoint_hdf5(out, "G/");
    if (pre_v2) {
      out.unlink("G/format_version");
      out.unlink("G/gmconv_valid");
    }
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  ::testing::internal::CaptureStderr();
  herm_matrix_hodlr G2(in, "G/");
  ::testing::internal::GetCapturedStderr();
  return G2;
}
}  // namespace

TEST(ConvTensorState, CheckpointValidity) {
  for (int xi : {-1, +1}) {
    Sys s(16, 2, xi);
    herm_matrix_hodlr Gv2 = ckpt_roundtrip(s.G, "st2_v2_" + std::to_string(xi) + ".h5", false);
    EXPECT_TRUE(Gv2.gmconv_current()) << "xi=" << xi << ": a current checkpoint keeps its valid tensor";
    herm_matrix_hodlr Gv1 = ckpt_roundtrip(s.G, "st2_v1_" + std::to_string(xi) + ".h5", true);
    if (xi == -1) {
      EXPECT_TRUE(Gv1.gmconv_current()) << "fermionic tensors were not affected by the bosonic sign fix";
    } else {
      EXPECT_FALSE(Gv1.gmconv_current()) << "a pre-v2 bosonic tensor has the wrong sign: invalid";
      Gv1.update_blocks(s.I);
      herm_matrix_hodlr S1 = s.S;
      S1.update_blocks(s.I);
      EXPECT_THROW_MSG(s.D.dyson_timestep(s.k + 1, Gv1, s.mu, s.H, S1, s.I, s.h), state_error, "pre-v2 bosonic checkpoint");
      Gv1.initGMConvTensor(s.dlr);
      EXPECT_NO_THROW(s.D.dyson_timestep(s.k + 1, Gv1, s.mu, s.H, S1, s.I, s.h));
    }
    // a checkpoint written with a stale tensor stays stale
    DMatrix m(2, 2);
    s.G.get_mat(1, m);
    m(1, 1) *= 1.01;
    s.G.set_mat(1, m);
    herm_matrix_hodlr Gst = ckpt_roundtrip(s.G, "st2_stale_" + std::to_string(xi) + ".h5", false);
    EXPECT_FALSE(Gst.gmconv_current());
  }
}

// ================================================================================================ ST-3
TEST(SavedG0State, FixHamNeedsSavedG0) {
  int r = 0;
  dlr_info dlr(r, 40., 1e-12, 4., 2, -1);
  herm_matrix_hodlr G(8, r, 2, 1e-12, 2, 2, -1, 2), S(8, r, 2, 1e-12, 2, 2, -1, 2);
  function H(8, 2, 2);
  ZMatrix h0(2, 2);
  h0 << 0.3, 0.1, 0.1, -0.2;
  H.set_constant(h0);
  dyson D(8, 2, 2, dlr);
  EXPECT_FALSE(D.g0_valid());
  EXPECT_THROW_MSG(D.dyson_mat(G, 0.1, H, S, true), state_error, "!fixHam || g0_valid_ violated (fixHam=1, g0_valid_=0)");
  DMatrix hd = h0.real();
  EXPECT_THROW(D.dyson_mat(G, 0.1, hd, S, true), state_error);
  EXPECT_NO_THROW(D.dyson_mat(G, 0.1, H, S, false));
  EXPECT_TRUE(D.g0_valid());
  EXPECT_NO_THROW(D.dyson_mat(G, 0.1, H, S, true));
  // restored from a checkpoint: the G0 buffers are not checkpointed
  std::string fn = "st3_dyson.h5";
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    D.write_checkpoint_hdf5(out, "D");
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  dyson D2(in, "D", dlr);
  EXPECT_FALSE(D2.g0_valid());
  EXPECT_THROW(D2.dyson_mat(G, 0.1, H, S, true), state_error);
  // DY-M6
  EXPECT_THROW(D.dyson_mat(G, 0.1, H, S, false, 0.), precondition_error);
  EXPECT_THROW(D.dyson_mat(G, 0.1, H, S, false, 1.5), precondition_error);
}

// ================================================================================================ D-11 Hermiticity
TEST(Hermiticity, NonHermitianHThrowsBeforeModifyingIt) {
  Sys s;
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  ZMatrix bad = s.H.get_map(T);
  bad(0, 1) += 0.5;  // O(1) anti-Hermitian part
  s.H.get_map(T) = bad;
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error,
                   "h_nessi::dyson::dyson_timestep: rel <= herm_err_tol_ violated (rel=");
  EXPECT_EQ((s.H.get_map(T) - bad).norm(), 0.0) << "h must not be modified when the call throws";
  // the same for the bootstrap, dyson_mat and green_from_H
  Sys s2;
  ZMatrix b0 = s2.H.get_map(0);
  b0(1, 0) += C(0, 0.3);
  for (int t = 0; t <= s2.k; t++) s2.H.get_map(t) = b0;
  EXPECT_THROW(s2.D.dyson_start(s2.G, s2.mu, s2.H, s2.S, s2.I, s2.h), precondition_error);
  EXPECT_THROW(s2.D.green_from_H(s2.G, s2.mu, b0, s2.h, s2.k), precondition_error);
  s2.H.get_map(-1) = b0;
  EXPECT_THROW(s2.D.dyson_mat(s2.G, s2.mu, s2.H, s2.S), precondition_error);
  DMatrix bd(2, 2);
  bd << 0.3, 0.2, -0.2, -0.4;
  EXPECT_THROW(s2.D.dyson_mat(s2.G, s2.mu, bd, s2.S), precondition_error);
}

TEST(Hermiticity, SmallAntiHermitianPartIsSymmetrisedInPlaceAndRecorded) {
  Sys s;
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  ZMatrix hT = s.H.get_map(T);
  ZMatrix pert(2, 2);
  pert << C(0, 2e-7), C(1e-7, 3e-7), C(-1e-7, 1e-7), C(0, -1e-7);  // anti-Hermitian? no: generic small perturbation
  ZMatrix hp = hT + pert;
  s.H.get_map(T) = hp;
  ZMatrix herm = 0.5 * (hp + hp.adjoint());
  double rel = (hp - hp.adjoint()).norm() / hp.norm();
  ::testing::internal::CaptureStderr();
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  std::string err = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(err, "") << "below herm_warn_tol: no warning";
  ZMatrix after = s.H.get_map(T);
  EXPECT_EQ((after - after.adjoint()).norm(), 0.0) << "h(T) must be exactly Hermitian after the call";
  EXPECT_LT((after - herm).norm(), 1e-16);
  EXPECT_NEAR(s.D.antihermiticity_per_slice()[T + 1], rel, 1e-12 * rel + 1e-22);
  EXPECT_NEAR(s.D.last_antihermiticity(), rel, 1e-12 * rel + 1e-22);
  // symmetrised slices are not modified again (bitwise), and the next call records 0 for them
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_EQ((ZMatrix(s.H.get_map(T)) - after).norm(), 0.0);
  EXPECT_EQ(s.D.last_antihermiticity(), 0.0);
  // the per-slice record keeps the value found before symmetrisation
  EXPECT_NEAR(s.D.antihermiticity_per_slice()[T + 1], rel, 1e-12 * rel + 1e-22);
}

TEST(Hermiticity, OptOutChecksButLeavesHUntouched) {
  Sys s;
  s.D.set_hermitian_symmetrization(false);
  EXPECT_FALSE(s.D.hermitian_symmetrization());
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  ZMatrix hp = s.H.get_map(T);
  hp(0, 1) += C(3e-6, 1e-6);
  s.H.get_map(T) = hp;
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_EQ((ZMatrix(s.H.get_map(T)) - hp).norm(), 0.0) << "opt-out: h is not modified";
  EXPECT_GT(s.D.last_antihermiticity(), 0.0) << "opt-out still checks and records";
  // it still throws above the error tolerance
  hp(0, 1) += 1.0;
  s.H.get_map(T) = hp;
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error);
  // DMatrix dyson_mat overload: opt-out leaves the real matrix untouched as well
  DMatrix hd(2, 2);
  hd << 0.3, 0.1 + 1e-7, 0.1, -0.4;
  DMatrix hd0 = hd;
  EXPECT_NO_THROW(s.D.dyson_mat(s.G, s.mu, hd, s.S));
  EXPECT_EQ((hd - hd0).norm(), 0.0);
  s.D.set_hermitian_symmetrization(true);
  EXPECT_NO_THROW(s.D.dyson_mat(s.G, s.mu, hd, s.S));
  EXPECT_EQ(hd(0, 1), hd(1, 0));
}

TEST(Hermiticity, WarnsOnceAndTolerancesAreSettable) {
  Sys s;
  int T = s.k + 1;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  ZMatrix hp = s.H.get_map(T);
  hp(1, 0) += 1e-3;  // rel ~ 1e-3: above warn (1e-4), below err (1e-2)
  s.H.get_map(T) = hp;
  ::testing::internal::CaptureStderr();
  s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h);
  std::string err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("h_nessi warning: h_nessi::dyson::dyson_timestep: h(t) has a relative anti-Hermitian part"),
            std::string::npos) << err;
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  hp = s.H.get_map(T + 1);
  hp(1, 0) += 1e-3;
  s.H.get_map(T + 1) = hp;
  ::testing::internal::CaptureStderr();
  s.D.dyson_timestep(T + 1, s.G, s.mu, s.H, s.S, s.I, s.h);
  err = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(err, "") << "warned once per dyson object";
  // tighter error tolerance: the same size now throws
  s.D.set_hermiticity_tolerances(1e-6, 1e-5);
  EXPECT_EQ(s.D.hermiticity_err_tol(), 1e-5);
  s.G.update_blocks(s.I);
  s.S.update_blocks(s.I);
  hp = s.H.get_map(T + 2);
  hp(1, 0) += 1e-3;
  s.H.get_map(T + 2) = hp;
  EXPECT_THROW(s.D.dyson_timestep(T + 2, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error);
  EXPECT_THROW(s.D.set_hermiticity_tolerances(1e-2, 1e-4), precondition_error);
  EXPECT_THROW(s.D.set_hermiticity_tolerances(-1., 1e-4), precondition_error);
}

// D1 (decided): a complex h(-1) cannot be represented by the real G^M -> dyson_mat throws
TEST(Hermiticity, ComplexHMinusOneThrowsInDysonMat) {
  Sys s;
  ZMatrix hc(2, 2);
  hc << 0.3, C(0.2, 0.1), C(0.2, -0.1), -0.4;  // Hermitian but complex
  s.H.get_map(-1) = hc;
  EXPECT_THROW_MSG(s.D.dyson_mat(s.G, s.mu, s.H, s.S), precondition_error, "h(-1) must be real");
  ZMatrix hr = hc.real().cast<C>();
  hr(0, 1) += C(0, 1e-15);  // round-off imaginary parts are accepted (Hermitian part)
  hr(1, 0) -= C(0, 1e-15);
  s.H.get_map(-1) = hr;
  EXPECT_NO_THROW(s.D.dyson_mat(s.G, s.mu, s.H, s.S));
}

// D-11 end to end: a RHO_HORIZONTAL run with a mean-field h(T) = h0(T) + g rho(T) built from the propagated G^<(T,T).
// HORIZONTAL leaves G^<(T,T) anti-Hermitian only up to the discretisation error, so h(T) inherits a small
// anti-Hermitian part. The run must pass the checks (no warning, no throw), h must be Hermitian in place afterwards,
// the per-slice diagnostic must hold the anti-Hermitian part, and the result must differ from a check-only run by no
// more than the size of that part (times a modest factor).
namespace {
struct MFRun {
  std::vector<ZMatrix> rho, hT;
  std::vector<double> diag;
  double maxanti_abs = 0;
};
MFRun mf_run(bool symmetrize, double gmf) {
  const int nt = 40, k = 3, nao = 2;
  int r = 0;
  double beta = 4., mu = 0.1, h = 0.05;
  dlr_info dlr(r, 40., 1e-12, beta, nao, -1);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, 2, 1e-12, nao, nao, -1, k), S(nt, r, 2, 1e-12, nao, nao, -1, k);
  function H(nt, nao, nao);
  dyson D(nt, nao, k, dlr, false /*RHO_HORIZONTAL*/, false);
  D.set_hermitian_symmetrization(symmetrize);
  ZMatrix h0(2, 2);
  h0 << 0.3, 0.25, 0.25, -0.4;
  auto h0t = [&](int t) {  // C^infinity switch-on of a real drive at ton = 0.5 > k*h (TTI bootstrap)
    ZMatrix x = h0;
    double tt = std::max(t, 0) * h, ton = 0.5;
    if (tt > ton) x(0, 1) = x(1, 0) = 0.25 + 0.3 * std::exp(-1.0 / (tt - ton)) * std::sin(2. * tt);
    return x;
  };
  for (int t = -1; t < nt; t++) H.get_map(t) = h0t(t);
  // Sigma: a fixed bath of two levels (exact free G of a bath hamiltonian), Sigma = V^2 g_bath
  herm_matrix_hodlr gb(nt, r, 2, 1e-12, nao, nao, -1, k);
  function Hb(nt, nao, nao);
  ZMatrix hb(2, 2);
  hb << 0.8, 0.0, 0.0, -0.6;
  Hb.set_constant(hb);
  dyson Db(nt, nao, k, dlr, false, false);
  herm_matrix_hodlr zero(nt, r, 2, 1e-12, nao, nao, -1, k);
  Db.green_from_H(gb, mu, hb, h, k, true);
  gb.initGMConvTensor(dlr);
  const double V2 = 0.3;
  auto copy_row = [&](int T) {  // S row T = V2 * gb row T
    ZMatrix m(nao, nao);
    for (int tp = 0; tp <= T; tp++) {
      gb.get_ret_curr(T, tp, m);
      S.set_ret_curr(T, tp, ZMatrix(V2 * m));
      gb.get_les_curr(tp, T, m);
      S.set_les_curr(tp, T, ZMatrix(V2 * m));
    }
    for (int j = 0; j < r; j++) {
      gb.get_tv(T, j, m);
      S.set_tv(T, j, ZMatrix(V2 * m));
    }
  };
  for (int j = 0; j < r; j++) {
    DMatrix m(nao, nao);
    gb.get_mat(j, m);
    S.set_mat(j, DMatrix(V2 * m));
  }
  for (int T = 0; T <= k; T++) copy_row(T);
  for (int T = 0; T <= k; T++)  // continuation of Sigma^R into the upper triangle of the window (TTI)
    for (int tp = T + 1; tp <= k; tp++) {
      ZMatrix m(nao, nao);
      gb.get_ret_curr(tp, T, m);
      S.set_ret_curr(T, tp, ZMatrix(-V2 * m.adjoint()));
    }
  // self-consistent mean-field equilibrium: h(-1) = h0 + g rho(-1), rho(-1) = -G^M(beta)
  DMatrix rhoM = DMatrix::Zero(nao, nao);
  for (int it = 0; it < 200; it++) {
    H.get_map(-1) = (h0.real() + gmf * rhoM).cast<C>();
    D.dyson_mat(G, mu, H, S, false, 0.7);
    DMatrix rnew(nao, nao);
    G.density_matrix(-1, dlr, rnew);
    double d = (rnew - rhoM).norm();
    rhoM = rnew;
    if (d < 1e-13) break;
  }
  for (int t = 0; t <= k; t++) H.get_map(t) = H.get_map(-1);  // TTI bootstrap in the mean-field equilibrium
  G.initGMConvTensor(dlr);
  D.dyson_start(G, mu, H, S, I, h);
  MFRun out;
  ZMatrix rho(nao, nao);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    S.update_blocks(I);
    gb.update_blocks(I);
    zero.update_blocks(I);
    Db.dyson_timestep(T, gb, mu, Hb, zero, I, h);  // free bath
    copy_row(T);
    D.extrapolate(G, I);
    for (int it = 0; it < 3; it++) {  // predictor-corrector on the mean field
      G.density_matrix(T, dlr, rho);
      H.get_map(T) = h0t(T) + gmf * rho;  // complex: rho(T) from the propagated G^<(T,T)
      ZMatrix hh = H.get_map(T);
      out.maxanti_abs = std::max(out.maxanti_abs, (hh - hh.adjoint()).norm());
      D.dyson_timestep(T, G, mu, H, S, I, h);
    }
    G.density_matrix(T, dlr, rho);
    out.rho.push_back(rho);
    out.hT.push_back(H.get_map(T));
  }
  out.diag = D.antihermiticity_per_slice();
  return out;
}
}  // namespace

TEST(Hermiticity, HorizontalMeanFieldRunPasses) {
  ::testing::internal::CaptureStderr();
  MFRun sym = mf_run(true, 0.8);
  std::string err = ::testing::internal::GetCapturedStderr();
  EXPECT_EQ(err.find("h_nessi warning"), std::string::npos) << err;
  MFRun raw = mf_run(false, 0.8);
  double maxdiag = 0, maxrho_anti = 0, drho = 0;
  for (double d : sym.diag) maxdiag = std::max(maxdiag, d);
  for (size_t i = 0; i < sym.hT.size(); i++) {
    EXPECT_EQ((sym.hT[i] - sym.hT[i].adjoint()).norm(), 0.0) << "h symmetrised in place, slice " << i;
    maxrho_anti = std::max(maxrho_anti, (raw.rho[i] - raw.rho[i].adjoint()).norm());
    drho = std::max(drho, (sym.rho[i] - raw.rho[i]).norm());
  }
  std::printf("HORIZONTAL mean-field run: max rel anti-Hermitian part of h = %.3e (abs %.3e), "
              "max ||rho-rho^dag|| = %.3e, max |rho_sym - rho_raw| = %.3e\n",
              maxdiag, raw.maxanti_abs, maxrho_anti, drho);
  EXPECT_GT(maxdiag, 0.0) << "HORIZONTAL must leave a (small) anti-Hermitian part in the mean field";
  EXPECT_LT(maxdiag, 1e-4) << "it must stay below the warning tolerance";
  EXPECT_LE(drho, 10. * raw.maxanti_abs + 1e-14) << "symmetrisation changes the result only at the size of the anti-Hermitian part";
}
