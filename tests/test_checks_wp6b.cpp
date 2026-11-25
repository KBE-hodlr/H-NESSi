// WP6b: the remaining precondition checks of agent_docs/61_precondition_audit.md (tier 2: throw tests; tier 1: death
// tests in builds with asserts). Grouped by check family:
//  * herm_matrix_hodlr: set_tstpmk, time2block failure path, DLR evaluators (tau range, row range, output sizes),
//    density_matrix, update_blocks (HM-U1/U2), checkpoint consistency (HM-C11), output writers (HM-W2..W5),
//    distances (HM-D*), tier-1 accessor asserts (HM-A*, HM-R*, HM-S*, BK-*).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <future>
#include <thread>
#include <unistd.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "common/test_util.hpp"
#include "h_nessi/bubble.hpp"
#include "h_nessi/checks.hpp"
#include "h_nessi/dlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/integration.hpp"
#include "h_nessi/read_inputfile.hpp"

using namespace h_nessi;
using C = std::complex<double>;

namespace {

// expects `stmt` to throw type E (or a subclass) with a message containing `sub`
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

#ifndef NDEBUG
#define HN_ASSERTS_ON 1
#else
#define HN_ASSERTS_ON 0
#endif

// Fills rows 0..T of a herm_matrix_hodlr with smooth data, calling update_blocks once per step (window [T-k, T]).
void fill_rows(herm_matrix_hodlr &G, Integration::Integrator &I, int T) {
  int n = G.size1(), k = G.k();
  auto row = [&](int t) {
    for (int tp = 0; tp <= t; tp++) {
      ZMatrix m(n, n);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) m(i, j) = C(std::cos(0.1 * t - 0.2 * tp + i), std::sin(0.05 * (t + tp) + j));
      G.set_ret_curr(t, tp, m);
      G.set_les_curr(tp, t, m);
    }
    for (int a = 0; a < G.r(); a++) {
      ZMatrix m = ZMatrix::Constant(n, n, C(0.01 * t, a));
      G.set_tv(t, a, m);
    }
  };
  for (int t = 0; t <= std::min(k, T); t++) row(t);
  for (int t = k + 1; t <= T; t++) {
    G.update_blocks(I);
    row(t);
  }
}

}  // namespace

// ============================================================================================ herm_matrix_hodlr
TEST(HodlrChecks, SetTstpmkRange) {
  herm_matrix_hodlr G(20, 3, 2, 1e-10, 1, 1, -1, 2);
  EXPECT_THROW_MSG(G.set_tstpmk(-1), precondition_error, "herm_matrix_hodlr::set_tstpmk");
  EXPECT_THROW_MSG(G.set_tstpmk(21), precondition_error, "tstpmk=21");
  EXPECT_NO_THROW(G.set_tstpmk(20));
  EXPECT_NO_THROW(G.set_tstpmk(0));
}

// HM-C21 / LX-12: a pair outside every block is an error in every build (tier 1 asserts the range first).
TEST(HodlrChecksDeathTest, Time2BlockFailurePath) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  herm_matrix_hodlr G(32, 3, 2, 1e-10, 1, 1, -1, 2);
  EXPECT_GE(G.time2block(31, 0), 0);
#if HN_ASSERTS_ON
  EXPECT_DEATH(G.time2block(5, 5), "herm_matrix_hodlr::time2block");  // on the diagonal: direct region
#else
  EXPECT_THROW_MSG(G.time2block(5, 5), precondition_error, "not inside a compressed block");
#endif
}

TEST(HodlrChecks, DlrEvaluatorsTauRowAndSizes) {
  int nt = 20, k = 2, r = 0, nao = 2;
  double beta = 5.;
  dlr_info dlr(r, 40., 1e-10, beta, nao, -1);
  herm_matrix_hodlr G(nt, r, 2, 1e-10, nao, nao, -1, k);
  DMatrix M(nao, nao), Mbad(1, 1);
  ZMatrix Z(nao, nao), Zbad(3, 1);
  // HM-I1: tau in [0, beta]; a few ulps above beta are accepted
  EXPECT_THROW_MSG(G.get_mat_tau(-0.1, dlr, M), precondition_error, "tau=-0.1");
  EXPECT_THROW_MSG(G.get_mat_tau(1.01 * beta, dlr, M.data()), precondition_error, "[0, beta]");
  EXPECT_NO_THROW(G.get_mat_tau(beta * (1 + 1e-15), dlr, M));
  EXPECT_THROW_MSG(G.get_mat_tau(0.5, dlr, Mbad), precondition_error, "M.rows()=1");
  EXPECT_THROW(G.get_mat_tau(0.5, dlr, Zbad), precondition_error);
  // HM-I2: the row must be computed (window [0,k] on a fresh object: rows 0..k)
  EXPECT_NO_THROW(G.get_tv_tau(k, 0.5, dlr, Z));
  EXPECT_THROW_MSG(G.get_tv_tau(k + 1, 0.5, dlr, Z), precondition_error, "herm_matrix_hodlr::get_tv_tau");
  EXPECT_THROW(G.get_tv_tau(-1, 0.5, dlr, Z.data()), precondition_error);
  EXPECT_THROW(G.get_tv_tau(0, 0.5, dlr, Zbad), precondition_error);
  // HM-I3: every tau of the array, output size
  DColVector taus(3);
  taus << 0., 0.5 * beta, beta;
  DMatrix A(3, nao * nao), Asmall(2, nao * nao);
  ZMatrix ZA(3, nao * nao);
  EXPECT_NO_THROW(G.get_mat_tau_array(taus, dlr, A));
  EXPECT_THROW_MSG(G.get_mat_tau_array(taus, dlr, Asmall), precondition_error, "not resized");
  EXPECT_NO_THROW(G.get_tv_tau_array(1, taus, dlr, ZA));
  EXPECT_THROW(G.get_tv_tau_array(k + 1, taus, dlr, ZA), precondition_error);
  taus(1) = -1.;
  EXPECT_THROW(G.get_mat_tau_array(taus, dlr, A), precondition_error);
  EXPECT_THROW(G.get_tv_tau_array(0, taus, dlr, ZA.data()), precondition_error);
  // HM-I4..I6: output sizes, (i,j), rows
  DMatrix R(r, nao * nao), Rsmall(r - 1, nao * nao), R1(r, 1);
  ZMatrix ZR(r, nao * nao), ZR1(r, 1);
  EXPECT_NO_THROW(G.get_mat_reversed(dlr, R));
  EXPECT_THROW(G.get_mat_reversed(dlr, Rsmall), precondition_error);
  EXPECT_NO_THROW(G.get_mat_reversed(dlr, R1, 1, 0));
  EXPECT_THROW_MSG(G.get_mat_reversed(dlr, R1, 2, 0), precondition_error, "i=2");
  EXPECT_THROW(G.get_mat_reversed(dlr, R1, 0, -1), precondition_error);
  EXPECT_NO_THROW(G.get_tv_reversed(k, dlr, ZR));
  EXPECT_THROW(G.get_tv_reversed(k + 1, dlr, ZR), precondition_error);
  EXPECT_THROW(G.get_tv_reversed(0, dlr, ZR1, 0, 2), precondition_error);
  EXPECT_NO_THROW(G.get_vt(0, dlr, ZR1, 1, 1));
  EXPECT_THROW(G.get_vt(k + 1, dlr, ZR.data()), precondition_error);
  EXPECT_THROW(G.get_vt(0, dlr, ZR1), precondition_error);  // r x 1 is too small for all orbitals
}

TEST(HodlrChecks, DensityMatrixRow) {
  int nt = 20, k = 2, r = 0, nao = 1;
  dlr_info dlr(r, 40., 1e-10, 5., nao, -1);
  herm_matrix_hodlr G(nt, r, 2, 1e-10, nao, nao, -1, k);
  DMatrix M(1, 1);
  ZMatrix Z(1, 1), Zbad(2, 2);
  EXPECT_NO_THROW(G.density_matrix(-1, dlr, M));
  EXPECT_NO_THROW(G.density_matrix(k, dlr, Z));
  EXPECT_THROW_MSG(G.density_matrix(k + 1, dlr, Z), precondition_error, "herm_matrix_hodlr::density_matrix");
  EXPECT_THROW(G.density_matrix(-2, dlr, M), precondition_error);
  EXPECT_THROW(G.density_matrix(0, dlr, Zbad), precondition_error);
}

TEST(HodlrChecks, UpdateBlocksOrderAndIntegrator) {
  int nt = 12, k = 2;
  Integration::Integrator I(k), I3(3);
  herm_matrix_hodlr G(nt, 2, 2, 1e-10, 1, 1, -1, k);
  EXPECT_THROW_MSG(G.update_blocks(I3), precondition_error, "I.k()=3");
  fill_rows(G, I, nt - 1);
  for (int i = 0; i <= k; i++) G.update_blocks(I);  // the k+1 final calls
  EXPECT_EQ(G.tstpmk(), nt);
  EXPECT_THROW_MSG(G.update_blocks(I), state_error, "update_blocks is called once per time step");
}

namespace {
std::string tmpfile(const std::string &tag) { return "wp6b_" + tag + "_" + std::to_string(getpid()) + ".h5"; }
}  // namespace

// HM-C11: the checkpoint constructor refuses datasets that do not match the stored geometry.
TEST(HodlrChecks, CheckpointDatasetSizes) {
  int nt = 16, k = 2;
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, 2, 2, 1e-10, 2, 2, -1, k);
  fill_rows(G, I, 9);
  auto write = [&](const std::string &fn, const std::function<void(h5e::File &)> &corrupt) {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    G.write_checkpoint_hdf5(out, "G");
    corrupt(out);
  };
  std::string fn = tmpfile("ckpt");
  write(fn, [](h5e::File &) {});
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_NO_THROW(herm_matrix_hodlr H(in, "G"));
  }
  write(fn, [](h5e::File &out) {
    out.unlink("G/blklen");
    h5e::dump(out, "G/blklen", std::vector<int>(2, 0));
  });
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_THROW_MSG(herm_matrix_hodlr H(in, "G"), precondition_error, "blklen");
  }
  write(fn, [](h5e::File &out) {
    out.unlink("G/tv");
    h5e::dump(out, "G/tv", std::vector<C>(5));
  });
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_THROW_MSG(herm_matrix_hodlr H(in, "G"), precondition_error, "name=tv");
  }
  write(fn, [](h5e::File &out) {
    out.unlink("G/ret/0_0_1/epsrank");
    h5e::dump(out, "G/ret/0_0_1/epsrank", 7);
  });
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_THROW_MSG(herm_matrix_hodlr H(in, "G"), precondition_error, "epsrank=7");
  }
  write(fn, [](h5e::File &out) {
    out.unlink("G/built_blocks");
    h5e::dump(out, "G/built_blocks", 9);
  });
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_THROW(herm_matrix_hodlr H(in, "G"), precondition_error);
  }
  std::remove(fn.c_str());
}

// HM-W2 (warning), HM-W3, HM-W4, HM-W5
TEST(HodlrChecks, OutputWriters) {
  int nt = 16, k = 2, r = 0;
  dlr_info dlr(r, 40., 1e-10, 5., 1, -1);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, 2, 1e-10, 1, 1, -1, k), G2(nt + 4, r, 2, 1e-10, 1, 1, -1, k);
  fill_rows(G, I, 9);
  std::string fn = tmpfile("out");
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    ::testing::internal::CaptureStderr();
    G.write_to_hdf5(out, "G");
    std::string err = ::testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("final k+1 update_blocks calls have not been made"), std::string::npos) << err;
    // a second object with another geometry under the same geometry label
    EXPECT_THROW_MSG(G2.write_to_hdf5(out, "G2"), precondition_error, "geometry_label");
    EXPECT_NO_THROW(G2.write_to_hdf5(out, "G3", "other"));
    EXPECT_THROW_MSG(G.write_rho_to_hdf5(out, "rho", dlr), precondition_error, "all rows 0..nt-1");
    EXPECT_THROW(G.write_GR0_to_hdf5(out, "G"), precondition_error);
    EXPECT_THROW(G.write_GL0_to_hdf5(out, "G"), precondition_error);
  }
  for (int t = 10; t < nt; t++) G.update_blocks(I);  // rows 10..nt-1 left as stored data; window [nt-1-k, nt-1]
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    EXPECT_NO_THROW(G.write_rho_to_hdf5(out, "rho", dlr));
    EXPECT_NO_THROW(G.write_GR0_to_hdf5(out, "G"));
    for (int i = 0; i <= k; i++) G.update_blocks(I);
    ::testing::internal::CaptureStderr();
    G.write_to_hdf5(out, "Gfinal");
    EXPECT_EQ(::testing::internal::GetCapturedStderr(), "");
  }
  std::remove(fn.c_str());
}

// HM-D1..D4, LX-20 (WP-N side finding 7)
TEST(HodlrChecks, Distances) {
  int nt = 16, k = 2;
  Integration::Integrator I(k);
  herm_matrix_hodlr A(nt, 3, 2, 1e-10, 2, 2, -1, k), B(nt, 3, 2, 1e-10, 2, 2, -1, k), Cs(nt, 3, 2, 1e-10, 1, 1, -1, k),
      Dr(nt, 4, 2, 1e-10, 2, 2, -1, k);
  fill_rows(A, I, 8);  // window [6,8]
  fill_rows(B, I, 8);
  EXPECT_EQ(distance_norm2_curr(8, A, B), 0.0);
  EXPECT_EQ(distance_norm2_curr(-1, A, B), 0.0);
  EXPECT_THROW_MSG(distance_norm2_curr_ret(8, A, Cs), precondition_error, "g2.size1()=1");
  EXPECT_THROW(distance_norm2_curr_les(8, A, Cs), precondition_error);
  EXPECT_THROW_MSG(distance_norm2_curr_ret(5, A, B), precondition_error, "current windows");
  EXPECT_THROW(distance_norm2_curr_les(9, A, B), precondition_error);
  EXPECT_THROW(distance_norm2_mat(A, Dr), precondition_error);
  EXPECT_THROW(distance_norm2_mat(A, Cs), precondition_error);
  EXPECT_THROW(distance_norm2_tv(0, A, Dr), precondition_error);
  EXPECT_THROW_MSG(distance_norm2_tv(nt, A, B), precondition_error, "tstp=16");
  EXPECT_THROW(distance_norm2_tv(-1, A, B), precondition_error);
  EXPECT_THROW(distance_norm2_curr(-2, A, B), precondition_error);
  EXPECT_THROW(distance_norm2_curr(5, A, B), precondition_error);
}

// Tier-1 accessor and pointer checks (HM-A*, HM-R*, HM-S*, BK-*): abort with the message in builds with asserts.
TEST(HodlrChecksDeathTest, Tier1Accessors) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#if HN_ASSERTS_ON
  int nt = 16, k = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, 2, 1e-10, 2, 2, -1, k);
  fill_rows(G, I, 8);  // window [6,8]
  ZMatrix M(2, 2), Mbad(1, 2);
  DMatrix D(2, 2), Dbad(3, 3);
  EXPECT_DEATH(G.get_ret(9, 0, M), "herm_matrix_hodlr::get_ret");           // HM-R1/LX-6: row above tstp()
  EXPECT_DEATH(G.get_les(0, -1, M), "herm_matrix_hodlr::get_les");          // HM-R6
  EXPECT_DEATH(G.get_ret_curr(5, 4, M), "herm_matrix_hodlr::get_ret_curr");  // row 5 below the window
  EXPECT_DEATH(G.get_les_curr(9, 2, M), "herm_matrix_hodlr::get_les_curr");  // LX-8: column 9 above the window
  EXPECT_DEATH(G.map_ret_curr(7, 8), "herm_matrix_hodlr::map_ret_curr");     // LX-10: t2 > max(t1,k)
  EXPECT_DEATH(G.map_les_curr(8, 7), "herm_matrix_hodlr::map_les_curr");     // LX-10: t1 > max(t2,k)
  EXPECT_DEATH(G.tvptr(nt, 0), "herm_matrix_hodlr::tvptr");                  // LX-3
  EXPECT_DEATH(G.tvptr(0, r), "herm_matrix_hodlr::tvptr");                   // HM-A3: tau index (was tvptr_trans)
  EXPECT_DEATH(G.tv_offset(-1, 0), "herm_matrix_hodlr::tv_offset");          // LX-4
  EXPECT_DEATH(G.curr_timestep_ret_ptr(5, 0), "curr_timestep_ret_ptr");      // LX-9: row below the window
  EXPECT_DEATH(G.curr_timestep_les_ptr(nt + 1, 8), "curr_timestep_les_ptr");
  EXPECT_DEATH(G.retptr_corr(8, 0), "herm_matrix_hodlr::retptr_corr");       // HM-A7
  EXPECT_DEATH(G.matptr(r), "herm_matrix_hodlr::matptr");
  EXPECT_DEATH(G.map_mat(-1), "herm_matrix_hodlr::map_mat");                 // HM-A2
  EXPECT_DEATH(G.set_mat(0, Dbad), "herm_matrix_hodlr::set_mat");            // HM-S4/LX-26
  EXPECT_DEATH(G.get_mat(0, Dbad), "herm_matrix_hodlr::get_mat");            // HM-R12
  EXPECT_DEATH(G.set_tv(0, 0, Mbad), "herm_matrix_hodlr::set_tv");           // HM-S6/LX-26
  EXPECT_DEATH(G.set_ret_curr(8, 0, Mbad), "herm_matrix_hodlr::set_ret_curr");
  EXPECT_DEATH(G.set_les_curr(7, 6, M), "herm_matrix_hodlr::set_les_curr");  // HM-S9: t1 > t2
  EXPECT_DEATH(G.get_tv(nt, 0, M), "herm_matrix_hodlr::get_tv");             // LX-5
  EXPECT_DEATH(G.set_tstp_zero(5), "herm_matrix_hodlr::set_tstp_zero");      // HM-S1
  EXPECT_DEATH(G.smul_curr(nt, 1.0), "herm_matrix_hodlr::smul_curr");        // HM-T6/LX-11
  EXPECT_DEATH(G.blkr2(G.nbox()), "herm_matrix_hodlr::blkr2");               // HM-C16
  EXPECT_DEATH(G.c1_dir(nt), "herm_matrix_hodlr::c1_dir");                   // HM-C18
  EXPECT_DEATH(G.time2direct_col(8, 0), "herm_matrix_hodlr::time2direct_col");  // HM-C19
  EXPECT_DEATH(G.ret().get_direct_col(-1, M), "ret_blocks::get_direct_col");  // LX-18
  EXPECT_DEATH(G.ret().get_compress(0, 0, G.nbox(), M), "blocks_list::get_compress");  // BK-5
  block b(1, 4, 1, 1e-10);
  EXPECT_DEATH(b.set_rank(-1), "block::set_rank");                           // BK-2
  // the valid neighbours of the above do not abort
  G.get_ret(8, 0, M);
  G.get_ret_curr(6, 8, M);
  G.get_les_curr(8, 2, M);
  G.get_ret(3, 7, M);
#else
  GTEST_SKIP() << "asserts are off (NDEBUG): the accessor checks are tier 1";
#endif
}

// ============================================================================================ dyson
// Test access to the ST-10 owner of dyson / dlr_info (friend structs declared in the classes).
namespace h_nessi {
struct dyson_test_access {
  static detail::single_thread_owner &owner(dyson &d) { return d.owner_; }
};
struct dlr_info_test_access {
  static detail::single_thread_owner &owner(dlr_info &d) { return d.owner_; }
};
}  // namespace h_nessi

namespace {
const C II(0, 1);

// Small working system (as in test_checks.cpp): nao = 2, Sigma = 0 (TTI), constant Hermitian h, after dyson_mat ->
// initGMConvTensor -> dyson_start, i.e. ready for dyson_timestep(k+1).
struct Sys {
  int nt, nao = 2, k, nlvl = 2, xi, r = 0;
  double beta = 4., mu = 0.2, h = 0.05;
  dlr_info dlr;
  Integration::Integrator I;
  herm_matrix_hodlr G, S;
  function H;
  dyson D;
  explicit Sys(int nt_ = 16, int k_ = 2, int xi_ = -1)
      : nt(nt_), k(k_), xi(xi_), dlr(r, 40., 1e-12, beta, 2, xi_), I(k_), G(nt_, r, 2, 1e-12, 2, 2, xi_, k_),
        S(nt_, r, 2, 1e-12, 2, 2, xi_, k_), H(nt_, 2, 2), D(nt_, 2, k_, dlr, true, false) {
    if (xi == 1) mu = -1.5;
    ZMatrix h0(2, 2);
    h0 << 0.3, C(0.2, 0.1), C(0.2, -0.1), -0.4;
    for (int t = -1; t < nt; t++) H.get_map(t) = h0;
    H.get_map(-1) = h0.real().cast<C>();
    for (int t = 0; t <= k; t++) H.get_map(t) = H.get_map(-1);
    D.dyson_mat(G, mu, H, S, false, 1.0);
    G.initGMConvTensor(dlr);
    D.dyson_start(G, mu, H, S, I, h);
  }
  void advance() {
    G.update_blocks(I);
    S.update_blocks(I);
  }
  // h(t) with a relative anti-Hermitian part ~ eps
  ZMatrix perturbed(int t, double eps) {
    ZMatrix hp = H.get_map(t);
    hp(1, 0) += eps * hp.norm();
    return hp;
  }
};
double antiherm(const ZMatrix &m) { return (m - m.adjoint()).norm() / m.norm(); }
}  // namespace

// WP6a side finding 1, maintainer decision (a): the Hermiticity tolerances are enforced on the h(T) used by the
// corrector solves, not on the h(T) built from the extrapolated predictor.
TEST(HermiticityCorrector, PredictorValueIsNotEnforcedWhenTheCorrectorRewritesH) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  ZMatrix hp = s.perturbed(T, 0.05);  // predictor h(T): rel ~ 5e-2 > herm_err_tol (1e-2)
  double relp = antiherm(hp);
  ASSERT_GT(relp, s.D.hermiticity_err_tol());
  s.H.get_map(T) = hp;
  ::testing::internal::CaptureStderr();
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_EQ(antiherm(s.H.get_map(T)), 0.0) << "the predictor slice is still symmetrised in place";
  EXPECT_NEAR(s.D.antihermiticity_per_slice()[T + 1], relp, 1e-12);
  // corrector: h(T) rebuilt (e.g. from the corrected rho) with a small anti-Hermitian part
  ZMatrix hc = s.perturbed(T, 1e-6);
  s.H.get_map(T) = hc;
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_EQ(::testing::internal::GetCapturedStderr(), "") << "no warning: the predictor value is not enforced";
  EXPECT_NEAR(s.D.antihermiticity_per_slice()[T + 1], antiherm(hc), 1e-12)
      << "the diagnostic holds the value of the corrector h(T)";
  // the next step: slice T is an old slice now (unchanged): nothing pending, no error
  s.advance();
  EXPECT_NO_THROW(s.D.dyson_timestep(T + 1, s.G, s.mu, s.H, s.S, s.I, s.h));
}

TEST(HermiticityCorrector, CorrectorValueIsEnforcedImmediately) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));  // predictor
  ZMatrix bad = s.perturbed(T, 0.05);
  s.H.get_map(T) = bad;
  EXPECT_THROW_MSG(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error,
                   "rel <= herm_err_tol_ violated");
  EXPECT_EQ((ZMatrix(s.H.get_map(T)) - bad).norm(), 0.0) << "not modified when the call throws";
}

TEST(HermiticityCorrector, UnchangedPredictorValueIsEnforcedAtTheNextStep) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  s.H.get_map(T) = s.perturbed(T, 0.05);
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_NO_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h)) << "h(T) not rewritten: still deferred";
  s.advance();
  // the solves of step T used the (symmetrised) predictor h(T) unchanged: its value is enforced now, naming t = T
  EXPECT_THROW_MSG(s.D.dyson_timestep(T + 1, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error,
                   "t=" + std::to_string(T));
}

TEST(HermiticityCorrector, OlderSlicesAndComponentCalls) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  // an older slice rewritten with an O(1) anti-Hermitian part throws in the first solve of the step
  s.H.get_map(T - 1) = s.perturbed(T - 1, 0.5);
  EXPECT_THROW(s.D.dyson_timestep(T, s.G, s.mu, s.H, s.S, s.I, s.h), precondition_error);
  s.H.get_map(T - 1) = s.H.get_map(T);  // repair
  // component calls of one solve: tv measures slice T first (deferred), les finds it unchanged
  Sys s2;
  s2.advance();
  s2.H.get_map(T) = s2.perturbed(T, 0.05);
  EXPECT_NO_THROW(s2.D.dyson_timestep_ret(T, s2.G, s2.mu, s2.H.ptr(0), s2.S, s2.I, s2.h));
  EXPECT_NO_THROW(s2.D.dyson_timestep_tv(T, s2.G, s2.mu, s2.H.ptr(0), s2.S, s2.I, s2.h));
  EXPECT_NO_THROW(s2.D.dyson_timestep_les(T, s2.G, s2.mu, s2.H.ptr(0), s2.S, s2.I, s2.h));
}

// ST-4 (DY-T7, DY-S6): order of the solver components
TEST(ComponentOrder, TimestepAndBootstrap) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  EXPECT_THROW_MSG(s.D.dyson_timestep_les(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error,
                   "call dyson_timestep_ret and dyson_timestep_tv");
  EXPECT_THROW(s.D.dyson_timestep_les_2leg(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  s.D.dyson_timestep_ret(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h);
  EXPECT_THROW(s.D.dyson_timestep_les(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_timestep_les_nobc(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  s.D.dyson_timestep_tv(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h);
  EXPECT_NO_THROW(s.D.dyson_timestep_les(T, s.G, s.mu, s.H.ptr(0), s.S, s.I, s.h));
  // the record belongs to (G, T): another G starts empty
  herm_matrix_hodlr G2 = s.G;
  EXPECT_THROW(s.D.dyson_timestep_les(T, G2, s.mu, s.H.ptr(0), s.S, s.I, s.h), state_error);
  EXPECT_NO_THROW(s.D.dyson_timestep(T, G2, s.mu, s.H, s.S, s.I, s.h));
  // bootstrap: tv before ret/les (TTI), tv and ret_ntti before les_ntti, ret_ntti before les_2leg
  herm_matrix_hodlr B(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k), Sb(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  s.D.dyson_mat(B, s.mu, s.H, Sb, false, 1.0);
  B.initGMConvTensor(s.dlr);
  EXPECT_THROW_MSG(s.D.dyson_start_ret(B, s.mu, s.H, Sb, s.I, s.h), state_error, "call dyson_start_tv");
  EXPECT_THROW(s.D.dyson_start_les(B, s.mu, s.H, Sb, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_start_les_ntti(B, s.mu, s.H, Sb, s.I, s.h), state_error);
  EXPECT_THROW(s.D.dyson_start_les_2leg(B, s.mu, s.H, Sb, s.I, s.h), state_error);
  s.D.dyson_start_ret_ntti(B, s.mu, s.H, Sb, s.I, s.h);
  EXPECT_THROW(s.D.dyson_start_les_ntti(B, s.mu, s.H, Sb, s.I, s.h), state_error) << "tv is still missing";
  s.D.dyson_start_tv(B, s.mu, s.H, Sb, s.I, s.h);
  EXPECT_NO_THROW(s.D.dyson_start_les_ntti(B, s.mu, s.H, Sb, s.I, s.h));
  EXPECT_NO_THROW(s.D.dyson_start_ret(B, s.mu, s.H, Sb, s.I, s.h));
}

// ST-10 (DY-Z2, DL-10): one dyson / dlr_info per thread
TEST(SingleThreadOwner, ConcurrentUseIsDetected) {
  Sys s;
  auto hold_in_other_thread = [](detail::single_thread_owner &o, const std::function<void()> &body) {
    std::promise<void> held, done;
    std::thread th([&] {
      o.acquire("test");
      held.set_value();
      done.get_future().wait();
      o.release();
    });
    held.get_future().wait();
    body();
    done.set_value();
    th.join();
  };
  hold_in_other_thread(dyson_test_access::owner(s.D), [&] {
    EXPECT_THROW_MSG(s.D.dyson_mat(s.G, s.mu, s.H, s.S), state_error, "use one object per thread");
    EXPECT_THROW(s.D.extrapolate(s.G, s.I), state_error);
  });
  EXPECT_FALSE(dyson_test_access::owner(s.D).held());
  EXPECT_NO_THROW(s.D.dyson_mat(s.G, s.mu, s.H, s.S));  // free again; nested public calls are fine
  DMatrix M(2, 2);
  hold_in_other_thread(dlr_info_test_access::owner(s.dlr), [&] {
    EXPECT_THROW(s.G.get_mat_tau(0.5, s.dlr, M), state_error);
  });
  EXPECT_NO_THROW(s.G.get_mat_tau(0.5, s.dlr, M));
  // copies start free
  detail::single_thread_owner a;
  EXPECT_TRUE(a.acquire("x"));
  EXPECT_FALSE(a.acquire("x")) << "nested acquire on the owning thread";
  detail::single_thread_owner b(a);
  EXPECT_FALSE(b.held());
  a.release();
}

// DL-8, DL-9: eval_point tau range
TEST(DlrChecks, EvalPointTauRange) {
  int r = 0;
  dlr_info dlr(r, 40., 1e-10, 2., 1, -1);
  std::vector<double> g(r, 1.0);
  double out;
  EXPECT_THROW_MSG(dlr.eval_point(-1e-3, g.data(), &out), precondition_error, "dlr_info::eval_point");
  EXPECT_THROW(dlr.eval_point(2.1, g.data(), &out), precondition_error);
  EXPECT_NO_THROW(dlr.eval_point(2.0, g.data(), &out));
  DColVector taus(2);
  taus << 0.5, 3.0;
  std::vector<double> outv(2);
  EXPECT_THROW(dlr.eval_point(taus, g.data(), outv.data()), precondition_error);
}

// ST-11 (D-9, DY-C3): the dyson checkpoint stores xi, r, beta, eps, lambda and verifies them on restart
TEST(DysonCheckpoint, VerifiesDlrParameters) {
  int r = 0, r2 = 0, r3 = 0;
  dlr_info dlr(r, 40., 1e-12, 4., 2, -1), dlr_beta(r2, 40., 1e-12, 4.5, 2, -1), dlr_xi(r3, 40., 1e-12, 4., 2, +1);
  dyson D(8, 2, 2, dlr);
  std::string fn = tmpfile("dyson");
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    D.write_checkpoint_hdf5(out, "D");
  }
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_NO_THROW(dyson D2(in, "D", dlr));
    EXPECT_THROW_MSG(dyson D3(in, "D", dlr_beta), state_error, "cbeta=4");
    EXPECT_THROW(dyson D4(in, "D", dlr_xi), state_error);
  }
  {  // a checkpoint written before WP6b: not verified, warning
    h5e::File out(fn, h5e::File::ReadWrite);
    out.unlink("D/xi");
  }
  {
    h5e::File in(fn, h5e::File::ReadOnly);
    ::testing::internal::CaptureStderr();
    EXPECT_NO_THROW(dyson D5(in, "D", dlr_beta));
    EXPECT_NE(::testing::internal::GetCapturedStderr().find("not verified"), std::string::npos);
  }
  std::remove(fn.c_str());
}

// DY-M10, DY-F10: bosons need h - mu > 0; DY-F9: complex h with G^M; DY-F7: rho is a density matrix
TEST(FreeAndMatChecks, BosonicSpectrumRealityAndRho) {
  Sys b(16, 2, +1);  // bosonic system with mu = -1.5 below the spectrum
  ZMatrix h0 = b.H.get_map(-1);
  function Hlow = b.H;
  EXPECT_THROW_MSG(b.D.dyson_mat(b.G, 0.0, b.H, b.S), precondition_error, "every eigenvalue of h - mu must be positive");
  herm_matrix_hodlr Gb(b.nt, b.r, 2, 1e-12, 2, 2, +1, b.k);
  EXPECT_THROW(b.D.green_from_H(Gb, 0.0, h0, b.h, b.k, true), precondition_error);
  ZMatrix rho = ZMatrix::Identity(2, 2) * 0.1;
  EXPECT_THROW(b.D.green_from_H_dm(Gb, 0.0, h0, rho, b.h, b.k), precondition_error);
  EXPECT_NO_THROW(b.D.green_from_H(Gb, b.mu, h0, b.h, b.k, true));
  // DY-F9
  Sys f;
  herm_matrix_hodlr G(f.nt, f.r, 2, 1e-12, 2, 2, -1, f.k);
  ZMatrix hc(2, 2);
  hc << 0.3, C(0.2, 0.1), C(0.2, -0.1), -0.4;
  EXPECT_THROW_MSG(f.D.green_from_H(G, f.mu, hc, f.h, f.k, true), precondition_error, "h(-1) must be real");
  EXPECT_NO_THROW(f.D.green_from_H(G, f.mu, hc, f.h, f.k, false));
  function Hc(f.nt, 2, 2);
  Hc.set_constant(hc);
  EXPECT_THROW(f.D.green_from_H(G, f.mu, Hc, f.h, f.k, true), precondition_error);
  std::vector<double> g0(f.r * 4);
  EXPECT_THROW(f.D.green_from_H_mat(g0.data(), f.mu, hc), precondition_error);
  // DY-F7: rho Hermitian, occupations in [0,1] for fermions
  ZMatrix rok(2, 2), rnh(2, 2), rneg(2, 2);
  rok << 0.6, C(0.1, 0.05), C(0.1, -0.05), 0.3;
  rnh << 0.6, 0.3, -0.3, 0.3;
  rneg = -rok;  // G^M(0^+)-like sign error
  EXPECT_NO_THROW(f.D.green_from_H_dm(G, f.mu, hc, rok, f.h, f.k));
  EXPECT_THROW_MSG(f.D.green_from_H_dm(G, f.mu, hc, rnh, f.h, f.k), precondition_error, "Hermitian) density matrix");
  EXPECT_THROW_MSG(f.D.green_from_H_dm(G, f.mu, hc, rneg, f.h, f.k), precondition_error, "occ_min=");
  ZMatrix rbig = 1.5 * ZMatrix::Identity(2, 2);
  EXPECT_THROW(f.D.green_from_H_dm(G, f.mu, hc, rbig, f.h, f.k), precondition_error);
}

// DY-S4, LX-19: relative TTI tolerance with values in the message
TEST(BootstrapChecks, TTIRelativeTolerance) {
  Sys s;
  herm_matrix_hodlr G(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k), S(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  s.D.dyson_mat(G, s.mu, s.H, S, false, 1.0);
  G.initGMConvTensor(s.dlr);
  // a large TTI Sigma^R(t,t') = A exp(-i (t-t')) with relative round-off: accepted
  double scale = 1e6;
  for (int t = 0; t <= s.k; t++)
    for (int tp = 0; tp <= t; tp++) {
      ZMatrix m = scale * std::exp(-II * double(t - tp) * s.h) * ZMatrix::Identity(2, 2);
      m(0, 0) *= 1 + 1e-15 * (t + tp);
      S.set_ret_curr(t, tp, m);
    }
  EXPECT_NO_THROW(s.D.dyson_start(G, s.mu, s.H, S, s.I, s.h));
  // a real violation reports t, tp, diff and tol
  ZMatrix m = ZMatrix::Identity(2, 2) * scale * 1.001;
  S.set_ret_curr(s.k, 1, m);  // compared with Sigma^R(k-1,0)
  EXPECT_THROW_MSG(s.D.dyson_start(G, s.mu, s.H, S, s.I, s.h), precondition_error, "tol=");
  // h not TTI
  S.set_ret_curr(s.k, 1, ZMatrix(scale * std::exp(-II * double(s.k - 1) * s.h) * ZMatrix::Identity(2, 2)));
  function H2 = s.H;
  H2.get_map(1) = s.perturbed(1, 0.0) * 1.01;
  EXPECT_THROW_MSG(s.D.dyson_start(G, s.mu, H2, S, s.I, s.h), precondition_error, "H is not TTI");
}

// DY-S7 (G^<(0,0) set by the caller for _nobc/_2leg), DY-S10 (bootstrap linear systems)
TEST(BootstrapChecks, Les00AndSingularSystems) {
  Sys s;
  herm_matrix_hodlr G(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k), S(s.nt, s.r, 2, 1e-12, 2, 2, -1, s.k);
  s.D.dyson_mat(G, s.mu, s.H, S, false, 1.0);
  G.initGMConvTensor(s.dlr);
  ::testing::internal::CaptureStderr();
  EXPECT_NO_THROW(s.D.dyson_start_ntti_nobc(G, s.mu, s.H, S, s.I, s.h, false));
  EXPECT_NE(::testing::internal::GetCapturedStderr().find("G^<(0,0) is zero"), std::string::npos);
  ZMatrix bad(2, 2);
  bad << C(0, 0.6), 0.4, 0.4, C(0, 0.3);  // not anti-Hermitian
  G.set_les_curr(0, 0, bad);
  EXPECT_THROW_MSG(s.D.dyson_start_2leg(G, s.mu, s.H, S, s.I, s.h, false), precondition_error,
                   "must be anti-Hermitian");
  ZMatrix good(2, 2);
  good << C(0, 0.6), C(0.1, 0.05), C(-0.1, 0.05), C(0, 0.3);  // i * Hermitian
  G.set_les_curr(0, 0, good);
  EXPECT_NO_THROW(s.D.dyson_start_2leg(G, s.mu, s.H, S, s.I, s.h, false));
  // a non-finite Sigma makes the bootstrap systems non-invertible: numerical_error
  ZMatrix nanm = ZMatrix::Constant(2, 2, C(std::numeric_limits<double>::quiet_NaN(), 0));
  S.set_ret_curr(1, 0, nanm);
  EXPECT_THROW(s.D.dyson_start_ntti(G, s.mu, s.H, S, s.I, s.h, false), numerical_error);
}

// DY-G1..G3 (gamma_integral), DY-I5 (tv_ret_int), DY-I6 (les_lesadv_int_0_tstp)
TEST(IntegralChecks, GammaAndInternalIntegrals) {
  Sys s;
  int k = s.k;
  std::vector<C> res(4);
  Integration::Integrator I3(3);
  EXPECT_NO_THROW(s.D.gamma_integral(0, s.G, s.G, s.h, s.I, res.data()));
  EXPECT_NO_THROW(s.D.gamma_integral_mat(s.G, s.G, s.h, s.I, res.data()));
  herm_matrix_hodlr Other(s.nt + 2, s.r, 2, 1e-12, 2, 2, -1, k);
  EXPECT_THROW_MSG(s.D.gamma_integral(0, s.G, Other, s.h, s.I, res.data()), precondition_error, "same geometry");
  EXPECT_THROW(s.D.gamma_integral_mat(s.G, Other, s.h, s.I, res.data()), precondition_error);
  EXPECT_THROW(s.D.gamma_integral(0, s.G, s.S, s.h, I3, res.data()), precondition_error);
  EXPECT_THROW(s.D.gamma_integral(k + 1, s.G, s.S, s.h, s.I, res.data()), state_error);  // row k+1 not in the window
  s.advance();  // window [1, k+1]
  EXPECT_THROW_MSG(s.D.gamma_integral(1, s.G, s.S, s.h, s.I, res.data()), state_error, "bootstrap windows");
  EXPECT_NO_THROW(s.D.dyson_timestep(k + 1, s.G, s.mu, s.H, s.S, s.I, s.h));
  EXPECT_NO_THROW(s.D.gamma_integral(k + 1, s.G, s.S, s.h, s.I, res.data()));
  // tv_ret_int: Sigma row tstp in its window
  EXPECT_THROW(s.D.tv_ret_int(k + 2, s.S, s.G, s.I, s.h), state_error);
  EXPECT_THROW(s.D.tv_ret_int(0, s.S, s.G, s.I, s.h), precondition_error);
  // les_lesadv_int_0_tstp: G and Sigma at the same step
  herm_matrix_hodlr S2 = s.S;
  S2.update_blocks(s.I);
  EXPECT_THROW_MSG(s.D.les_lesadv_int_0_tstp(k + 1, S2, s.G, s.h, s.I), state_error, "same step");
  EXPECT_THROW(s.D.les_lesadv_int_0_tstp(k + 2, s.S, s.G, s.h, s.I), state_error);
  EXPECT_NO_THROW(s.D.les_lesadv_int_0_tstp(k + 1, s.S, s.G, s.h, s.I));
}

// ST-6 (DY-X3): a write into row tstp() before extrapolate makes extrapolate a no-op (it would read the overwritten
// slot of row tstp()-k-1)
TEST(ExtrapolateState, WriteIntoRowTBeforeExtrapolate) {
  Sys s;
  int T = s.k + 1;
  s.advance();
  EXPECT_TRUE(s.G.can_extrap());
  ZMatrix m = ZMatrix::Constant(2, 2, C(0.25, -0.5));
  s.G.set_ret_curr(T, 0, m);
  EXPECT_FALSE(s.G.can_extrap());
  s.D.extrapolate(s.G, s.I);
  ZMatrix after(2, 2);
  s.G.get_ret_curr(T, 0, after);
  EXPECT_EQ((after - m).norm(), 0.0) << "extrapolate is a no-op after the write";
  // other mutators of row T
  Sys s2;
  s2.advance();
  s2.G.set_tstp_zero(T);
  EXPECT_FALSE(s2.G.can_extrap());
  Sys s3;
  s3.advance();
  s3.G.smul_curr(T, 2.0);
  EXPECT_FALSE(s3.G.can_extrap());
  Sys s4;
  s4.advance();
  s4.G.set_ret_curr(T - 1, 0, m);  // row T-1: no effect
  EXPECT_TRUE(s4.G.can_extrap());
}

// IN-2: the weights of every supported order pass the post-condition
TEST(IntegratorChecks, WeightsOfEveryOrder) {
  for (int k = 0; k <= KMAX; k++) EXPECT_NO_THROW(Integration::Integrator I(k)) << "k=" << k;
}

TEST(IntegratorChecksDeathTest, Tier1Accessors) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#if HN_ASSERTS_ON
  Integration::Integrator I(3);
  EXPECT_DEATH(I.poly_interp(-1, 0), "Integrator::poly_interp");  // LX-16
  EXPECT_DEATH(I.gregory_weights(5, 6), "Integrator::gregory_weights");  // IN-7: j > max(n,k)
  EXPECT_DEATH(I.gregory_weights(1, -1), "Integrator::gregory_weights");
  I.gregory_weights(1, 3);  // start weights: j <= k is valid for n <= k
  EXPECT_DEATH(I.bd_weights(5), "Integrator::bd_weights");
  EXPECT_DEATH(I.ex_weights(-1), "Integrator::ex_weights");
  int r = 0;
  dlr_info dlr(r, 40., 1e-10, 2., 1, -1);
  EXPECT_DEATH(dlr.it0B(r), "dlr_info::it0B");  // DL-11
#else
  GTEST_SKIP() << "asserts are off (NDEBUG): the accessor checks are tier 1";
#endif
}

// ============================================================================================ function, Bubble
TEST(FunctionChecks, ConstructionSizesAndCheckpoint) {
  EXPECT_THROW_MSG(function(10, 2, 3), precondition_error, "square orbital matrices");
  EXPECT_THROW(function(-1, 2, 2), precondition_error);
  EXPECT_THROW(function(10, 0, 0), precondition_error);
  function f(10, 2, 2);
  ZMatrix bad(3, 3);
  EXPECT_THROW_MSG(f.set_constant(bad), precondition_error, "function::set_constant");
  ZMatrix ok = ZMatrix::Identity(2, 2);
  f.set_constant(ok);
  std::string fn = tmpfile("func");
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    f.write_checkpoint_hdf5(out, "f");
    f.write_checkpoint_hdf5(out, "g");
    out.unlink("g/data");
    h5e::dump(out, "g/data", ZMatrix(ZMatrix::Zero(5, 1)));
    f.write_checkpoint_hdf5(out, "e");
    out.unlink("e/element_size");
    h5e::dump(out, "e/element_size", 3);
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  EXPECT_NO_THROW(function f2(in, "f"));
  EXPECT_THROW_MSG(function f3(in, "g"), precondition_error, "(nt+1)*size1*size2");
  EXPECT_THROW(function f4(in, "e"), precondition_error);
  std::remove(fn.c_str());
}

TEST(FunctionChecksDeathTest, Tier1Accessors) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#if HN_ASSERTS_ON
  function f(10, 2, 2);
  ZMatrix bad(3, 3);
  EXPECT_DEATH(f.ptr(10), "function::ptr");      // FN-2
  EXPECT_DEATH(f.ptr(-2), "function::ptr");
  EXPECT_DEATH(f(0, 2, 0), "function::operator");  // FN-3
  EXPECT_DEATH(f.value(0, 0, -1), "function::value");
  EXPECT_DEATH(f.set_value(0, bad), "function::set_value");  // FN-4
  function g(std::move(f));
  EXPECT_DEATH(f.ptr(0), "function::ptr");       // FN-7: moved-from
  f.size1();
#else
  GTEST_SKIP() << "asserts are off (NDEBUG): the accessor checks are tier 1";
#endif
}

TEST(BubbleChecks, AliasingIndicesRankAndWindow) {
  int nt = 12, k = 2;
  Integration::Integrator I(k);
  int r = 0;
  dlr_info dlr(r, 40., 1e-10, 2., 2, -1);
  herm_matrix_hodlr A(nt, r, 2, 1e-10, 2, 2, -1, k), B(nt, r, 2, 1e-10, 2, 2, -1, k), C(nt, r, 2, 1e-10, 2, 2, +1, k),
      Cr(nt, r + 1, 2, 1e-10, 2, 2, +1, k);
  EXPECT_NO_THROW(Bubble1(1, C, 0, 1, A, 1, 0, B, 0, 0, dlr));
  EXPECT_NO_THROW(Bubble2(-1, C, 0, 1, A, 1, 0, B, 0, 0));
  EXPECT_THROW_MSG(Bubble2(1, A, 0, 1, A, 1, 0, B, 0, 0), precondition_error, "must not alias");
  EXPECT_THROW(Bubble1(1, B, 0, 1, A, 1, 0, B, 0, 0, dlr), precondition_error);
  EXPECT_THROW_MSG(Bubble2(1, C, 0, 2, A, 1, 0, B, 0, 0), precondition_error, "c2=2");
  EXPECT_THROW(Bubble1(1, C, 0, 0, A, -1, 0, B, 0, 0, dlr), precondition_error);
  EXPECT_THROW(Bubble2(1, Cr, 0, 0, A, 0, 0, B, 0, 0), precondition_error);
  EXPECT_THROW_MSG(Bubble2(k + 1, C, 0, 0, A, 0, 0, B, 0, 0), precondition_error, "current windows");
  A.update_blocks(I);
  EXPECT_THROW(Bubble2(0, C, 0, 0, A, 0, 0, B, 0, 0), precondition_error);  // row 0 left A's window
}

// ============================================================================================ read_inputfile
TEST(ReadInputfileChecks, WholeTokenCountsAndFiles) {
  std::string fn = tmpfile("inp") + ".inp";
  {
    FILE *f = std::fopen(fn.c_str(), "w");
    std::fprintf(f, "__n_float= 1.5\n__n_ok= 7  \n__x_bad= 2x\n__b_bad= truex\n__v= 1 2 3\n__f= --%s.missing\n",
                 fn.c_str());
    std::fclose(f);
  }
  int n;
  double x;
  bool b;
  try {
    find_param((char *)fn.c_str(), "__n_float=", n);  // RI-3: "1.5" used to give 1
    ADD_FAILURE() << "no exception";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("trailing characters"), std::string::npos) << e.what();
  }
  EXPECT_NO_THROW(find_param((char *)fn.c_str(), "__n_ok=", n));
  EXPECT_EQ(n, 7);
  EXPECT_THROW(find_param((char *)fn.c_str(), "__x_bad=", x), std::runtime_error);
  EXPECT_THROW(find_param((char *)fn.c_str(), "__b_bad=", b), std::runtime_error);
  // RI-4: dim >= 0, nt >= -1
  std::vector<double> v;
  DColVector dv;
  EXPECT_THROW(find_param((char *)fn.c_str(), "__v=", v, -1), std::runtime_error);
  EXPECT_THROW(find_param((char *)fn.c_str(), "__v=", dv, -2), std::runtime_error);
  EXPECT_NO_THROW(find_param((char *)fn.c_str(), "__v=", v, 3));
  char line[] = "1.0 2.0";
  std::vector<double> tv;
  EXPECT_THROW_MSG(read_param_tvector(line, tv, -2), std::runtime_error, "nt = -2");
  // RI-6: a --FILE argument that cannot be read throws with the file name
  char fline[512];
  std::snprintf(fline, sizeof fline, "--%s.missing", fn.c_str());
  EXPECT_THROW_MSG(read_param_tvector(fline, tv, 4), std::runtime_error, ".missing");
  std::remove(fn.c_str());
}
