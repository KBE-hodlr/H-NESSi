// HODLR storage (docs 10-12): rank-1 TSVD update, fill / archive / read-back round trips, checkpoint I/O,
// copy semantics (B22, B6: get/set_timestep_curr, operator=, copy constructor), geometry edge cases (B23), memory report (B26), checkpoint timing default (B9).
// Sources: the former tests/linalg.cpp (updated to the h_nessi API), agent_scratch/52/s9.cpp (zero first rows),
// agent_scratch/52/s1.cpp (B22).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <malloc.h>
#include <map>
#include <string>
#include <unistd.h>
#include <hdf5.h>

#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/utils.hpp"
#include "common/test_util.hpp"

using namespace h_nessi;
namespace {
const cplx II(0, 1);
}

// mgs and updatetsvd are private since WP6b (doc 61 Q-12); the class grants this struct access for the unit tests.
namespace h_nessi {
struct herm_matrix_hodlr_test_access {
  static double mgs(herm_matrix_hodlr &G, const ZMatrix &V, const ZRowVector &b, const ZRowVector &x,
                    const ZRowVector &Vb) {
    return G.mgs(V, b, x, Vb);
  }
  static void updatetsvd(herm_matrix_hodlr &G, block &B, const ZRowVector &row, int row_cur) {
    G.updatetsvd(B, row, row_cur);
  }
};
}  // namespace h_nessi
using TA = h_nessi::herm_matrix_hodlr_test_access;

// ------------------------------------------------------------------------------------------------ linear algebra
TEST(LinearAlgebra, MgsOrthogonalises) {
  int rows = 15, cols = 10, nlvl = 3, sig = -1, kt = 5, r = 10, size = 1;
  double epsilon = 1e-4;
  ZMatrix R = ZMatrix::Random(rows, cols), U, V;
  DColVector Stmp;
  svd(R, U, V, Stmp);
  int rank = 4;
  ZMatrix Vnew = V.block(0, 0, cols, rank);
  ZRowVector x = ZRowVector::Random(cols), y(cols), Vb(cols);
  herm_matrix_hodlr G(cols, r, nlvl, epsilon, size, size, sig, kt);
  double nrm = TA::mgs(G, Vnew, x, y, Vb);
  double err = 0;
  for (int i = 0; i < rank; i++) err += std::abs(Vnew.col(i).dot(y.transpose()));
  EXPECT_LT(err, 1e-12) << "q must be orthogonal to the basis";
  EXPECT_NEAR(y.norm(), 1.0, 1e-12) << "q must be normalised";
  EXPECT_GT(nrm, 0.0);
}

// Incremental rank-1 TSVD (updatetsvd) of a matrix with geometric singular values reproduces the matrix and the
// eps-rank (the update is exact up to the truncation; P7: accumulated error within a few eps).
TEST(LinearAlgebra, IncrementalTsvdReproducesMatrix) {
  int rows = 50, cols = 50, nlvl = 3, sig = -1, kt = 5, r = 10, size = 1, ssize = 10;
  double epsilon = 1e-5;
  ZMatrix S = ZMatrix::Zero(ssize, ssize);
  for (int i = 0; i < ssize; i++) S(i, i) = std::exp(-i * 3.0);
  ZMatrix R = ZMatrix::Random(rows, cols), U1, V1;
  DColVector Stmp;
  svd(R, U1, V1, Stmp);
  ZMatrix M = U1.block(0, 0, rows, ssize) * S * V1.block(0, 0, cols, ssize).adjoint();
  block b(1, cols, 1, epsilon);
  ZRowVector y = M.row(0);
  b.U() = ZMatrix::Ones(1, 1);
  b.S() = y.norm() * DMatrix::Ones(1, 1);
  b.V() = y.adjoint() / y.norm();
  b.set_rank(1);
  herm_matrix_hodlr G(rows, r, nlvl, epsilon, size, size, sig, kt);
  for (int i = 1; i < rows; i++) TA::updatetsvd(G, b, M.row(i), i);
  DColVector S1;
  svd(M, S1);
  EXPECT_EQ(b.epsrank(), (S1.array() >= epsilon).count());
  ZMatrix rec = b.U() * b.S().asDiagonal() * b.V().adjoint();
  EXPECT_LT((M - rec).norm(), 5 * epsilon);
}

// ------------------------------------------------------------------------------------------------ round trips
// Fill G^R and G^< with smooth analytic data step by step (as a time propagation does), flush with the final
// k+1 update_blocks, and read every (t,t') back through get_ret / get_les.
namespace {
struct RoundTrip { double eR, eL; int nblocks_zero_start; };
RoundTrip roundtrip(int nt, int nlvl, int k, int ton, double tol) {
  double hs = 0.05;
  auto f = [&](int t, int tp) -> cplx {
    if (t < ton) return 0.;
    double d = (t - tp) * hs;
    return std::exp(-II * 1.3 * d) * 0.7 + 0.3 * std::exp(-II * (0.4 * d + 0.2 * sin(t * hs) - 0.2 * sin(tp * hs)));
  };
  auto gl = [&](int t, int tp) -> cplx {
    if (tp < ton) return 0.;
    double d = (t - tp) * hs;
    return II * 0.4 * std::exp(-II * 0.9 * d) + 0.1 * II * std::exp(-II * 2.1 * d);
  };
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, 2, nlvl, tol, 1, 1, -1, k);
  auto fill = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix Y(1, 1);
      Y(0, 0) = f(T, tp);
      G.set_ret_curr(T, tp, Y);
      Y(0, 0) = gl(tp, T);
      G.set_les_curr(tp, T, Y);
    }
  };
  for (int T = 0; T <= k; T++) fill(T);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    fill(T);
  }
  for (int i = 0; i <= k; i++) G.update_blocks(I);
  RoundTrip res{0, 0, 0};
  ZMatrix M(1, 1);
  for (int t = 0; t < nt; t++)
    for (int tp = 0; tp <= t; tp++) {
      G.get_ret(t, tp, M);
      res.eR = std::max(res.eR, std::abs(M(0, 0) - f(t, tp)));
      G.get_les(tp, t, M);
      res.eL = std::max(res.eL, std::abs(M(0, 0) - gl(tp, t)));
    }
  for (int b = 0; b < G.nbox(); b++)
    if (G.blkr1(b) < ton && G.blkr2(b) >= ton) res.nblocks_zero_start++;
  return res;
}
}  // namespace

class HodlrRoundTrip : public ::testing::TestWithParam<std::tuple<int, int, int>> {};
TEST_P(HodlrRoundTrip, FillArchiveReadBack) {
  auto [nt, nlvl, k] = GetParam();
  RoundTrip r = roundtrip(nt, nlvl, k, 0, 1e-12);
  EXPECT_LT(r.eR, 1e-10);
  EXPECT_LT(r.eL, 1e-10);
}
INSTANTIATE_TEST_SUITE_P(Storage, HodlrRoundTrip,
                         ::testing::Values(std::make_tuple(37, 3, 2), std::make_tuple(40, 3, 3), std::make_tuple(40, 4, 5),
                                           std::make_tuple(33, 5, 3), std::make_tuple(40, 3, 1), std::make_tuple(200, 3, 3),
                                           std::make_tuple(161, 4, 5), std::make_tuple(24, 1, 2)));

// Former ISSUES S9 (false positive, kept as regression): blocks whose first rows are identically zero.
TEST(HodlrStorage, ZeroFirstRowsCompressCorrectly) {
  for (int ton : {60, 130}) {
    RoundTrip r = roundtrip(200, 3, 3, ton, 1e-10);
    EXPECT_GT(r.nblocks_zero_start, 0) << "geometry must contain a block that starts with zero rows";
    EXPECT_LT(r.eR, 1e-12) << "ton=" << ton;
    EXPECT_LT(r.eL, 1e-12) << "ton=" << ton;
  }
}

// Checkpoint write / read round trip in the middle of a propagation.
TEST(HodlrStorage, CheckpointRoundTrip) {
  int nt = 40, nlvl = 3, k = 3, nao = 2, r = 5;
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, 1e-12, nao, nao, -1, k);
  auto val = [](int a, int b, int i, int j) { return cplx(0.3 * a - 0.1 * b + i, 0.2 * b + 0.05 * a * j) * std::exp(-0.01 * (a - b)); };
  auto fill = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix Y(nao, nao), L(nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) {
          Y(i, j) = val(T, tp, i, j);
          L(i, j) = val(tp, T, j, i) * II;
        }
      G.set_ret_curr(T, tp, Y);
      G.set_les_curr(tp, T, L);
    }
    for (int m = 0; m < r; m++) {
      ZMatrix X(nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) X(i, j) = cplx(T + m, i - j);
      G.set_tv(T, m, X);
    }
  };
  for (int T = 0; T <= k; T++) fill(T);
  int Tstop = 25;
  for (int T = k + 1; T <= Tstop; T++) {
    G.update_blocks(I);
    fill(T);
  }
  std::string fn = "hodlr_checkpoint_roundtrip.h5";
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    G.write_checkpoint_hdf5(out, "G/");
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  herm_matrix_hodlr G2(in, "G/");
  EXPECT_EQ(G2.tstpmk(), G.tstpmk());
  EXPECT_EQ(G2.built_blocks(), G.built_blocks());
  ZMatrix A(nao, nao), B(nao, nao);
  double e = 0;
  for (int t = 0; t <= Tstop; t++)
    for (int tp = 0; tp <= t; tp++) {
      G.get_ret(t, tp, A);
      G2.get_ret(t, tp, B);
      e = std::max(e, (A - B).norm());
      G.get_les(tp, t, A);
      G2.get_les(tp, t, B);
      e = std::max(e, (A - B).norm());
    }
  for (int t = 0; t <= Tstop; t++)
    for (int m = 0; m < r; m++) {
      G.get_tv(t, m, A);
      G2.get_tv(t, m, B);
      e = std::max(e, (A - B).norm());
    }
  EXPECT_EQ(e, 0.0);
}

// ------------------------------------------------------------------------------------------------ B22, B6
// Copy semantics (WP4).  get_timestep_curr(G) copies THIS object's current window into G (G.tstpmk := this->tstpmk)
// and leaves the source untouched (agent_scratch/52/s1.cpp); the tstp == -1 (Matsubara) copy has no window
// requirement.  operator= / the copy constructor are a deep copy of the complete state (B6).
// Full-state comparisons use the checkpoint dump: every dataset written by write_checkpoint_hdf5, compared bitwise.
namespace {
typedef std::map<std::string, std::vector<char>> HodlrState;

herr_t collect_ds(hid_t, const char *name, const H5O_info2_t *info, void *op) {
  if (info->type == H5O_TYPE_DATASET) ((std::vector<std::string> *)op)->push_back(name);
  return 0;
}

// Every dataset of G's checkpoint as raw native bytes, keyed by its path.
HodlrState snapshot(herm_matrix_hodlr &G, const std::string &tag) {
  std::string fn = "hodlr_state_" + tag + "_" + std::to_string(getpid()) + ".h5";
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    G.write_checkpoint_hdf5(out, "G/");
  }
  HodlrState st;
  hid_t f = H5Fopen(fn.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  std::vector<std::string> names;
  H5Ovisit3(f, H5_INDEX_NAME, H5_ITER_INC, collect_ds, &names, H5O_INFO_BASIC);
  for (auto &n : names) {
    hid_t d = H5Dopen2(f, n.c_str(), H5P_DEFAULT), t = H5Dget_type(d), nt = H5Tget_native_type(t, H5T_DIR_ASCEND);
    hid_t sp = H5Dget_space(d);
    std::vector<char> buf(H5Tget_size(nt) * H5Sget_simple_extent_npoints(sp));
    if (!buf.empty()) H5Dread(d, nt, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    st[n] = buf;
    H5Sclose(sp);
    H5Tclose(nt);
    H5Tclose(t);
    H5Dclose(d);
  }
  H5Fclose(f);
  std::remove(fn.c_str());
  return st;
}

// Bitwise equality of two snapshots; `timing` (wall-clock durations of update_blocks) may be skipped.
void expect_same_state(const HodlrState &a, const HodlrState &b, const std::string &what, bool skip_timing = false) {
  EXPECT_GT(a.size(), 20u) << what << ": checkpoint has too few datasets";
  EXPECT_EQ(a.size(), b.size()) << what;
  int ncmp = 0;
  for (auto &kv : a) {
    if (skip_timing && kv.first.find("timing") != std::string::npos) continue;
    auto it = b.find(kv.first);
    ASSERT_TRUE(it != b.end()) << what << ": dataset " << kv.first << " missing";
    EXPECT_TRUE(kv.second == it->second) << what << ": dataset " << kv.first << " differs";
    ncmp++;
  }
  EXPECT_GT(ncmp, 20) << what;
}

// Matrix-valued (nao = 2) test data for R, <, tv and mat, step by step as a propagation writes it.
struct CopyFill {
  int nao, r;
  double s;  // scale, so that two fills can be told apart
  cplx val(int a, int b, int i, int j) const {
    return s * cplx(0.3 * a - 0.1 * b + i, 0.2 * b + 0.05 * a * j) * std::exp(-0.01 * (a - b));
  }
  void row(herm_matrix_hodlr &G, int T) const {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix Y(nao, nao), L(nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) {
          Y(i, j) = val(T, tp, i, j);
          L(i, j) = val(tp, T, j, i) * II;
        }
      G.set_ret_curr(T, tp, Y);
      G.set_les_curr(tp, T, L);
    }
    for (int m = 0; m < r; m++) {
      ZMatrix X(nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) X(i, j) = s * cplx(T + m, i - j);
      G.set_tv(T, m, X);
    }
  }
  void mat(herm_matrix_hodlr &G) const {
    for (int m = 0; m < r; m++) {
      DMatrix M(nao, nao);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) M(i, j) = s * (0.1 * m + i + j);
      G.set_mat(m, M);
    }
    for (int a = 0; a < G.GMConvTensMatrix().size(); a++) G.GMConvTensMatrix().data()[a] = s * std::sin(0.37 * a);
  }
  // fill rows 0..k, then step up to row Tend (window [Tend-k, Tend])
  void run(herm_matrix_hodlr &G, Integration::Integrator &I, int Tend) const {
    mat(G);
    for (int T = 0; T <= G.k(); T++) row(G, T);
    for (int T = G.k() + 1; T <= Tend; T++) {
      G.update_blocks(I);
      row(G, T);
    }
  }
};
}  // namespace

// The s1 scenario: source at window [8,10], fresh destination.
TEST(HodlrCopy, GetTimestepCurrWindow) {
  hn_test::run_isolated([] {
    int nt = 40, nlvl = 2, k = 2, nao = 1, r = 3;
    Integration::Integrator I(k);
    herm_matrix_hodlr A(nt, r, nlvl, 1e-10, nao, nao, -1, k), B(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    auto f = [](int T, int tp) { return cplx(T, tp); };
    auto fill = [&](int T) {
      for (int tp = 0; tp <= T; tp++) {
        ZMatrix Y(1, 1);
        Y(0, 0) = f(T, tp);
        A.set_ret_curr(T, tp, Y);
        A.set_les_curr(tp, T, Y);
      }
    };
    for (int T = 0; T <= k; T++) fill(T);
    for (int T = k + 1; T <= 10; T++) {
      A.update_blocks(I);
      fill(T);
    }
    ASSERT_EQ(A.tstpmk(), 8);
    A.get_timestep_curr(B);
    EXPECT_EQ(A.tstpmk(), 8) << "source window must not change";
    ASSERT_EQ(B.tstpmk(), 8) << "destination must receive the source window";
    for (int T = 8; T <= 10; T++)
      for (int tp = 0; tp <= T; tp++) EXPECT_EQ(B.curr_timestep_ret_ptr(T, tp)[0], f(T, tp)) << "T=" << T << " tp=" << tp;
    // the source continues correctly: next archive is row 8
    A.update_blocks(I);
    fill(11);
    for (int i = 0; i <= k; i++) A.update_blocks(I);
    ZMatrix M(1, 1);
    for (int t = 0; t <= 11; t++)
      for (int tp = 0; tp <= t; tp++) {
        A.get_ret(t, tp, M);
        EXPECT_LT(std::abs(M(0, 0) - f(t, tp)), 1e-8) << "A(" << t << "," << tp << ") after copy";
      }
  });
}

// WP4 side finding 4 (fixed in WP9a): the window copies carry can_extrap, so extrapolate behaves the same on the copy
// (the window includes slot T = tstpmk+k, which holds the archived row T-k-1 that extrapolate reads).  The per-step
// copy into slot T of the destination clears the destination's can_extrap (ST-6), like every other writer of row T.
TEST(HodlrCopy, WindowCopyCarriesCanExtrap) {
  hn_test::run_isolated([] {
    int nt = 40, nlvl = 2, k = 2, nao = 2, r = 0;
    dlr_info dlr(r, 20., 1e-10, 2.0, nao, -1);
    Integration::Integrator I(k);
    dyson D(nt, nao, k, dlr);
    herm_matrix_hodlr A(nt, r, nlvl, 1e-10, nao, nao, -1, k), B(nt, r, nlvl, 1e-10, nao, nao, -1, k),
        C(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    auto fill = [&](int T) {
      for (int tp = 0; tp <= T; tp++) {
        ZMatrix Y = ZMatrix::Constant(nao, nao, std::exp(-II * 0.3 * double(T - tp))), L = -Y.adjoint();
        A.set_ret_curr(T, tp, Y);
        A.set_les_curr(tp, T, L);
      }
    };
    for (int T = 0; T <= k; T++) fill(T);
    for (int T = k + 1; T <= 10; T++) {
      A.update_blocks(I);
      fill(T);
    }
    A.update_blocks(I);   // tstpmk = 9, row 11 not written yet: A can extrapolate
    ASSERT_TRUE(A.can_extrap());
    A.get_timestep_curr(B);
    EXPECT_TRUE(B.can_extrap()) << "get_timestep_curr(G) must carry can_extrap";
    C.set_timestep_curr(A);
    EXPECT_TRUE(C.can_extrap()) << "set_timestep_curr(G) must carry can_extrap";
    D.extrapolate(A, I);
    D.extrapolate(B, I);
    const int T = A.tstpmk() + k;
    for (int tp = 0; tp <= T; tp++)
      for (int e = 0; e < nao * nao; e++) {
        ASSERT_EQ(A.curr_timestep_ret_ptr(T, tp)[e], B.curr_timestep_ret_ptr(T, tp)[e]) << "tp=" << tp;
        ASSERT_EQ(A.curr_timestep_les_ptr(tp, T)[e], B.curr_timestep_les_ptr(tp, T)[e]) << "tp=" << tp;
      }
    EXPECT_FALSE(A.can_extrap());
    EXPECT_FALSE(B.can_extrap());
    // the window copy carries false as well; a copy of row T into the destination's slot T clears its can_extrap
    A.get_timestep_curr(C);
    EXPECT_FALSE(C.can_extrap());
    B.can_extrap() = true;
    A.get_timestep_curr(T, B);
    EXPECT_FALSE(B.can_extrap()) << "per-step copy into slot T must clear the destination's can_extrap (ST-6)";
  });
}

// tstp == -1 copies mat in both directions with no window requirement (formerly unreachable in Debug: the window
// assert came before the tstp == -1 branch).  The two objects deliberately have different windows.
TEST(HodlrCopy, GetTimestepCurrMatsubara) {
  hn_test::run_isolated([] {
    int nt = 20, nlvl = 2, k = 2, nao = 2, r = 4;
    Integration::Integrator I(k);
    herm_matrix_hodlr A(nt, r, nlvl, 1e-10, nao, nao, -1, k), B(nt, r, nlvl, 1e-10, nao, nao, -1, k),
        C(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    CopyFill{nao, r, 1.0}.run(A, I, 9);  // window [7,9]; B, C stay at [0,2]
    for (int i = 0; i < r; i++) {
      DMatrix m(nao, nao);
      m << 0.1 * i, 1.0, 1.0, -0.2 * i;
      A.set_mat(i, m);
    }
    HodlrState sA = snapshot(A, "mat_src");
    A.get_timestep_curr(-1, B);
    for (int a = 0; a < r * nao * nao; a++) EXPECT_EQ(B.mat()[a], A.mat()[a]);
    EXPECT_EQ(B.tstpmk(), 0) << "the Matsubara copy must not touch the window";
    expect_same_state(sA, snapshot(A, "mat_src2"), "source after get_timestep_curr(-1)");
    C.set_timestep_curr(-1, A);
    for (int a = 0; a < r * nao * nao; a++) EXPECT_EQ(C.mat()[a], A.mat()[a]);
    EXPECT_EQ(C.tstpmk(), 0);
    expect_same_state(sA, snapshot(A, "mat_src3"), "source after set_timestep_curr(-1)");
  });
}

// get_timestep_curr(G): the source is bitwise unchanged (all checkpointed buffers, tstpmk, built_blocks, can_extrap);
// G receives the source's window and current data; a following update_blocks on both keeps them identical.
// G holds the same archive as the source (copy), but its window and window data are overwritten first.
TEST(HodlrCopy, GetTimestepCurrSourceUnchangedAndContinues) {
  int nt = 40, nlvl = 3, k = 3, nao = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr A(nt, r, nlvl, 1e-12, nao, nao, -1, k);
  CopyFill fa{nao, r, 1.0}, junk{nao, r, -3.7};
  fa.run(A, I, 20);
  ASSERT_EQ(A.tstpmk(), 17);
  HodlrState s0 = snapshot(A, "src0");

  herm_matrix_hodlr B(A);
  for (int T = A.tstpmk(); T <= A.tstp(); T++) junk.row(B, T);  // corrupt the whole window of B
  B.set_tstpmk(0);
  A.get_timestep_curr(B);
  expect_same_state(s0, snapshot(A, "src1"), "source after get_timestep_curr(G)");
  EXPECT_EQ(A.tstpmk(), 17);
  EXPECT_EQ(B.tstpmk(), 17) << "destination must receive the source window";
  expect_same_state(s0, snapshot(B, "dst1"), "destination after get_timestep_curr(G)");

  for (int T = 21; T < nt; T++) {
    A.update_blocks(I);
    B.update_blocks(I);
    fa.row(A, T);
    fa.row(B, T);
  }
  for (int i = 0; i <= k; i++) {
    A.update_blocks(I);
    B.update_blocks(I);
  }
  expect_same_state(snapshot(A, "src2"), snapshot(B, "dst2"), "after continued stepping", true);
  ZMatrix MA(nao, nao), MB(nao, nao);
  double e = 0;
  for (int t = 0; t < nt; t++)
    for (int tp = 0; tp <= t; tp++) {
      B.get_ret(t, tp, MB);
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) MA(i, j) = fa.val(t, tp, i, j);
      e = std::max(e, (MA - MB).norm());
    }
  EXPECT_LT(e, 1e-9) << "destination archive must hold the source's rows";
}

// set_timestep_curr(G) is the mirror operation: it restores *this's window from G and leaves G unchanged.
TEST(HodlrCopy, SetTimestepCurrSourceUnchanged) {
  int nt = 30, nlvl = 2, k = 2, nao = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr A(nt, r, nlvl, 1e-12, nao, nao, -1, k);
  CopyFill{nao, r, 1.0}.run(A, I, 12);
  HodlrState s0 = snapshot(A, "set0");
  herm_matrix_hodlr B(A);
  for (int T = A.tstpmk(); T <= A.tstp(); T++) CopyFill{nao, r, 2.5}.row(B, T);
  B.set_tstpmk(0);
  B.set_timestep_curr(A);
  EXPECT_EQ(B.tstpmk(), A.tstpmk());
  expect_same_state(s0, snapshot(A, "set1"), "source after set_timestep_curr(G)");
  expect_same_state(s0, snapshot(B, "set2"), "destination after set_timestep_curr(G)");
}

// Window / geometry preconditions of the per-step copies (WP4 asserts promoted to tier 2 in WP6b, HM-T1..T5):
// they throw h_nessi::precondition_error in every build.
TEST(HodlrCopyChecks, WindowAndGeometryThrow) {
  int nt = 20, k = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr A(nt, r, 2, 1e-10, 1, 1, -1, k), B(nt, r, 2, 1e-10, 1, 1, -1, k), C(nt, r, 2, 1e-10, 2, 2, -1, k);
  CopyFill{1, r, 1.0}.run(A, I, 9);  // A: [7,9], B: [0,2]
  A.get_timestep_curr(-1, B);        // fine in every build
  EXPECT_THROW(A.get_timestep_curr(9, B), precondition_error);   // not in G's window
  EXPECT_THROW(A.get_timestep_curr(1, B), precondition_error);   // not in the source's window
  EXPECT_THROW(B.set_timestep_curr(9, A), precondition_error);   // not in *this's window
  EXPECT_THROW(A.get_timestep_curr(-1, C), precondition_error);  // geometry mismatch
  EXPECT_THROW(A.get_timestep_curr(C), precondition_error);
  EXPECT_THROW(C.set_timestep_curr(A), precondition_error);
  EXPECT_THROW(A.set_timestep_curr(nt, B), precondition_error);  // LX-11: outer bound
  EXPECT_THROW(A.incr_timestep_curr(9, B, 1.0), precondition_error);  // HM-T5: G's slot 9 is not in G's window
  EXPECT_THROW(A.incr_timestep_curr(-1, C, 1.0), precondition_error);
  EXPECT_NO_THROW(A.incr_timestep_curr(-1, B, 1.0));
  try {
    A.get_timestep_curr(9, B);
  } catch (const precondition_error &e) {
    EXPECT_NE(std::string(e.what()).find("herm_matrix_hodlr::get_timestep_curr"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("G.tstpmk()=0"), std::string::npos) << e.what();
  }
}

// B6: a = b is a deep copy of the complete state; later changes to either object do not reach the other.  The
// target may have a different geometry before the assignment.
TEST(HodlrAssign, DeepCopyIndependent) {
  int nt = 30, nlvl = 2, k = 3, nao = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr b(nt, r, nlvl, 1e-12, nao, nao, -1, k), a(7, 2, 1, 1e-6, 1, 1, +1, 1), c;
  CopyFill fb{nao, r, 1.0};
  fb.run(b, I, 15);
  HodlrState sb = snapshot(b, "asg_b");
  a = b;
  c = b;
  herm_matrix_hodlr d(b);
  expect_same_state(sb, snapshot(a, "asg_a"), "a = b");
  expect_same_state(sb, snapshot(c, "asg_c"), "default-constructed c = b");
  expect_same_state(sb, snapshot(d, "asg_d"), "copy constructor");
  EXPECT_EQ(a.nt(), nt);
  EXPECT_EQ(a.size1(), nao);
  EXPECT_EQ(a.sig(), -1);
  EXPECT_EQ(a.k(), k);
  // modify b afterwards (window data, mat, tv, blocks via update_blocks): a is unchanged
  CopyFill{nao, r, -2.0}.mat(b);
  CopyFill{nao, r, -2.0}.row(b, b.tstp());
  b.update_blocks(I);
  fb.row(b, b.tstp());
  expect_same_state(sb, snapshot(a, "asg_a2"), "a after modifying b");
  // and the other way round
  CopyFill{nao, r, 5.0}.mat(d);
  d.update_blocks(I);
  expect_same_state(sb, snapshot(c, "asg_c2"), "c after modifying d");
  // self-assignment is harmless
  herm_matrix_hodlr &cref = c;
  c = cref;
  expect_same_state(sb, snapshot(c, "asg_c3"), "self-assignment");
}
// B6: a copy taken in the middle of a real propagation (dyson_mat -> initGMConvTensor -> dyson_start ->
// {update_blocks, extrapolate, set Sigma row, dyson_timestep}) continues exactly like the original.  G and Sigma are
// copied after step Tcopy (copy constructor and operator=); both pairs are then stepped to the end with their own
// dyson object, and the final states agree bitwise (except the wall-clock timing).  Sigma = V^2 g_bath (free bath
// level, exact formulas), h(t) driven after t = 0.3.
TEST(HodlrAssign, CopyContinuesDysonIdentically) {
  const int NO = 2, k = 3, nt = 41, nlvl = 2, Tcopy = 17;
  const double beta = 4., mu = 0.1, h = 0.05, eb = 0.7, V2 = 0.4;
  int r = 0, xi = -1;
  dlr_info dlr(r, 40., 1e-12, beta, NO, xi);
  Integration::Integrator I(k);
  DMatrix Mv(NO, NO);
  Mv << 1.0, 0.3, 0.3, 0.5;
  ZMatrix Mz = V2 * Mv.cast<cplx>();
  const double fb = 1. / (std::exp(beta * eb) - xi);
  auto SR = [&](int t, int tp) { return ZMatrix(-II * std::exp(-II * eb * double(t - tp) * h) * Mz); };
  auto SL = [&](int t, int tp) { return ZMatrix(-II * double(xi) * fb * std::exp(-II * eb * double(t - tp) * h) * Mz); };
  auto Stv = [&](int t, double tau) { return ZMatrix(-II * double(xi) * fb * std::exp(-II * eb * double(t) * h) * std::exp(eb * tau) * Mz); };
  function H(nt, NO, NO);
  for (int t = -1; t < nt; t++) {
    double tt = t < 0 ? 0. : t * h, drv = tt > 0.3 ? 0.4 * std::sin(2. * (tt - 0.3)) : 0.;
    ZMatrix Ht(NO, NO);
    Ht << cplx(0.2 + drv, 0), cplx(0.5, 0.1 * drv), cplx(0.5, -0.1 * drv), cplx(-0.3, 0);
    H.get_map(t) = Ht;
  }
  herm_matrix_hodlr G(nt, r, nlvl, 1e-12, NO, NO, xi, k), Sig(nt, r, nlvl, 1e-12, NO, NO, xi, k);
  auto set_sigma_row = [&](herm_matrix_hodlr &S, int T) {
    for (int tp = 0; tp <= T; tp++) {
      S.set_ret_curr(T, tp, SR(T, tp));
      S.set_les_curr(tp, T, SL(tp, T));
    }
    for (int j = 0; j < r; j++) S.set_tv(T, j, Stv(T, dlr.it0B(j)));
  };
  for (int j = 0; j < r; j++) {
    DMatrix sm = -(1. + xi * fb) * std::exp(-eb * dlr.it0B(j)) * V2 * Mv;
    Sig.set_mat(j, sm);
  }
  for (int T = 0; T <= k; T++) set_sigma_row(Sig, T);
  for (int T = 0; T <= k; T++)
    for (int tp = T + 1; tp <= k; tp++) Sig.set_ret_curr(T, tp, SR(T, tp));

  dyson D(nt, NO, k, dlr);
  D.dyson_mat(G, mu, H, Sig, false, 1.0);
  G.initGMConvTensor(dlr);
  D.dyson_start(G, mu, H, Sig, I, h);
  auto step = [&](dyson &Dx, herm_matrix_hodlr &Gx, herm_matrix_hodlr &Sx, int T) {
    Gx.update_blocks(I);
    Sx.update_blocks(I);
    Dx.extrapolate(Gx, I);
    set_sigma_row(Sx, T);
    Dx.dyson_timestep(T, Gx, mu, H, Sx, I, h);
  };
  for (int T = k + 1; T <= Tcopy; T++) step(D, G, Sig, T);

  herm_matrix_hodlr G2(G), Sig2;
  Sig2 = Sig;
  dyson D2(nt, NO, k, dlr);
  for (int T = Tcopy + 1; T < nt; T++) {
    step(D, G, Sig, T);
    step(D2, G2, Sig2, T);
  }
  for (int i = 0; i <= k; i++) {
    G.update_blocks(I);
    G2.update_blocks(I);
  }
  expect_same_state(snapshot(G, "dys_G"), snapshot(G2, "dys_G2"), "G vs its copy after continued propagation", true);
  expect_same_state(snapshot(Sig, "dys_S"), snapshot(Sig2, "dys_S2"), "Sigma vs its copy", true);
  ZMatrix A(NO, NO);
  G.get_ret(nt - 1, 0, A);
  EXPECT_GT(A.norm(), 1e-3) << "propagation must produce non-trivial data";
}

// ------------------------------------------------------------------------------------------------ B23
// B23 (fixed in WP6a): nlvl = 0 used to segfault in init_shape. The constructor now rejects every geometry with an
// empty HODLR segment (HM-C2: 1 <= nlvl and 2^nlvl <= nt) with h_nessi::precondition_error; nlvl = 1 still works.
TEST(HodlrGeometry, NlvlZeroRejected_B23) {
  hn_test::run_isolated([] {
    int nt = 20, k = 2;
    EXPECT_THROW(herm_matrix_hodlr(nt, 4, 0, 1e-10, 1, 1, -1, k), h_nessi::precondition_error);
    EXPECT_THROW(herm_matrix_hodlr(nt, 4, 5, 1e-10, 1, 1, -1, k), h_nessi::precondition_error);  // 2^5 = 32 > nt
    try {
      herm_matrix_hodlr G(nt, 4, 0, 1e-10, 1, 1, -1, k);
    } catch (const std::invalid_argument &e) {  // the std base class catches it too
      EXPECT_NE(std::string(e.what()).find("nlvl >= 1"), std::string::npos) << e.what();
    }
    Integration::Integrator I(k);
    herm_matrix_hodlr G(nt, 4, 1, 1e-10, 1, 1, -1, k);
    auto f = [](int T, int tp) { return cplx(T, tp); };
    for (int T = 0; T < nt; T++) {
      if (T > k) G.update_blocks(I);
      for (int tp = 0; tp <= T; tp++) {
        ZMatrix Y(1, 1);
        Y(0, 0) = f(T, tp);
        G.set_ret_curr(T, tp, Y);
      }
    }
    for (int i = 0; i <= k; i++) G.update_blocks(I);
    ZMatrix M(1, 1);
    for (int t = 0; t < nt; t++)
      for (int tp = 0; tp <= t; tp++) {
        G.get_ret(t, tp, M);
        EXPECT_LT(std::abs(M(0, 0) - f(t, tp)), 1e-10);
      }
  });
}

// ------------------------------------------------------------------------------------------------ B26
// B26 (fixed in WP9a): get_memory_usage must count the les U/V with the les ranks and include the singular values
// of ret and les.
// The block contribution of the report is isolated as the difference to a fresh object of the same geometry (its
// blocks hold rank-0 factors; every other term has the same size) and compared with the true storage of the compressed
// blocks (U, V complex; S real).  The formula of the pre-WP9a code (les U/V counted with the ret ranks, S not counted)
// must differ from it in this setup (different ret and les ranks).
TEST(HodlrMemory, MemoryReport_B26) {
  hn_test::run_isolated([] {
    int nt = 64, nlvl = 2, k = 2, nao = 2, r = 3;
    double hs = 0.05;
    Integration::Integrator I(k);
    herm_matrix_hodlr G(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    // ret: rank 1 (pure exponential); les: rank 3 (sum of three exponentials) -> different ranks
    auto fR = [&](int t, int tp) { return std::exp(-II * 0.7 * double(t - tp) * hs); };
    auto fL = [&](int t, int tp) {
      double d = (t - tp) * hs;
      return II * (0.4 * std::exp(-II * 0.9 * d) + 0.3 * std::exp(-II * 2.1 * d) + 0.2 * std::exp(II * 1.7 * d));
    };
    auto fill = [&](int T) {
      for (int tp = 0; tp <= T; tp++) {
        ZMatrix Y = ZMatrix::Constant(nao, nao, fR(T, tp)), L = ZMatrix::Constant(nao, nao, fL(tp, T));
        G.set_ret_curr(T, tp, Y);
        G.set_les_curr(tp, T, L);
      }
    };
    for (int T = 0; T <= k; T++) fill(T);
    for (int T = k + 1; T < nt; T++) {
      G.update_blocks(I);
      fill(T);
    }
    for (int i = 0; i <= k; i++) G.update_blocks(I);
    const double D = (double)sizeof(double) / 1e9, Z = 2 * D;
    double buggy = 0, correct = 0;
    bool ranks_differ = false;
    for (int b = 0; b < G.built_blocks(); b++)
      for (int i = 0; i < nao; i++)
        for (int j = 0; j < nao; j++) {
          auto &R = G.ret().data().blocks()[b][i][j];
          auto &L = G.les().data().blocks()[b][i][j];
          if (R.U().cols() != L.U().cols()) ranks_differ = true;
          buggy += Z * (R.U().rows() * R.U().cols() + R.V().rows() * R.V().cols());
          buggy += Z * (L.U().rows() * R.U().cols() + L.V().rows() * R.V().cols());
          correct += Z * (R.U().size() + R.V().size() + L.U().size() + L.V().size()) + D * (R.S().size() + L.S().size());
        }
    ASSERT_TRUE(ranks_differ) << "test setup must produce different ret and les ranks";
    herm_matrix_hodlr F(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    double blocks = G.get_memory_usage(false) - F.get_memory_usage(false);
    EXPECT_NEAR(blocks, correct, 1e-9 * correct) << "block contribution of the memory report (GB)";
    EXPECT_GT(std::abs(buggy - correct), 1e-3 * correct) << "test setup must distinguish the pre-WP9a formula";
  });
}

// ------------------------------------------------------------------------------------------------ B9
// B9 (fixed in WP9a): reading a checkpoint that has no /timing dataset must leave timing zero-initialised (the other constructors use
// DMatrix::Zero).  malloc perturbation makes uninitialised heap memory non-zero deterministically.
TEST(HodlrCheckpoint, TimingZeroWhenMissing_B9) {
  hn_test::run_isolated([] {
    mallopt(M_PERTURB, 0x5a);
    int nt = 20, nlvl = 2, k = 2, nao = 1, r = 3;
    herm_matrix_hodlr G(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    std::string fn = "hodlr_checkpoint_notiming.h5", fn2 = "hodlr_checkpoint_notiming_out.h5";
    std::remove(fn.c_str());
    std::remove(fn2.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      G.write_checkpoint_hdf5(out, "G/");
      out.unlink("G/timing");
    }
    h5e::File in(fn, h5e::File::ReadOnly);
    herm_matrix_hodlr G2(in, "G/");
    {
      h5e::File out(fn2, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      G2.write_checkpoint_hdf5(out, "G/");
    }
    h5e::File chk(fn2, h5e::File::ReadOnly);
    DMatrix timing = chk.getDataSet("G/timing").read<DMatrix>();
    ASSERT_GT(timing.size(), 0);
    EXPECT_EQ(timing.cwiseAbs().maxCoeff(), 0.0) << "timing read from a checkpoint without /timing is uninitialised";
  });
}

// ------------------------------------------------------------------------------------------------ h5_join (WP9a)
// WP3 side finding 5: HDF5 paths are joined without doubled slashes; "G" and "G/" name the same objects.
TEST(HodlrCheckpoint, H5JoinAndTrailingSlashLabels) {
  EXPECT_EQ(h5_join({"G/", "ret"}), "G/ret");
  EXPECT_EQ(h5_join({"G", "ret"}), "G/ret");
  EXPECT_EQ(h5_join({"G//", "/ret/", "U"}), "G/ret/U");
  EXPECT_EQ(h5_join({"", "x"}), "x");
  EXPECT_EQ(h5_join({"geometry", "", "blkr1"}), "geometry/blkr1");
  EXPECT_EQ(h5_join({"/abs/", "x"}), "/abs/x");
  hn_test::run_isolated([] {
    int nt = 12, nlvl = 2, k = 2, nao = 2, r = 3;
    herm_matrix_hodlr G(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    std::string fn = "hodlr_h5join.h5";
    std::remove(fn.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      G.write_checkpoint_hdf5(out, "A/");
      G.write_checkpoint_hdf5(out, "B");
    }
    h5e::File in(fn, h5e::File::ReadOnly);
    auto a = in.getGroup("A").listObjectNames(), b = in.getGroup("B").listObjectNames();
    EXPECT_EQ(a, b);
    herm_matrix_hodlr G2(in, "A"), G3(in, "B/");
    EXPECT_EQ(G2.nt(), nt);
    EXPECT_EQ(G3.nt(), nt);
  });
}

// ------------------------------------------------------------------------------------------------ B33
// B33 (fixed in WP9a): the constructor must zero-initialise GMConvTensMatrix_, so a checkpoint of a fresh object (no
// initGMConvTensor yet) holds zeros, not heap garbage.  malloc perturbation makes uninitialised memory non-zero.
TEST(HodlrCheckpoint, FreshGMConvTensorZero_B33) {
  hn_test::run_isolated([] {
    mallopt(M_PERTURB, 0x5a);
    int nt = 20, nlvl = 2, k = 2, nao = 2, r = 5;
    herm_matrix_hodlr G(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    std::string fn = "hodlr_checkpoint_fresh_gmconv.h5";
    std::remove(fn.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      G.write_checkpoint_hdf5(out, "G");
    }
    h5e::File chk(fn, h5e::File::ReadOnly);
    DMatrix M = chk.getDataSet("G/GMConvTensMatrix").read<DMatrix>();
    ASSERT_EQ(M.size(), (Eigen::Index)r * r * nao * nao);
    EXPECT_EQ(M.cwiseAbs().maxCoeff(), 0.0) << "GMConvTensMatrix of a fresh object is uninitialised";
    DMatrix T = chk.getDataSet("G/timing").read<DMatrix>();
    EXPECT_EQ(T.cwiseAbs().maxCoeff(), 0.0);
  });
}

// ------------------------------------------------------------------------------------------------ checkpoint format_version (WP3, WP6c)
// write_checkpoint_hdf5 writes label/format_version = 3 (2: GMConvTensMatrix carries the xi factor, BUGS B1 fix;
// 3: no tv_trans dataset, ISSUES H27). Loading a bosonic checkpoint without it (written before WP3) still works, but
// warns on std::cerr that the stored GMConvTensMatrix is stale and must be rebuilt with initGMConvTensor(dlr).
// Fermionic checkpoints never warn. Files of version 1 and 2 carry a tv_trans dataset, which the reader ignores.
namespace {
struct CkptCase {
  int xi, nt = 12, nlvl = 2, k = 2, nao = 2, r = 0;
  dlr_info dlr;
  herm_matrix_hodlr G;
  explicit CkptCase(int xi_) : xi(xi_), dlr(r, 40., 1e-12, 4., 2, xi_), G(nt, r, nlvl, 1e-12, nao, nao, xi_, k) {
    for (int m = 0; m < r; m++) {
      double tau = dlr.it0B(m);
      DMatrix M(nao, nao);
      M << -exp(-0.5 * tau), 0.1 * exp(-tau), 0.1 * exp(-tau), -0.7 * exp(-1.3 * tau);
      G.set_mat(m, M);
    }
    G.initGMConvTensor(dlr);
    for (int t = 0; t < nt; t++)   // non-symmetric complex tv data (all rows), so that the tv round trip is checked
      for (int m = 0; m < r; m++)
        for (int i = 0; i < nao; i++)
          for (int j = 0; j < nao; j++) G.tvptr(t, m)[i * nao + j] = cplx(0.1 * t + m + 0.3 * i, -0.2 * j + 0.01 * m * t);
  }
  // write a checkpoint; optionally rewrite it as an older format: strip_version (a pre-WP3 file, version 1) or
  // as_version = 2 (WP3..WP6b); both older formats get a tv_trans dataset (here filled with garbage, which the reader
  // must ignore). Load it, capturing stderr. *tv_diff: max |tv - tv read back|.
  std::string roundtrip(const std::string &fn, bool strip_version, int *version_on_disk, double *tens_diff,
                        int as_version = 0, double *tv_diff = nullptr) {
    std::remove(fn.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      G.write_checkpoint_hdf5(out, "G/");
      if (strip_version || as_version != 0) {
        std::vector<cplx> junk((size_t)nt * r * nao * nao, cplx(1e300, -1e300));
        h5e::dump(out, "G/tv_trans", junk);
      }
      if (strip_version) out.unlink("G/format_version");
      else if (as_version != 0) h5e::dump(out, "G/format_version", as_version, h5e::DumpMode::Overwrite);
    }
    h5e::File in(fn, h5e::File::ReadOnly);
    *version_on_disk = in.exist("G/format_version") ? in.getDataSet("G/format_version").read<int>() : -1;
    ::testing::internal::CaptureStderr();
    herm_matrix_hodlr G2(in, "G/");
    std::string err = ::testing::internal::GetCapturedStderr();
    double d = 0;
    for (int i = 0; i < r * r * nao * nao; i++) d = std::max(d, std::abs(G.GMConvTens()[i] - G2.GMConvTens()[i]));
    *tens_diff = d;
    if (tv_diff) {
      double e = 0;
      for (size_t q = 0; q < G.tv().data_vector().size(); q++)
        e = std::max(e, std::abs(G.tv().data_vector()[q] - G2.tv().data_vector()[q]));
      *tv_diff = e;
    }
    return err;
  }
};
}  // namespace

TEST(HodlrCheckpoint, FormatVersionNewBosonicLoadsSilently) {
  CkptCase c(+1);
  int v = 0;
  double d = 1;
  double dtv = 1;
  std::string err = c.roundtrip("hodlr_ckpt_v3_boson.h5", false, &v, &d, 0, &dtv);
  EXPECT_EQ(v, herm_matrix_hodlr::checkpoint_format_version);
  EXPECT_EQ(v, 3);
  EXPECT_EQ(err, "") << "a current bosonic checkpoint must load without a warning";
  EXPECT_EQ(d, 0.0);
  EXPECT_EQ(dtv, 0.0);
}

// WP6c (H27): version 3 writes no tv_trans; the tv data round-trips exactly (both statistics).
TEST(HodlrCheckpoint, FormatV3HasNoTvTransAndRoundTrips) {
  for (int xi : {-1, +1}) {
    CkptCase c(xi);
    std::string fn = "hodlr_ckpt_v3_notvtrans.h5";
    int v = 0;
    double d = 1, dtv = 1;
    std::string err = c.roundtrip(fn, false, &v, &d, 0, &dtv);
    h5e::File in(fn, h5e::File::ReadOnly);
    EXPECT_TRUE(in.exist("G/tv"));
    EXPECT_FALSE(in.exist("G/tv_trans")) << "xi=" << xi;
    EXPECT_EQ(v, 3);
    EXPECT_EQ(err, "");
    EXPECT_EQ(dtv, 0.0) << "xi=" << xi;
  }
}

// WP6c (H27): a version-2 checkpoint (with a tv_trans dataset) still loads; its tv_trans is ignored, tv and the
// conv tensor are read unchanged, and no warning is printed (version 2 already has the bosonic-sign fix).
TEST(HodlrCheckpoint, FormatV2WithTvTransStillLoads) {
  for (int xi : {-1, +1}) {
    CkptCase c(xi);
    int v = 0;
    double d = 1, dtv = 1;
    std::string err = c.roundtrip("hodlr_ckpt_v2_tvtrans.h5", false, &v, &d, 2, &dtv);
    EXPECT_EQ(v, 2);
    EXPECT_EQ(err, "") << "xi=" << xi;
    EXPECT_EQ(d, 0.0);
    EXPECT_EQ(dtv, 0.0) << "the (garbage) tv_trans of an old file must not be read";
    EXPECT_TRUE(c.G.gmconv_current());
  }
}

// A checkpoint from a newer format than this library knows is refused (tier 2).
TEST(HodlrCheckpoint, FormatVersionNewerThanLibraryThrows) {
  CkptCase c(-1);
  std::string fn = "hodlr_ckpt_v_future.h5";
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    c.G.write_checkpoint_hdf5(out, "G/");
    h5e::dump(out, "G/format_version", herm_matrix_hodlr::checkpoint_format_version + 1, h5e::DumpMode::Overwrite);
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  EXPECT_THROW(herm_matrix_hodlr G2(in, "G/"), precondition_error);
}

TEST(HodlrCheckpoint, FormatVersionMissingBosonicWarns) {
  CkptCase c(+1);
  int v = 0;
  double d = 1;
  double dtv = 1;
  std::string err = c.roundtrip("hodlr_ckpt_v1_boson.h5", true, &v, &d, 0, &dtv);
  EXPECT_EQ(v, -1) << "format_version must have been removed";
  EXPECT_EQ(dtv, 0.0) << "the tv_trans of a version-1 file must not be read";
  EXPECT_NE(err.find("format_version 1 < 2"), std::string::npos) << "stderr was: " << err;
  EXPECT_NE(err.find("initGMConvTensor"), std::string::npos) << "stderr was: " << err;
  EXPECT_EQ(d, 0.0) << "an old checkpoint must still load unchanged";
}

TEST(HodlrCheckpoint, FormatVersionMissingFermionicSilent) {
  CkptCase c(-1);
  int v = 0;
  double d = 1;
  std::string err = c.roundtrip("hodlr_ckpt_v1_fermion.h5", true, &v, &d);
  EXPECT_EQ(err, "") << "fermionic tensors are unaffected by the bosonic-sign fix";
  EXPECT_EQ(d, 0.0);
}

// write_to_hdf5 (output files, not checkpoints) is unchanged: no format_version.
TEST(HodlrCheckpoint, FormatVersionNotInOutputFiles) {
  CkptCase c(+1);
  std::string fn = "hodlr_output_noversion.h5";
  std::remove(fn.c_str());
  {
    h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    c.G.write_to_hdf5(out, "G");
  }
  h5e::File in(fn, h5e::File::ReadOnly);
  EXPECT_TRUE(in.exist("G/nt"));
  EXPECT_FALSE(in.exist("G/format_version"));
}
