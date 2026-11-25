// HODLR storage (docs 10-12): rank-1 TSVD update, fill / archive / read-back round trips, checkpoint I/O,
// copy semantics (B22), geometry edge cases (B23), memory report (B26), checkpoint timing default (B9).
// Sources: the former tests/linalg.cpp (updated to the h_nessi API), agent_scratch/52/s9.cpp (zero first rows),
// agent_scratch/52/s1.cpp (B22).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <malloc.h>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/utils.hpp"
#include "common/test_util.hpp"

using namespace h_nessi;
namespace {
const cplx II(0, 1);
}

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
  double nrm = G.mgs(Vnew, x, y, Vb);
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
  for (int i = 1; i < rows; i++) G.updatetsvd(b, M.row(i), i);
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

// ------------------------------------------------------------------------------------------------ B22
// get_timestep_curr(G) must copy THIS object's current window into G (G.tstpmk := this->tstpmk) and leave the
// source untouched (agent_scratch/52/s1.cpp).
TEST(HodlrCopy, KnownBug_B22_GetTimestepCurrWindow) {
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

// The per-step overload asserts tstp >= tstpmk_ before its tstp == -1 branch, so the Matsubara copy is
// unreachable when asserts are active (Debug).  In Release this passes.
TEST(HodlrCopy, KnownBugDebug_B22_GetTimestepCurrMatsubara) {
  hn_test::run_isolated([] {
    int nt = 20, nlvl = 2, k = 2, nao = 2, r = 4;
    herm_matrix_hodlr A(nt, r, nlvl, 1e-10, nao, nao, -1, k), B(nt, r, nlvl, 1e-10, nao, nao, -1, k);
    for (int i = 0; i < r; i++) {
      DMatrix m(nao, nao);
      m << 0.1 * i, 1.0, 1.0, -0.2 * i;
      A.set_mat(i, m);
    }
    A.get_timestep_curr(-1, B);
    for (int a = 0; a < r * nao * nao; a++) EXPECT_EQ(B.mat()[a], A.mat()[a]);
  });
}

// ------------------------------------------------------------------------------------------------ B23
// nlvl = 0 must not crash: either it is rejected with an exception or it works (no compression).
TEST(HodlrGeometry, KnownBug_B23_NlvlZeroNoCrash) {
  hn_test::run_isolated([] {
    try {
      int nt = 20, k = 2;
      Integration::Integrator I(k);
      herm_matrix_hodlr G(nt, 4, 0, 1e-10, 1, 1, -1, k);
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
    } catch (...) {
      // rejecting nlvl = 0 with an exception is acceptable
    }
  });
}

// ------------------------------------------------------------------------------------------------ B26
// get_memory_usage must count the les U/V with the les ranks and include the singular values of ret and les.
// The expected value replaces the buggy ret/les contribution of the code's formula by the true storage of the
// compressed blocks (U, V complex; S real); all other contributions are left as computed by the library.
TEST(HodlrMemory, KnownBug_B26_MemoryReport) {
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
    double total = G.get_memory_usage(false);
    double expected = total - buggy + correct;
    EXPECT_NEAR(total, expected, 1e-6 * expected) << "memory report (GB)";
  });
}

// ------------------------------------------------------------------------------------------------ B9
// Reading a checkpoint that has no /timing dataset must leave timing zero-initialised (the other constructors use
// DMatrix::Zero).  malloc perturbation makes uninitialised heap memory non-zero deterministically.
TEST(HodlrCheckpoint, KnownBug_B9_TimingZeroWhenMissing) {
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
