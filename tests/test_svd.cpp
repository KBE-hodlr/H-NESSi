// Small-SVD kernels (BUGS B38): svd()/tsvd() of general complex matrices and arrow_svd(), the SVD of the arrow matrix
// K = [diag(S) 0; conj(Vrow) nrm] of the incremental TSVD (herm_matrix_hodlr::updatetsvd, doc 12).
// Eigen 3.4.0's BDCSVD returns wrong factorisations (orthonormal U, V, wrong product) for some matrices of size >= 16,
// complex and real. Data (tests/data): b38_K_bad.bin = the 18x18 complex matrix found in the SVD study (SC nt=2048,
// DeltaPlusSigma R block 191, row 171; stored as K^H, i.e. with the arrow in the last column); b38_A_real_bad_n68.bin =
// a 68x68 real arrow matrix on which Eigen 3.4.0's real BDCSVD fails (relative residual 1e-7). File format: int32 n,
// then n*n column-major entries (complex<double> resp. double).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <Eigen/SVD>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/utils.hpp"

using namespace h_nessi;

namespace h_nessi {
struct herm_matrix_hodlr_test_access {
  static void updatetsvd(herm_matrix_hodlr &G, block &B, const ZRowVector &row, int row_cur) {
    G.updatetsvd(B, row, row_cur);
  }
};
}  // namespace h_nessi
using TA = h_nessi::herm_matrix_hodlr_test_access;

namespace {
const double eps = std::numeric_limits<double>::epsilon();
constexpr bool eigen_340 = EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION == 4 && EIGEN_MINOR_VERSION == 0;

template <class Mat> Mat read_matrix(const std::string &name) {
  const std::string fn = std::string(HN_TEST_DATA_DIR) + "/" + name;
  FILE *f = std::fopen(fn.c_str(), "rb");
  EXPECT_NE(f, nullptr) << fn;
  if (!f) return Mat();
  int32_t n = 0;
  EXPECT_EQ(std::fread(&n, 4, 1, f), 1u);
  Eigen::Matrix<typename Mat::Scalar, Eigen::Dynamic, Eigen::Dynamic> M(n, n);   // column-major on disk
  EXPECT_EQ(std::fread(M.data(), sizeof(typename Mat::Scalar), n * n, f), size_t(n) * n);
  std::fclose(f);
  return M;
}

// the three parts of a lower arrow matrix K = [diag(s) 0; w beta]
struct Arrow {
  DColVector s;
  ZRowVector w;
  double beta;
  ZMatrix K() const {
    const int r = s.size();
    ZMatrix k = ZMatrix::Zero(r + 1, r + 1);
    for (int i = 0; i < r; i++) k(i, i) = s(i);
    k.block(r, 0, 1, r) = w;
    k(r, r) = beta;
    return k;
  }
};
Arrow arrow_from(const ZMatrix &K) {
  const int r = K.rows() - 1;
  Arrow a{DColVector(r), K.block(r, 0, 1, r), K(r, r).real()};
  for (int i = 0; i < r; i++) a.s(i) = K(i, i).real();
  return a;
}

double orth(const ZMatrix &Q) { return (Q.adjoint() * Q - ZMatrix::Identity(Q.cols(), Q.cols())).norm(); }

// residual, orthogonality, ordering and agreement of S with JacobiSVD of K; tolerances relative to ||K||_F
void check_factorisation(const ZMatrix &K, const ZMatrix &U, const ZMatrix &V, const DColVector &S, double tol,
                         const std::string &what) {
  const double kn = std::max(K.norm(), 1e-300);
  EXPECT_LE((K - U * S.asDiagonal() * V.adjoint()).norm(), tol * kn) << what;
  EXPECT_LE(orth(U), tol) << what;
  EXPECT_LE(orth(V), tol) << what;
  for (int i = 1; i < S.size(); i++) EXPECT_GE(S(i - 1), S(i)) << what << " i=" << i;
  Eigen::JacobiSVD<ZMatrix> jc(K);
  EXPECT_LE((S - jc.singularValues()).cwiseAbs().maxCoeff(), tol * kn) << what;
}
}  // namespace

// The matrix of the SVD study: the library's arrow path factorises it to round-off (Eigen 3.4.0's complex BDCSVD,
// used before the fix, has residual 1.4e-3 on K and 5.9e-3 on K^H).
TEST(SvdB38, KBadArrowSvd) {
  ZMatrix Kh = read_matrix<ZMatrix>("b38_K_bad.bin");
  ASSERT_EQ(Kh.rows(), 18);
  ZMatrix K = Kh.adjoint();   // the updatetsvd form: diag(S) and the last row conj(Vrow), nrm
  Arrow a = arrow_from(K);
  EXPECT_LE((a.K() - K).norm(), 0.0) << "K_bad is not an arrow matrix";
  ZMatrix U, V;
  DColVector S;
  arrow_svd(a.s, a.w, a.beta, U, V, S);
  check_factorisation(K, U, V, S, 1e-13, "arrow_svd(K_bad^H)");
  if (eigen_340) {   // document the Eigen defect the fix works around
    Eigen::BDCSVD<ZMatrix> bd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double res = (K - bd.matrixU() * bd.singularValues().asDiagonal() * bd.matrixV().adjoint()).norm() / K.norm();
    EXPECT_GT(res, 1e-6) << "Eigen 3.4.0 complex BDCSVD unexpectedly correct on K_bad";
  }
}

// The public svd()/tsvd() (formerly plain BDCSVD) are correct on K_bad and K_bad^H.
TEST(SvdB38, KBadPublicSvdTsvd) {
  ZMatrix Kh = read_matrix<ZMatrix>("b38_K_bad.bin");
  for (int t = 0; t < 2; t++) {
    ZMatrix K = t ? ZMatrix(Kh.adjoint()) : Kh, U, V;
    DColVector S, S2;
    svd(K, U, V, S);
    check_factorisation(K, U, V, S, 1e-13, t ? "svd(K)" : "svd(K^H)");
    svd(K, S2);
    EXPECT_LE((S - S2).cwiseAbs().maxCoeff(), 1e-14 * K.norm());
    const double tol = 1e-7;
    int rk = tsvd(K, U, V, S, tol);
    EXPECT_EQ(rk, (S2.array() >= tol).count());
    double dropped = std::sqrt(S2.tail(S2.size() - rk).squaredNorm());
    EXPECT_LE((K - U * S.asDiagonal() * V.adjoint()).norm(), dropped + 1e-13 * K.norm());
  }
}

// A real arrow matrix (n = 68) on which Eigen 3.4.0's real BDCSVD fails: arrow_svd detects it and uses JacobiSVD.
TEST(SvdB38, RealBdcFailureFallsBackToJacobi) {
  Eigen::MatrixXd A = read_matrix<Eigen::MatrixXd>("b38_A_real_bad_n68.bin");
  ASSERT_EQ(A.rows(), 68);
  const int r = A.rows() - 1;
  // a complex arrow matrix with |w| = A's last row and random phases
  std::mt19937 g(3);
  std::uniform_real_distribution<double> ph(-M_PI, M_PI);
  Arrow a{DColVector(r), ZRowVector(r), A(r, r)};
  for (int i = 0; i < r; i++) {
    a.s(i) = A(i, i);
    a.w(i) = std::polar(A(r, i), ph(g));
  }
  ZMatrix U, V;
  DColVector S;
  svd_path p = arrow_svd(a.s, a.w, a.beta, U, V, S);
  check_factorisation(a.K(), U, V, S, 1e-13, "arrow_svd(A_real_bad)");
  if (eigen_340) {
    EXPECT_EQ(p, svd_path::bdc_rejected_jacobi);
    Eigen::BDCSVD<Eigen::MatrixXd> bd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_GT((A - bd.matrixU() * bd.singularValues().asDiagonal() * bd.matrixV().transpose()).norm() / A.norm(), 1e-9);
  }
}

// Random arrow matrices r = 0..120 (n = r+1 up to 121) with graded S as in the HODLR blocks and degenerate cases:
// repeated / (nearly) equal S, zero S, zero entries of w, w ~ 1e-16, beta = 0, all zero.
TEST(SvdB38, ArrowSvdRandomVsJacobi) {
  std::mt19937 g(2026);
  std::normal_distribution<double> N;
  std::uniform_real_distribution<double> U01(0, 1);
  int counts[3] = {0, 0, 0};
  for (int trial = 0; trial < 1210; trial++) {
    const int r = trial % 121, mode = (trial / 121) % 10;
    Arrow a{DColVector(r), ZRowVector(r), std::abs(N(g)) * 1e-7};
    for (int i = 0; i < r; i++) a.s(i) = std::pow(10., -12 * U01(g));
    std::sort(a.s.data(), a.s.data() + r, std::greater<double>());
    for (int i = 0; i < r; i++) a.w(i) = cplx(N(g), N(g)) * std::pow(10., -8 * U01(g));
    switch (mode) {
    case 1: a.s.setOnes(); break;
    case 2: for (int i = r / 2; i < r; i++) a.s(i) = 0; break;
    case 3: for (int i = 0; i < r; i += 2) a.w(i) = 0; break;
    case 4: a.w *= 1e-16; break;
    case 5: for (int i = 1; i < r; i++) a.s(i) = a.s(0) * (1 + 1e-15 * i); break;
    case 6: a.beta = 0; break;
    case 7: a.s.setZero(); a.w.setZero(); a.beta = 0; break;
    case 8: a.w.setZero(); break;
    default: break;
    }
    ZMatrix U, V;
    DColVector S;
    svd_path p = arrow_svd(a.s, a.w, a.beta, U, V, S);
    counts[static_cast<int>(p)]++;
    ASSERT_EQ(S.size(), r + 1);
    check_factorisation(a.K(), U, V, S, 1e-13, "r=" + std::to_string(r) + " mode=" + std::to_string(mode));
    if (HasFailure()) break;
  }
  std::printf("arrow_svd paths: jacobi %d, bdc %d, bdc rejected -> jacobi %d\n", counts[0], counts[1], counts[2]);
  EXPECT_GT(counts[1], 0);
}

// Public svd()/tsvd(): random complex matrices (square, tall, wide; full rank and low rank) up to 120.
TEST(SvdB38, PublicSvdRandomComplex) {
  std::mt19937 g(11);
  std::uniform_int_distribution<int> dim(1, 120);
  for (int trial = 0; trial < 60; trial++) {
    const int m = dim(g), n = (trial % 3 == 0) ? m : dim(g), k = std::min(m, n);
    ZMatrix M = ZMatrix::Random(m, n);
    if (trial % 2) {   // graded low-rank
      ZMatrix A = ZMatrix::Random(m, k), B = ZMatrix::Random(k, n);
      for (int i = 0; i < k; i++) A.col(i) *= std::pow(10., -0.5 * i);
      M = A * B;
    }
    ZMatrix U, V;
    DColVector S;
    svd(M, U, V, S);
    const std::string what = std::to_string(m) + "x" + std::to_string(n);
    EXPECT_LE((M - U * S.asDiagonal() * V.adjoint()).norm(), 1e-13 * M.norm()) << what;
    EXPECT_LE(orth(U), 1e-13) << what;
    EXPECT_LE(orth(V), 1e-13) << what;
    Eigen::JacobiSVD<ZMatrix> jc(M);
    EXPECT_LE((S - jc.singularValues()).cwiseAbs().maxCoeff(), 1e-13 * M.norm()) << what;
    const double tol = 1e-6 * S(0);
    int rk = tsvd(M, U, V, S, tol);
    EXPECT_EQ(rk, (jc.singularValues().array() >= tol).count()) << what;
  }
}

// updatetsvd on a block whose arrow matrix is K_bad^H (the failing step of the SVD study): the updated block
// reproduces the data up to the discarded singular value (before the fix: error 1.4e-3, ~1e4 svdtol).
TEST(SvdB38, UpdatetsvdOnKBad) {
  ZMatrix K = ZMatrix(read_matrix<ZMatrix>("b38_K_bad.bin").adjoint());
  const int r = K.rows() - 1, cols = 40;
  const double svdtol = 1e-7;
  Arrow a = arrow_from(K);
  // block with U = I (r rows), S = s, V = r orthonormal columns; the new row has projection Vrow = conj(w) on V and
  // the residual norm beta along q orthogonal to V, so that updatetsvd forms exactly K.
  ZMatrix Q;
  {
    ZMatrix X = ZMatrix::Random(cols, r + 1);
    Eigen::HouseholderQR<ZMatrix> qr(X);
    Q = qr.householderQ() * ZMatrix::Identity(cols, r + 1);
  }
  block b(r, cols, r, svdtol);
  b.U() = ZMatrix::Identity(r, r);
  b.S() = a.s;
  b.V() = Q.leftCols(r);
  b.set_rank(r);
  ZColVector rowH = Q.leftCols(r) * a.w.conjugate().transpose() + a.beta * Q.col(r);
  ZRowVector row = rowH.adjoint();
  ZMatrix B(r + 1, cols);
  B.topRows(r) = a.s.asDiagonal() * Q.leftCols(r).adjoint();
  B.row(r) = row;
  herm_matrix_hodlr G(64, 1, 2, svdtol, 1, 1, -1, 5);
  TA::updatetsvd(G, b, row, r);
  Eigen::JacobiSVD<ZMatrix> jc(K);
  const DColVector sv = jc.singularValues();
  const int rk = b.epsrank();
  EXPECT_EQ(rk, sv(r) >= svdtol ? r + 1 : r);
  const double dropped = rk == r ? sv(r) : 0.0;
  const double err = (B - b.U() * b.S().asDiagonal() * b.V().adjoint()).norm();
  EXPECT_LE(err, dropped * (1 + 1e-6) + 1e-13 * B.norm()) << "dropped sigma " << dropped;
  EXPECT_LE(orth(b.U()), 1e-13);
  EXPECT_LE(orth(b.V()), 1e-13);
}
