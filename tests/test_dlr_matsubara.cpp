// DLR maps, Matsubara/mixed accessors, imaginary-time integrals and dyson_mat (doc 03).
// Ported from agent_scratch/03/test_dlr.cpp and agent_scratch/51/t_mat.cpp.
// References: analytic free G^M of a real symmetric H (with the correct xi-periodic extension
// G^M(s<0) = xi G^M(s+beta)), analytic exponential sums for G^tv/Sigma^tv, composite Gauss-Legendre quadrature,
// and the exact hybridised G^M of an enlarged Hamiltonian for dyson_mat.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <tuple>

#include "h_nessi/dlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "common/test_util.hpp"

using namespace h_nessi;
typedef std::complex<double> C;

namespace {

void gauleg(int n, std::vector<double> &x, std::vector<double> &w) {
  x.resize(n);
  w.resize(n);
  for (int i = 0; i < n; i++) {
    double z = cos(M_PI * (i + 0.75) / (n + 0.5)), pp, z1;
    do {
      double p1 = 1, p2 = 0;
      for (int j = 0; j < n; j++) {
        double p3 = p2;
        p2 = p1;
        p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
      }
      pp = n * (z * p1 - p2) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
    } while (fabs(z - z1) > 1e-15);
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * pp * pp);
  }
}
void quad(double a, double b, std::vector<double> &X, std::vector<double> &W) {
  std::vector<double> x, w;
  gauleg(30, x, w);
  int nsub = 40;
  X.clear();
  W.clear();
  for (int s = 0; s < nsub; s++) {
    double a1 = a + (b - a) * s / nsub, b1 = a + (b - a) * (s + 1) / nsub;
    for (int i = 0; i < 30; i++) {
      X.push_back(0.5 * (a1 + b1) + 0.5 * (b1 - a1) * x[i]);
      W.push_back(0.5 * (b1 - a1) * w[i]);
    }
  }
}

// analytic free G^M for real symmetric H (mu=0), s in [-beta, beta]
struct FreeG {
  int n, xi;
  double beta;
  DMatrix U;
  DColVector e;
  FreeG(const DMatrix &H, double b, int x) : n(H.rows()), xi(x), beta(b) {
    Eigen::MatrixXd Hc = H;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hc);
    U = es.eigenvectors();
    e = es.eigenvalues();
  }
  DMatrix operator()(double s) const {
    double sg = 1.;
    if (s < 0) {
      s += beta;
      sg = xi;
    }
    DColVector d(n);
    for (int i = 0; i < n; i++) {
      if (xi == -1) d(i) = -exp(-e(i) * s) / (1 + exp(-beta * e(i)));
      else d(i) = -exp(-e(i) * s) / (1 - exp(-beta * e(i)));
    }
    return sg * U * d.asDiagonal() * U.transpose();
  }
};

struct ExpSum {
  int n;
  std::vector<double> a;
  std::vector<ZMatrix> c;
  ExpSum(int n_, int seed) : n(n_) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-1, 1);
    a = {0.37, -0.52, 1.9, 4.1};
    for (size_t p = 0; p < a.size(); p++) {
      ZMatrix m(n, n);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) m(i, j) = C(u(g), u(g));
      c.push_back(m);
    }
  }
  ZMatrix operator()(double tau) const {
    ZMatrix r = ZMatrix::Zero(n, n);
    for (size_t p = 0; p < a.size(); p++) r += c[p] * exp(-a[p] * tau);
    return r;
  }
};

struct DlrMetrics {
  double ipmat_asym, get_mat_tau, density_matrix, mat_reversed, tv_tau, tv_reversed;
  double tv_it_conv, tv_it_conv_ref, tv_it_conv_row;           // abs err, |ref|, single-row mismatch
  double les_it_int, les_it_int_ref, les_it_int_row;
  double gamma_mixed, gamma_mixed_ref;
  double gamma_mat, gamma_mat_ref;
  double dyson_mat, dyson_mat_ret_mismatch, dyson_mat_mu, dyson_mat_alpha;
};

DlrMetrics compute(int xi, int nao) {
  DlrMetrics R{};
  double beta = 10., lambda = 100., eps = 1e-12;
  int r = 0;
  dlr_info dlr(r, lambda, eps, beta, nao, xi);
  {
    DMatrixMap ip(dlr.ipmat(), r, r);
    R.ipmat_asym = (ip - DMatrix(ip.transpose())).norm() / ip.norm();
  }
  int es = nao * nao, nt = 64, nlvl = 3, k = 5;
  DMatrix H(nao, nao);
  if (nao == 1) H << 0.7;
  else H << 0.7, 0.3, 0.3, 1.4;
  FreeG g(H, beta, xi);
  herm_matrix_hodlr G(nt, r, nlvl, 1e-10, nao, nao, xi, k);
  herm_matrix_hodlr S(nt, r, nlvl, 1e-10, nao, nao, xi, k);
  for (int i = 0; i < r; i++) G.set_mat(i, g(dlr.it0B(i)));

  double err = 0;
  for (int q = 0; q <= 50; q++) {
    double tau = beta * q / 50.;
    DMatrix M(nao, nao);
    G.get_mat_tau(tau, dlr, M);
    err = std::max(err, (M - g(tau)).norm());
  }
  R.get_mat_tau = err;
  {
    DMatrix rho(nao, nao);
    G.density_matrix(-1, dlr, rho);
    R.density_matrix = (rho + g(beta)).norm();
  }
  {
    std::vector<double> rev(r * es);
    G.get_mat_reversed(dlr, rev.data());
    double e2 = 0;
    for (int i = 0; i < r; i++) e2 = std::max(e2, (DMatrixMap(rev.data() + i * es, nao, nao) - g(beta - dlr.it0B(i))).norm());
    R.mat_reversed = e2;
  }
  ExpSum sf(nao, 11 + xi), gf(nao, 23 + xi);
  for (int t = 0; t <= 3; t++)
    for (int i = 0; i < r; i++) {
      S.set_tv(t, i, ZMatrix(sf(dlr.it0B(i)) * C(1. + 0.1 * t, 0.05 * t)));
      G.set_tv(t, i, ZMatrix(gf(dlr.it0B(i)) * C(1. - 0.1 * t, 0.2 * t)));
    }
  {
    double e3 = 0, e4 = 0;
    std::vector<C> rev(r * es);
    G.get_tv_reversed(2, dlr, rev.data());
    for (int q = 0; q <= 20; q++) {
      double tau = beta * q / 20.;
      ZMatrix M(nao, nao);
      G.get_tv_tau(2, tau, dlr, M);
      e3 = std::max(e3, (M - gf(tau) * C(0.8, 0.4)).norm());
    }
    for (int i = 0; i < r; i++)
      e4 = std::max(e4, (ZMatrixMap(rev.data() + i * es, nao, nao) - gf(beta - dlr.it0B(i)) * C(0.8, 0.4)).norm());
    R.tv_tau = e3;
    R.tv_reversed = e4;
  }
  // tv_it_conv(t) = int_0^beta Sigma^tv(t,tb) G^M(tb - tau) dtb, with G^M(s<0) = xi G^M(s+beta)
  G.initGMConvTensor(dlr);
  dyson D(nt, nao, k, dlr);
  std::vector<C> res(r * es, C(0.));
  int tt = 1;
  D.tv_it_conv(tt, S, G, res.data());
  double e5 = 0, nrm = 0, rowmis = 0;
  for (int i = 0; i < r; i++) {
    double ti = dlr.it0B(i);
    ZMatrix ex = ZMatrix::Zero(nao, nao);
    std::vector<double> X, W;
    quad(0, ti, X, W);
    for (size_t q = 0; q < X.size(); q++) ex += W[q] * sf(X[q]) * C(1.1, 0.05) * g(X[q] - ti).cast<C>();
    quad(ti, beta, X, W);
    for (size_t q = 0; q < X.size(); q++) ex += W[q] * sf(X[q]) * C(1.1, 0.05) * g(X[q] - ti).cast<C>();
    ZMatrixMap got(res.data() + i * es, nao, nao);
    e5 = std::max(e5, (got - ex).norm());
    nrm = std::max(nrm, ex.norm());
    std::vector<C> r1(es, C(0.));
    D.tv_it_conv(i, tt, S, G, r1.data());
    rowmis = std::max(rowmis, (ZMatrixMap(r1.data(), nao, nao) - got).norm());
  }
  R.tv_it_conv = e5;
  R.tv_it_conv_ref = nrm;
  R.tv_it_conv_row = rowmis;
  // les_it_int: -i int Sigma^tv(t,tau) G^vt(tau,T), G^vt(tau,T) = -xi [G^tv(T,beta-tau)]^dagger; returned transposed
  {
    int T = 3;
    std::vector<C> Q((T + 1) * es, C(0.));
    D.les_it_int(T, S, G, Q.data());
    double e6 = 0, n6 = 0, m6 = 0;
    C pref[4] = {C(1, 0), C(1.1, 0.05), C(1.2, 0.1), C(1.3, 0.15)};
    C gpref = C(1. - 0.1 * T, 0.2 * T);
    for (int t = 0; t <= T; t++) {
      std::vector<double> X, W;
      quad(0, beta, X, W);
      ZMatrix ex = ZMatrix::Zero(nao, nao);
      for (size_t q = 0; q < X.size(); q++) {
        ZMatrix Gvt = -double(xi) * (gf(beta - X[q]) * gpref).adjoint();
        ex += W[q] * C(0, -1) * sf(X[q]) * pref[t] * Gvt;
      }
      ZMatrix got = ZMatrixMap(Q.data() + t * es, nao, nao).transpose();
      e6 = std::max(e6, (got - ex).norm());
      n6 = std::max(n6, ex.norm());
      std::vector<C> r1(es, C(0.));
      D.les_it_int(t, T, S, G, r1.data());
      m6 = std::max(m6, (ZMatrixMap(r1.data(), nao, nao) - ZMatrixMap(Q.data() + t * es, nao, nao)).norm());
    }
    R.les_it_int = e6;
    R.les_it_int_ref = n6;
    R.les_it_int_row = m6;
  }
  // gamma_integral at tstp=0 (only the mixed term is non-zero): -i [G*Sigma]^<(0,0) mixed part
  //   = -i * (-i int G^tv(0,tau) Sigma^vt(tau,0)),  Sigma^vt(tau,0) = -xi [Sigma^tv(0,beta-tau)]^dagger
  {
    std::vector<C> gi(es);
    Integration::Integrator I(k);
    D.gamma_integral(0, G, S, 0.01, I, gi.data());
    std::vector<double> X, W;
    quad(0, beta, X, W);
    ZMatrix ex = ZMatrix::Zero(nao, nao);
    for (size_t q = 0; q < X.size(); q++) {
      ZMatrix Svt = -double(xi) * (sf(beta - X[q])).adjoint();
      ex += W[q] * gf(X[q]) * Svt;
    }
    ex *= C(0, -1) * C(0, -1);
    R.gamma_mixed = (ZMatrixMap(gi.data(), nao, nao) - ex).norm();
    R.gamma_mixed_ref = ex.norm();
  }
  // gamma_integral_mat = int_0^beta G^M(beta-tau) D^M(tau)
  {
    herm_matrix_hodlr Dl(nt, r, nlvl, 1e-10, nao, nao, xi, k);
    DMatrix Hb(nao, nao);
    if (nao == 1) Hb << 1.3;
    else Hb << 1.3, -0.2, -0.2, 0.9;
    FreeG gb(Hb, beta, xi);
    for (int i = 0; i < r; i++) Dl.set_mat(i, gb(dlr.it0B(i)));
    std::vector<C> gm(es);
    Integration::Integrator I(k);
    D.gamma_integral_mat(G, Dl, 0.01, I, gm.data());
    std::vector<double> X, W;
    quad(0, beta, X, W);
    DMatrix ex = DMatrix::Zero(nao, nao);
    for (size_t q = 0; q < X.size(); q++) ex += W[q] * g(beta - X[q]) * gb(X[q]);
    R.gamma_mat = (ZMatrixMap(gm.data(), nao, nao) - ex.cast<C>()).norm();
    R.gamma_mat_ref = ex.norm();
  }
  // dyson_mat with Sigma = V gb V^T: exact = system block of the free G of [[H,V],[V^T,Hb]]
  {
    DMatrix Hb(nao, nao), V(nao, nao);
    if (nao == 1) {
      Hb << 1.3;
      V << 0.4;
    } else {
      Hb << 1.3, -0.2, -0.2, 0.9;
      V << 0.4, 0.1, -0.15, 0.3;
    }
    FreeG gb(Hb, beta, xi);
    DMatrix Hfull(2 * nao, 2 * nao);
    Hfull << H, V, V.transpose(), Hb;
    FreeG gfull(Hfull, beta, xi);
    herm_matrix_hodlr Sg(nt, r, nlvl, 1e-10, nao, nao, xi, k);
    for (int i = 0; i < r; i++) Sg.set_mat(i, DMatrix(V * gb(dlr.it0B(i)) * V.transpose()));
    herm_matrix_hodlr G2(nt, r, nlvl, 1e-10, nao, nao, xi, k);
    for (int i = 0; i < r; i++) G2.set_mat(i, g(dlr.it0B(i)));
    std::vector<double> old(G2.mat(), G2.mat() + r * es);
    double ret = D.dyson_mat(G2, 0.0, H, Sg, false, 1.0);
    double e7 = 0, diff = 0;
    for (int i = 0; i < r; i++) {
      DMatrix ex = gfull(dlr.it0B(i)).block(0, 0, nao, nao);
      e7 = std::max(e7, (DMatrixMap(G2.matptr(i), nao, nao) - ex).norm());
      for (int a = 0; a < es; a++) diff += pow(G2.mat()[i * es + a] - old[i * es + a], 2);
    }
    R.dyson_mat = e7;
    R.dyson_mat_ret_mismatch = std::abs(ret - sqrt(diff));
    herm_matrix_hodlr G3(nt, r, nlvl, 1e-10, nao, nao, xi, k);
    double mu = 0.25;
    D.dyson_mat(G3, mu, H, Sg, false, 1.0);
    DMatrix Hf2 = Hfull;
    Hf2.block(0, 0, nao, nao) -= mu * DMatrix::Identity(nao, nao);
    FreeG gf2(Hf2, beta, xi);
    double e8 = 0;
    for (int i = 0; i < r; i++) e8 = std::max(e8, (DMatrixMap(G3.matptr(i), nao, nao) - gf2(dlr.it0B(i)).block(0, 0, nao, nao)).norm());
    R.dyson_mat_mu = e8;
    herm_matrix_hodlr G4(nt, r, nlvl, 1e-10, nao, nao, xi, k);
    D.dyson_mat(G4, 0.0, H, Sg, false, 0.3);
    double e9 = 0;
    for (int i = 0; i < r; i++) e9 = std::max(e9, (DMatrixMap(G4.matptr(i), nao, nao) - 0.3 * DMatrixMap(G2.matptr(i), nao, nao)).norm());
    R.dyson_mat_alpha = e9;
  }
  return R;
}

const DlrMetrics &metrics(int xi, int nao) {
  static std::map<std::pair<int, int>, DlrMetrics> cache;
  auto key = std::make_pair(xi, nao);
  auto it = cache.find(key);
  if (it == cache.end()) it = cache.emplace(key, compute(xi, nao)).first;
  return it->second;
}

}  // namespace

// ---------------------------------------------------------------- statistics-independent checks
class DlrBoth : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(DlrBoth, Accessors) {
  auto [xi, nao] = GetParam();
  const auto &m = metrics(xi, nao);
  EXPECT_LT(m.ipmat_asym, 1e-11);
  EXPECT_LT(m.get_mat_tau, 1e-11);
  EXPECT_LT(m.density_matrix, 1e-13);
  EXPECT_LT(m.mat_reversed, 1e-11);
  EXPECT_LT(m.tv_tau, 1e-9);
  EXPECT_LT(m.tv_reversed, 1e-9);
}

TEST_P(DlrBoth, LesItInt) {
  auto [xi, nao] = GetParam();
  const auto &m = metrics(xi, nao);
  EXPECT_LT(m.les_it_int / m.les_it_int_ref, 1e-11);
  EXPECT_LT(m.les_it_int_row / m.les_it_int_ref, 1e-12);
}

TEST_P(DlrBoth, TvItConvSingleRowConsistent) {
  auto [xi, nao] = GetParam();
  const auto &m = metrics(xi, nao);
  EXPECT_LT(m.tv_it_conv_row / m.tv_it_conv_ref, 1e-12);
}

TEST_P(DlrBoth, GammaIntegralMat) {
  auto [xi, nao] = GetParam();
  const auto &m = metrics(xi, nao);
  EXPECT_LT(m.gamma_mat / m.gamma_mat_ref, 1e-10);
}

TEST_P(DlrBoth, DysonMatExactHybridised) {
  auto [xi, nao] = GetParam();
  const auto &m = metrics(xi, nao);
  EXPECT_LT(m.dyson_mat, 1e-12);
  EXPECT_LT(m.dyson_mat_ret_mismatch, 1e-12) << "returned error must be ||G_new - G_old||";
  EXPECT_LT(m.dyson_mat_mu, 1e-12);
  EXPECT_LT(m.dyson_mat_alpha, 1e-14);
}

INSTANTIATE_TEST_SUITE_P(Dlr, DlrBoth,
                         ::testing::Values(std::make_tuple(-1, 1), std::make_tuple(-1, 2), std::make_tuple(1, 1), std::make_tuple(1, 2)));

// ---------------------------------------------------------------- statistics-dependent (B1, B2)
class DlrFermion : public ::testing::TestWithParam<int> {};
TEST_P(DlrFermion, TvItConvVsQuadrature) {
  const auto &m = metrics(-1, GetParam());
  EXPECT_LT(m.tv_it_conv / m.tv_it_conv_ref, 1e-11);
}
TEST_P(DlrFermion, GammaIntegralMixedTerm) {
  const auto &m = metrics(-1, GetParam());
  EXPECT_LT(m.gamma_mixed / m.gamma_mixed_ref, 1e-11);
}
INSTANTIATE_TEST_SUITE_P(Dlr, DlrFermion, ::testing::Values(1, 2));

// B1 (fixed in WP3; was: bosonic tv_it_conv returned exactly -reference).
TEST(DlrBoson, TvItConvVsQuadrature) {
  hn_test::run_isolated([] {
    for (int nao : {1, 2}) {
      const auto &m = metrics(+1, nao);
      EXPECT_LT(m.tv_it_conv / m.tv_it_conv_ref, 1e-11) << "nao=" << nao;
    }
  });
}
// B2 (fixed in WP3; was: bosonic gamma_integral mixed term had the fermionic sign).
TEST(DlrBoson, GammaIntegralMixedTerm) {
  hn_test::run_isolated([] {
    for (int nao : {1, 2}) {
      const auto &m = metrics(+1, nao);
      EXPECT_LT(m.gamma_mixed / m.gamma_mixed_ref, 1e-11) << "nao=" << nao;
    }
  });
}

// ---------------------------------------------------------------- initGMConvTensor dlr consistency (WP3, WP6a)
// initGMConvTensor needs a dlr_info with the same xi (Phi carries it) and r as G. Tier 2 since WP6a (was a Debug-only
// assert): h_nessi::precondition_error in every build.
TEST(GMConvTensor, DlrXiOrRankMismatchThrows) {
  int rf = 0, rb = 0;
  dlr_info dlrf(rf, 50., 1e-12, 5., 1, -1), dlrb(rb, 50., 1e-12, 5., 1, +1);
  herm_matrix_hodlr Gf(4, rf, 1, 1e-12, 1, 1, -1, 2), Gb(4, rb, 1, 1e-12, 1, 1, +1, 2);
  Gf.initGMConvTensor(dlrf);  // consistent: fine in every build
  Gb.initGMConvTensor(dlrb);
  EXPECT_THROW(Gb.initGMConvTensor(dlrf), h_nessi::precondition_error);
  EXPECT_THROW(Gf.initGMConvTensor(dlrb), h_nessi::precondition_error);
  herm_matrix_hodlr Gr(4, rf - 1, 1, 1e-12, 1, 1, -1, 2);  // r mismatch
  try {
    Gr.initGMConvTensor(dlrf);
    FAIL() << "no exception";
  } catch (const h_nessi::precondition_error &e) {
    EXPECT_NE(std::string(e.what()).find("h_nessi::herm_matrix_hodlr::initGMConvTensor: dlr.xi()==sig_ && dlr.r()==r_ violated"),
              std::string::npos) << e.what();
  }
}

// ---------------------------------------------------------------- dyson_mat buffers and fixHam (B4, H29; fixed in WP2)
namespace {
// Small 2-orbital problem shared by the dyson_mat buffer tests: Sigma^M = 0.3 * G0_bath^M (static hybridisation).
struct MatSetup {
  int nao, xi, nt = 10, k = 2, r = 0;
  double beta = 5.;
  dlr_info dlr;
  function H;
  DMatrix h0;
  herm_matrix_hodlr S;
  MatSetup(int nao_, int xi_)
      : nao(nao_), xi(xi_), dlr(r, 50., 1e-12, beta, nao_, xi_), H(nt, nao_, nao_), h0(nao_, nao_),
        S(nt, r, 1, 1e-12, nao_, nao_, xi_, k) {
    ZMatrix hb(nao, nao);
    if (nao == 1) {
      h0 << 0.4;
      hb << 1.0;
    } else {
      h0 << 0.4, 0.2, 0.2, -0.3;
      hb << 1.0, 0.3, 0.3, -0.8;
    }
    H.get_map(-1) = h0.cast<C>();
    herm_matrix_hodlr tmp(nt, r, 1, 1e-12, nao, nao, xi, k);
    dyson D0(nt, nao, k, dlr, true, false);
    D0.green_from_H(tmp, 0., hb, 0.01, k, true);
    for (int i = 0; i < r; i++) S.set_mat(i, DMatrix(0.3 * DMatrixMap(tmp.matptr(i), nao, nao)));
  }
  herm_matrix_hodlr newG() { return herm_matrix_hodlr(nt, r, 1, 1e-12, nao, nao, xi, k); }
  double reldiff(herm_matrix_hodlr &A, herm_matrix_hodlr &B) {
    int es = nao * nao;
    return (DMatrixMap(A.mat(), r, es) - DMatrixMap(B.mat(), r, es)).norm() / DMatrixMap(A.mat(), r, es).norm();
  }
};
}  // namespace

// With identical inputs, a second call with fixHam=true must give the same G^M as a second call with fixHam=false
// (G0 is unchanged, only "not recomputed"). Formerly KnownBug_B4 (agent_scratch/51/t_mat.cpp): rel. difference 1.10.
TEST(DysonMat, FixHamSecondCall) {
  MatSetup P(2, -1);
  double mu = 0.;
  herm_matrix_hodlr Ga = P.newG(), Gb = P.newG();
  dyson D(P.nt, P.nao, P.k, P.dlr, true, false);
  D.dyson_mat(Ga, mu, P.H, P.S, false, 1.0);
  D.dyson_mat(Ga, mu, P.H, P.S, false, 1.0);
  dyson D2(P.nt, P.nao, P.k, P.dlr, true, false);
  D2.dyson_mat(Gb, mu, P.H, P.S, false, 1.0);
  D2.dyson_mat(Gb, mu, P.H, P.S, true, 1.0);
  EXPECT_LT(P.reldiff(Ga, Gb), 1e-12) << "relative difference fixHam=true vs false";
}

// fixHam=true reuses the G0 saved in the dyson object by the last fixHam=false call, for both overloads, both
// statistics, with mixing, on a different G, after initGMConvTensor of G, and after a call with a different Sigma.
class DysonMatFixHam : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(DysonMatFixHam, ReusesSavedG0) {
  auto [xi, nao] = GetParam();
  MatSetup P(nao, xi);
  double mu = 0.15;
  DMatrix hm = P.h0;
  // reference: fixHam=false throughout
  herm_matrix_hodlr Gref = P.newG();
  dyson Dref(P.nt, P.nao, P.k, P.dlr, true, false);
  for (int it = 0; it < 3; it++) Dref.dyson_mat(Gref, mu, P.H, P.S, false, 0.5);
  for (int ovl = 0; ovl < 2; ovl++) {
    herm_matrix_hodlr G0 = P.newG(), G = P.newG();
    dyson D(P.nt, P.nao, P.k, P.dlr, true, false);
    // the saving call is made on another G, and G's conv tensor is then initialised (it must not be used as G0)
    if (ovl == 0) D.dyson_mat(G0, mu, P.H, P.S, false, 1.0);
    else D.dyson_mat(G0, mu, hm, P.S, false, 1.0);
    G0.initGMConvTensor(P.dlr);
    // a fixHam=true call with a different Sigma must not disturb the saved G0 either
    herm_matrix_hodlr S2 = P.newG();
    for (int i = 0; i < P.r; i++) S2.set_mat(i, DMatrix(2.0 * DMatrixMap(P.S.matptr(i), nao, nao)));
    D.dyson_mat(G0, mu, P.H, S2, true, 1.0);
    for (int it = 0; it < 3; it++) {
      if (ovl == 0) D.dyson_mat(G, mu, P.H, P.S, true, 0.5);
      else D.dyson_mat(G, mu, hm, P.S, true, 0.5);
    }
    EXPECT_LT(P.reldiff(Gref, G), 1e-12) << "overload " << ovl;
  }
}
INSTANTIATE_TEST_SUITE_P(Dlr, DysonMatFixHam,
                         ::testing::Values(std::make_tuple(-1, 1), std::make_tuple(-1, 2), std::make_tuple(1, 1), std::make_tuple(1, 2)));

// H29: dyson_mat must not write to Sigma (mat, every tv row and tv_trans, conv tensor) nor to G.GMConvTensMatrix_.
TEST_P(DysonMatFixHam, LeavesSigmaAndConvTensorUnchanged) {
  auto [xi, nao] = GetParam();
  MatSetup P(nao, xi);
  ExpSum sf(nao, 5 + xi);
  for (int t = 0; t < P.nt; t++)  // tv holds nt rows (t = 0..nt-1)
    for (int i = 0; i < P.r; i++) P.S.set_tv(t, i, ZMatrix(sf(P.dlr.it0B(i)) * C(1. + 0.1 * t, -0.03 * t)));
  P.S.initGMConvTensor(P.dlr);
  herm_matrix_hodlr G = P.newG();
  dyson D(P.nt, P.nao, P.k, P.dlr, true, false);
  D.dyson_mat(G, 0., P.H, P.S, false, 1.0);
  G.initGMConvTensor(P.dlr);
  std::vector<double> smat(P.S.mat(), P.S.mat() + P.r * nao * nao);
  std::vector<C> stv = P.S.tv().data_vector(), stvt = P.S.tv().data_trans_vector();
  DMatrix sconv = P.S.GMConvTensMatrix(), gconv = G.GMConvTensMatrix();
  DMatrix hm = P.h0;
  for (bool fix : {false, true}) {
    D.dyson_mat(G, 0., P.H, P.S, fix, 0.5);
    D.dyson_mat(G, 0., hm, P.S, fix, 0.5);
  }
  EXPECT_TRUE(std::equal(smat.begin(), smat.end(), P.S.mat())) << "Sigma^M changed";
  EXPECT_TRUE(stv == P.S.tv().data_vector()) << "Sigma^tv changed";
  EXPECT_TRUE(stvt == P.S.tv().data_trans_vector()) << "Sigma^tv tv_trans changed";
  EXPECT_TRUE(sconv == P.S.GMConvTensMatrix()) << "Sigma conv tensor changed";
  EXPECT_TRUE(gconv == G.GMConvTensMatrix()) << "G conv tensor changed";
}

// memory_usage_bytes (printed by print_memory_usage) must count every heap buffer of dyson with its real size
// (B32, fixed in WP2). The sizes below are computed independently from nt, nao, k, r and profile.
TEST(DysonMemory, CountsAllBuffers) {
  for (int nt : {37, 38})
    for (bool prof : {false, true}) {
      int nao = 2, es = 4, k = 4, r = 0;
      dlr_info dlr(r, 50., 1e-12, 5., nao, -1);
      dyson D(nt, nao, k, dlr, true, prof);
      std::vector<double> m = D.memory_usage_bytes();
      ASSERT_EQ(m.size(), 7u);
      double Db = sizeof(double), Zb = sizeof(C);
      double dlrbuf = Zb * 2 * r * es + Db * 2 * r * es;                   // NTauTmp_, NTauTmp2_, DNTauTmp_, DNTauTmp2_
      double g0 = Db * r * es, g0conv = Db * r * r * es;                    // DG0_, DG0ConvMat_
      double matmul = Zb * 2 * ((nt + 1) / 2) * nao;                        // epsnao_tmp_, epsnao_tmp_2_
      double solver = Zb * (2 * (nt + 1) * es + (k + 1) * k * es + 2 * es);  // Q_, X_, M_, iden_, bound_
      double tim = Db * (prof ? nt : 0) * 25 + Db * (nt + 1);               // timing + Hermiticity diagnostic (D-11)
      EXPECT_EQ(m[0], dlrbuf);
      EXPECT_EQ(m[1], g0);
      EXPECT_EQ(m[2], g0conv);
      EXPECT_EQ(m[3], matmul);
      EXPECT_EQ(m[4], solver);
      EXPECT_EQ(m[5], tim);
      EXPECT_EQ(m[6], dlrbuf + g0 + g0conv + matmul + solver + tim) << "nt=" << nt << " profile=" << prof;
    }
}

// ---------------------------------------------------------------- B3: vector eval_point (fixed in WP-N)
// Formerly KnownBug_B3_*: the vector eval_point overloads wrote into an empty `taurel` and evaluated every point at
// taurel[0] (child SIGSEGV).
namespace {
void b3_setup(int xi, int nao, dlr_info *&dlr, herm_matrix_hodlr *&G, FreeG *&g, ExpSum *&gf, double &beta) {
  beta = 10.;
  int r = 0;
  dlr = new dlr_info(r, 100., 1e-12, beta, nao, xi);
  DMatrix H(nao, nao);
  if (nao == 1) H << 0.7;
  else H << 0.7, 0.3, 0.3, 1.4;
  g = new FreeG(H, beta, xi);
  gf = new ExpSum(nao, 5);
  G = new herm_matrix_hodlr(16, r, 2, 1e-10, nao, nao, xi, 3);
  for (int i = 0; i < r; i++) G->set_mat(i, (*g)(dlr->it0B(i)));
  for (int i = 0; i < r; i++) G->set_tv(1, i, ZMatrix((*gf)(dlr->it0B(i))));
}
void b3_cleanup(dlr_info *dlr, herm_matrix_hodlr *G, FreeG *g, ExpSum *gf) {
  delete dlr;
  delete G;
  delete g;
  delete gf;
}
// tau points in [0, beta]: both endpoints, points within 1e-14..1e-6 of each endpoint, an equidistant grid
// (NESSi's grid, 0..beta in ntau steps) and some random interior points.
DColVector b3_taus(double beta) {
  std::vector<double> t = {0., beta};
  for (double d : {1e-14, 1e-12, 1e-9, 1e-6, 1e-3}) {
    t.push_back(d * beta);
    t.push_back(beta - d * beta);
  }
  int ntau = 50;
  for (int i = 0; i <= ntau; i++) t.push_back(i * beta / ntau);
  std::mt19937 rg(11);
  std::uniform_real_distribution<double> u(0., beta);
  for (int i = 0; i < 20; i++) t.push_back(u(rg));
  DColVector taus(t.size());
  for (size_t i = 0; i < t.size(); i++) taus(i) = t[i];
  return taus;
}
}  // namespace

// The array accessors must agree pointwise with the scalar accessors (same arithmetic, so to roundoff),
// and with the analytic function to the DLR accuracy; for both statistics and N_o = 1, 2.
class DlrArrayP : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(DlrArrayP, MatTauArrayEqualsPointwise) {
  int xi = std::get<0>(GetParam()), nao = std::get<1>(GetParam()), es = nao * nao;
  dlr_info *dlr;
  herm_matrix_hodlr *G;
  FreeG *g;
  ExpSum *gf;
  double beta;
  b3_setup(xi, nao, dlr, G, g, gf, beta);
  DColVector taus = b3_taus(beta);
  int nt = taus.size();
  // all four output overloads
  std::vector<double> outd(nt * es);
  std::vector<C> outz(nt * es);
  DMatrix outD(nt, es);
  ZMatrix outZ(nt, es);
  G->get_mat_tau_array(taus, *dlr, outd.data());
  G->get_mat_tau_array(taus, *dlr, outz.data());
  G->get_mat_tau_array(taus, *dlr, outD);
  G->get_mat_tau_array(taus, *dlr, outZ);
  double maxdiff = 0, maxexact = 0;
  for (int q = 0; q < nt; q++) {
    DMatrix M(nao, nao);
    G->get_mat_tau(taus(q), *dlr, M);
    double sc = 1. + M.norm();
    maxdiff = std::max(maxdiff, (DMatrixMap(outd.data() + q * es, nao, nao) - M).norm() / sc);
    maxdiff = std::max(maxdiff, (ZMatrixMap(outz.data() + q * es, nao, nao) - M.cast<C>()).norm() / sc);
    maxdiff = std::max(maxdiff, (DMatrixMap(outD.data() + q * es, nao, nao) - M).norm() / sc);
    maxdiff = std::max(maxdiff, (ZMatrixMap(outZ.data() + q * es, nao, nao) - M.cast<C>()).norm() / sc);
    maxexact = std::max(maxexact, (DMatrixMap(outd.data() + q * es, nao, nao) - (*g)(taus(q))).norm());
  }
  EXPECT_LT(maxdiff, 1e-14) << "xi=" << xi << " nao=" << nao;
  EXPECT_LT(maxexact, 1e-10) << "xi=" << xi << " nao=" << nao;
  b3_cleanup(dlr, G, g, gf);
}

TEST_P(DlrArrayP, TvTauArrayEqualsPointwise) {
  int xi = std::get<0>(GetParam()), nao = std::get<1>(GetParam()), es = nao * nao;
  dlr_info *dlr;
  herm_matrix_hodlr *G;
  FreeG *g;
  ExpSum *gf;
  double beta;
  b3_setup(xi, nao, dlr, G, g, gf, beta);
  DColVector taus = b3_taus(beta);
  int nt = taus.size();
  std::vector<C> outz(nt * es);
  ZMatrix outZ(nt, es);
  G->get_tv_tau_array(1, taus, *dlr, outz.data());
  G->get_tv_tau_array(1, taus, *dlr, outZ);
  double maxdiff = 0, maxexact = 0;
  for (int q = 0; q < nt; q++) {
    ZMatrix M(nao, nao);
    G->get_tv_tau(1, taus(q), *dlr, M);
    double sc = 1. + M.norm();
    maxdiff = std::max(maxdiff, (ZMatrixMap(outz.data() + q * es, nao, nao) - M).norm() / sc);
    maxdiff = std::max(maxdiff, (ZMatrixMap(outZ.data() + q * es, nao, nao) - M).norm() / sc);
    maxexact = std::max(maxexact, (ZMatrixMap(outz.data() + q * es, nao, nao) - (*gf)(taus(q))).norm() / sc);
  }
  EXPECT_LT(maxdiff, 1e-14) << "xi=" << xi << " nao=" << nao;
  EXPECT_LT(maxexact, 1e-9) << "xi=" << xi << " nao=" << nao;
  b3_cleanup(dlr, G, g, gf);
}

INSTANTIATE_TEST_SUITE_P(XiNao, DlrArrayP,
                         ::testing::Combine(::testing::Values(-1, 1), ::testing::Values(1, 2)));

// The original WP0 reproducers (formerly KnownBug_B3_*), kept as plain tests.
TEST(DlrArray, GetMatTauArray) {
  int nao = 2;
  dlr_info *dlr;
  herm_matrix_hodlr *G;
  FreeG *g;
  ExpSum *gf;
  double beta;
  b3_setup(-1, nao, dlr, G, g, gf, beta);
  DColVector taus(4);
  taus << 0.0, 1.0, 2.5, 7.0;
  std::vector<double> out(4 * nao * nao);
  G->get_mat_tau_array(taus, *dlr, out.data());
  for (int q = 0; q < 4; q++)
    EXPECT_LT((DMatrixMap(out.data() + q * nao * nao, nao, nao) - (*g)(taus(q))).norm(), 1e-11) << "tau=" << taus(q);
  b3_cleanup(dlr, G, g, gf);
}

TEST(DlrArray, GetTvTauArray) {
  int nao = 2;
  dlr_info *dlr;
  herm_matrix_hodlr *G;
  FreeG *g;
  ExpSum *gf;
  double beta;
  b3_setup(-1, nao, dlr, G, g, gf, beta);
  DColVector taus(3);
  taus << 0.5, 3.0, 9.0;
  std::vector<C> out(3 * nao * nao);
  G->get_tv_tau_array(1, taus, *dlr, out.data());
  for (int q = 0; q < 3; q++)
    EXPECT_LT((ZMatrixMap(out.data() + q * nao * nao, nao, nao) - (*gf)(taus(q))).norm(), 1e-9) << "tau=" << taus(q);
  b3_cleanup(dlr, G, g, gf);
}
