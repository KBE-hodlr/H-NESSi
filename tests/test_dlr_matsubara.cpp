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

// B1: bosonic tv_it_conv returns exactly -reference.
TEST(DlrBoson, KnownBug_B1_TvItConvVsQuadrature) {
  hn_test::run_isolated([] {
    for (int nao : {1, 2}) {
      const auto &m = metrics(+1, nao);
      EXPECT_LT(m.tv_it_conv / m.tv_it_conv_ref, 1e-11) << "nao=" << nao;
    }
  });
}
// B2: bosonic gamma_integral mixed term has the fermionic sign.
TEST(DlrBoson, KnownBug_B2_GammaIntegralMixedTerm) {
  hn_test::run_isolated([] {
    for (int nao : {1, 2}) {
      const auto &m = metrics(+1, nao);
      EXPECT_LT(m.gamma_mixed / m.gamma_mixed_ref, 1e-11) << "nao=" << nao;
    }
  });
}

// ---------------------------------------------------------------- B4: dyson_mat(fixHam=true)
// Contract test (agent_scratch/51/t_mat.cpp): with identical inputs, a second call with fixHam=true must give the
// same G^M as a second call with fixHam=false (G0 is unchanged, only "not recomputed").
TEST(DysonMat, KnownBug_B4_FixHamSecondCall) {
  hn_test::run_isolated([] {
    int nao = 2, xi = -1, nt = 10, k = 2;
    double beta = 5., mu = 0.;
    int r = 0;
    dlr_info dlr(r, 50., 1e-12, beta, nao, xi);
    function H(nt, nao, nao);
    DMatrix h0(nao, nao);
    h0 << 0.4, 0.2, 0.2, -0.3;
    H.get_map(-1) = h0.cast<C>();
    herm_matrix_hodlr S(nt, r, 1, 1e-12, nao, nao, xi, k);
    {
      herm_matrix_hodlr tmp(nt, r, 1, 1e-12, nao, nao, xi, k);
      dyson D0(nt, nao, k, dlr, true, false);
      ZMatrix hb(nao, nao);
      hb << 1.0, 0.3, 0.3, -0.8;
      D0.green_from_H(tmp, 0., hb, 0.01, k, true);
      for (int i = 0; i < r; i++) S.set_mat(i, DMatrix(0.3 * DMatrixMap(tmp.matptr(i), nao, nao)));
    }
    herm_matrix_hodlr Ga(nt, r, 1, 1e-12, nao, nao, xi, k), Gb(nt, r, 1, 1e-12, nao, nao, xi, k);
    dyson D(nt, nao, k, dlr, true, false);
    D.dyson_mat(Ga, mu, H, S, false, 1.0);
    D.dyson_mat(Ga, mu, H, S, false, 1.0);
    dyson D2(nt, nao, k, dlr, true, false);
    D2.dyson_mat(Gb, mu, H, S, false, 1.0);
    D2.dyson_mat(Gb, mu, H, S, true, 1.0);
    double d = (DMatrixMap(Ga.mat(), r, nao * nao) - DMatrixMap(Gb.mat(), r, nao * nao)).norm() /
               DMatrixMap(Ga.mat(), r, nao * nao).norm();
    EXPECT_LT(d, 1e-12) << "relative difference fixHam=true vs false";
  });
}

// ---------------------------------------------------------------- B3: vector eval_point (segfault)
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
}  // namespace

TEST(DlrArray, KnownBug_B3_GetMatTauArray) {
  hn_test::run_isolated([] {
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
  });
}

TEST(DlrArray, KnownBug_B3_GetTvTauArray) {
  hn_test::run_isolated([] {
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
  });
}
