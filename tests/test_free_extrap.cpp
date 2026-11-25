// dyson class helpers (doc 23): green_from_H / green_from_H_dm, extrapolate, gamma_integral.
// Ported from agent_scratch/23/ta.cpp, tb.cpp, tc.cpp.
#include <gtest/gtest.h>

#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>

#include "h_nessi/dyson.hpp"
#include "common/test_util.hpp"

using namespace h_nessi;
namespace {
const cplx II(0, 1);
}

// ------------------------------------------------------------------------------------------------ (a) green_from_H
// Exact free G for a time-independent 2x2 H (Eqs. 22-24 with the corrected prefactor, ISSUES P10):
//   G^R = -i U(t)U(t')^dag, G^< = -i xi U(t) rho U(t')^dag, G^tv = -i xi U(t) rho e^{tau E}, G^M = -e^{-tau E}(1 + xi rho)
namespace {
struct FreeCase {
  int xi, cplxH;
  double beta = 5, lambda = 100, eps = 1e-12, h = 0.05;
  int nao = 2, k = 3, nt = 40, r = 0;
  double mu;
  ZMatrix H, E, rho;
  explicit FreeCase(int xi_, int c) : xi(xi_), cplxH(c) {
    H.resize(nao, nao);
    H << 0.7, 0.3, 0.3, 1.4;
    if (cplxH) {
      H(0, 1) = cplx(0.3, 0.2);
      H(1, 0) = cplx(0.3, -0.2);
    }
    mu = xi == -1 ? 0.9 : 0.0;  // bosons need H - mu > 0
    E = H - mu * ZMatrix::Identity(nao, nao);
    rho = (ZMatrix((beta * E).exp()) - double(xi) * ZMatrix::Identity(nao, nao)).inverse();
  }
  ZMatrix U(double t) const { return ZMatrix((-II * t * E).exp()); }
  ZMatrix ex(double tau) const { return ZMatrix((tau * E).exp()); }
  ZMatrix GR(int m, int n) const { return ZMatrix(-II * U(m * h) * U(n * h).adjoint()); }
  ZMatrix GL(int n, int m) const { return ZMatrix(-II * double(xi) * U(n * h) * rho * U(m * h).adjoint()); }
  ZMatrix GTV(int n, double tau) const { return ZMatrix(-II * double(xi) * U(n * h) * rho * ex(tau)); }
  ZMatrix GM(double tau) const { return ZMatrix(-ex(-tau) * (ZMatrix::Identity(nao, nao) + double(xi) * rho)); }
};

void check_green_from_H(int xi, int cplxH) {
  FreeCase c(xi, cplxH);
  dlr_info dlr(c.r, c.lambda, c.eps, c.beta, c.nao, xi);
  herm_matrix_hodlr G(c.nt, c.r, 2, 1e-12, c.nao, c.nao, xi, c.k);
  dyson D(c.nt, c.nao, c.k, dlr, true, false);
  D.green_from_H(G, c.mu, c.H, c.h, c.k, true);
  double eR = 0, eL = 0, eTV = 0, eM = 0;
  ZMatrix M(c.nao, c.nao);
  DMatrix DM(c.nao, c.nao);
  for (int m = 0; m <= c.k; m++)
    for (int n = 0; n <= m; n++) {
      G.get_ret_curr(m, n, M);
      eR = std::max(eR, (M - c.GR(m, n)).norm());
      G.get_les_curr(n, m, M);
      eL = std::max(eL, (M - c.GL(n, m)).norm());
    }
  for (int n = 0; n <= c.k; n++)
    for (int i = 0; i < c.r; i++) {
      double tau = dlr.it0B(i);
      G.get_tv(n, i, M);
      eTV = std::max(eTV, (M - c.GTV(n, tau)).norm() / c.GTV(n, tau).norm());
    }
  EXPECT_LT(eR, 1e-13);
  EXPECT_LT(eL, 1e-13);
  EXPECT_LT(eTV, 1e-12);
  ZMatrix rho0(c.nao, c.nao);
  G.density_matrix(0, dlr, rho0);
  EXPECT_LT((rho0 - c.rho).norm(), 1e-13) << "density_matrix(0)";
  if (!cplxH) {  // G^M is stored real (ISSUES D1): only meaningful for real H
    for (int i = 0; i < c.r; i++) {
      G.get_mat(i, DM);
      eM = std::max(eM, (DM - c.GM(dlr.it0B(i)).real()).norm());
    }
    EXPECT_LT(eM, 1e-13);
    ZMatrix rhoM(c.nao, c.nao);
    G.density_matrix(-1, dlr, rhoM);
    EXPECT_LT((rhoM - c.rho).norm(), 1e-13) << "density_matrix(-1)";
  }
}

// green_from_H_dm with rho = density_matrix(-1) (decision D-1: rho is the density matrix)
void check_green_from_H_dm(int xi) {
  FreeCase c(xi, 0);
  dlr_info dlr(c.r, c.lambda, c.eps, c.beta, c.nao, xi);
  herm_matrix_hodlr G(c.nt, c.r, 2, 1e-12, c.nao, c.nao, xi, c.k);
  dyson D(c.nt, c.nao, c.k, dlr, true, false);
  D.green_from_H(G, c.mu, c.H, c.h, c.k, true);
  ZMatrix rhoM(c.nao, c.nao), M(c.nao, c.nao);
  G.density_matrix(-1, dlr, rhoM);
  herm_matrix_hodlr G2(c.nt, c.r, 2, 1e-12, c.nao, c.nao, xi, c.k);
  D.green_from_H_dm(G2, c.mu, c.H, rhoM, c.h, c.k);
  double eL = 0, eTV = 0;
  for (int m = 0; m <= c.k; m++)
    for (int n = 0; n <= m; n++) {
      G2.get_les_curr(n, m, M);
      eL = std::max(eL, (M - c.GL(n, m)).norm());
    }
  for (int n = 0; n <= c.k; n++)
    for (int i = 0; i < c.r; i++) {
      G2.get_tv(n, i, M);
      eTV = std::max(eTV, (M - c.GTV(n, dlr.it0B(i))).norm() / c.GTV(n, dlr.it0B(i)).norm());
    }
  ZMatrix rr(c.nao, c.nao);
  G2.density_matrix(0, dlr, rr);
  EXPECT_LT(eL, 1e-13) << "xi=" << xi;
  EXPECT_LT(eTV, 1e-12) << "xi=" << xi;
  EXPECT_LT((rr - c.rho).norm(), 1e-13) << "xi=" << xi << " density_matrix(0) after green_from_H_dm";
}
}  // namespace

class GreenFromH : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(GreenFromH, ExactFree) {
  auto [xi, c] = GetParam();
  check_green_from_H(xi, c);
}
INSTANTIATE_TEST_SUITE_P(Free, GreenFromH,
                         ::testing::Values(std::make_tuple(-1, 0), std::make_tuple(-1, 1), std::make_tuple(1, 0), std::make_tuple(1, 1)));

TEST(GreenFromHdm, FermionExact) { check_green_from_H_dm(-1); }
TEST(GreenFromHdm, KnownBug_B10_BosonExact) {
  hn_test::run_isolated([] { check_green_from_H_dm(+1); });
}

// ------------------------------------------------------------------------------------------------ (b) extrapolate
// Predicted row T vs an analytic smooth G: O(h^{k+1}) (degree-k extrapolation from k+1 rows).
namespace {
const int NAO = 2;
ZMatrix Eps() {
  ZMatrix E(NAO, NAO);
  E << 0.5, cplx(0.2, 0.1), cplx(0.2, -0.1), -0.4;
  return E;
}
ZMatrix Bm() {
  ZMatrix B(NAO, NAO);
  B << 0.3, 0.8, 0.8, -0.6;
  return B;
}
ZMatrix W(double t) {
  ZMatrix K = Eps() * t + 0.7 * sin(1.3 * t) * Bm();
  return ZMatrix((-II * K).exp());
}
ZMatrix rho0() {
  ZMatrix R(NAO, NAO);
  R << 0.6, cplx(0.1, 0.05), cplx(0.1, -0.05), 0.3;
  return R;
}
ZMatrix xGR(double t, double tp) { return ZMatrix(-II * W(t) * W(tp).adjoint()); }
ZMatrix xGL(double t, double tp) { return ZMatrix(II * W(t) * rho0() * W(tp).adjoint()); }
ZMatrix xGTV(double t, double tau) { return ZMatrix(II * W(t) * rho0() * (std::exp(0.3 * tau) + 0.2 * std::exp(-0.5 * tau)) / 1.2); }

struct ExtrapErr { double R, L, TV, R2, L2; bool noop; };
ExtrapErr extrap_errors(int k, double h) {
  double beta = 10, Tphys = 2.0;
  int r = 0;
  dlr_info dlr(r, 100., 1e-12, beta, NAO, -1);
  int nt = (int)std::lround(Tphys / h) + 1;
  herm_matrix_hodlr G(nt, r, 1, 1e-14, NAO, NAO, -1, k), G2(nt, r, 1, 1e-14, NAO, NAO, -1, k);
  Integration::Integrator I(k);
  dyson D(nt, NAO, k, dlr, true, false);
  auto fill = [&](herm_matrix_hodlr &X, int T) {
    for (int tp = 0; tp <= T; tp++) {
      X.set_ret_curr(T, tp, xGR(T * h, tp * h));
      X.set_les_curr(tp, T, xGL(tp * h, T * h));
    }
    for (int i = 0; i < r; i++) X.set_tv(T, i, xGTV(T * h, dlr.it0B(i)));
  };
  for (int T = 0; T <= k; T++) {
    fill(G, T);
    fill(G2, T);
  }
  ExtrapErr e{0, 0, 0, 0, 0, true};
  ZMatrix M(NAO, NAO);
  while (G.tstp() < nt - 1) {
    G.update_blocks(I);
    G2.update_blocks(I);
    int T = G.tstp();
    D.extrapolate(G, I);
    D.extrapolate_2leg(G2, I);
    if (T == nt - 1) {
      for (int tp = 0; tp <= T; tp++) {
        G.get_ret_curr(T, tp, M);
        e.R = std::max(e.R, (M - xGR(T * h, tp * h)).norm());
        G.get_les_curr(tp, T, M);
        e.L = std::max(e.L, (M - xGL(tp * h, T * h)).norm());
        G2.get_ret_curr(T, tp, M);
        e.R2 = std::max(e.R2, (M - xGR(T * h, tp * h)).norm());
        G2.get_les_curr(tp, T, M);
        e.L2 = std::max(e.L2, (M - xGL(tp * h, T * h)).norm());
      }
      for (int i = 0; i < r; i++) {
        G.get_tv(T, i, M);
        e.TV = std::max(e.TV, (M - xGTV(T * h, dlr.it0B(i))).norm());
      }
      G.get_ret_curr(T, 0, M);
      ZMatrix before = M;
      D.extrapolate(G, I);  // second call must be a no-op
      G.get_ret_curr(T, 0, M);
      e.noop = (M - before).norm() == 0;
    }
    fill(G, T);
    fill(G2, T);
  }
  return e;
}
}  // namespace

class Extrapolate : public ::testing::TestWithParam<int> {};
TEST_P(Extrapolate, OrderKPlus1) {
  const int k = GetParam();
  ExtrapErr a = extrap_errors(k, 0.025), b = extrap_errors(k, 0.0125);
  const double want = k + 1 - 0.25;
  EXPECT_GT(hn_test::rate(a.R, b.R), want) << "ret";
  EXPECT_GT(hn_test::rate(a.L, b.L), want) << "les";
  EXPECT_GT(hn_test::rate(a.TV, b.TV), want) << "tv";
  EXPECT_EQ(a.R, a.R2) << "extrapolate_2leg ret must equal extrapolate";
  EXPECT_EQ(a.L, a.L2) << "extrapolate_2leg les must equal extrapolate";
  EXPECT_TRUE(a.noop && b.noop) << "second extrapolate call in the same step must be a no-op";
}
INSTANTIATE_TEST_SUITE_P(Extrap, Extrapolate, ::testing::Values(1, 2, 3, 4, 5));

// ------------------------------------------------------------------------------------------------ (c) gamma_integral
// gamma_integral(T) vs dense quadrature of -i[G*D]^<(T,T) = -i( int_0^T G^R D^< + G^< D^A  -i int_0^beta G^tv D^vt ).
namespace {
struct F {
  ZMatrix E, B, rho;
  double a, om, c1, c2;
  ZMatrix W(double t) const {
    ZMatrix K = E * t + a * sin(om * t) * B;
    return ZMatrix((-II * K).exp());
  }
  ZMatrix R(double t, double tp) const { return ZMatrix(-II * W(t) * W(tp).adjoint() * (1. + 0.2 * cos(t - tp))); }
  ZMatrix L(double t, double tp) const { return ZMatrix(II * W(t) * rho * W(tp).adjoint()); }
  double g(double tau) const { return (std::exp(c1 * tau) + 0.2 * std::exp(c2 * tau)) / 1.2; }
  ZMatrix TV(double t, double tau) const { return ZMatrix(II * W(t) * rho * g(tau)); }
};
template <class Fn> ZMatrix simpson(Fn f, double a, double b, int N = 4000) {
  ZMatrix s = ZMatrix::Zero(NAO, NAO);
  double hh = (b - a) / N;
  for (int i = 0; i <= N; i++) {
    double w = (i == 0 || i == N) ? 1 : (i % 2 ? 4 : 2);
    s += w * f(a + i * hh);
  }
  return ZMatrix(s * hh / 3.);
}
struct GammaRes { std::vector<double> err; double err_last; };
GammaRes gamma_errors(int xi, int k, double h) {
  double beta = 4;
  int r = 0;
  dlr_info dlr(r, 50., 1e-13, beta, NAO, xi);
  F g, d;
  g.E.resize(2, 2);
  g.E << 0.5, cplx(0.2, 0.1), cplx(0.2, -0.1), -0.4;
  g.B.resize(2, 2);
  g.B << 0.3, 0.8, 0.8, -0.6;
  g.rho.resize(2, 2);
  g.rho << 0.6, cplx(0.1, 0.05), cplx(0.1, -0.05), 0.3;
  g.a = 0.7; g.om = 1.3; g.c1 = 0.3; g.c2 = -0.5;
  d.E.resize(2, 2);
  d.E << -0.2, cplx(0.4, -0.3), cplx(0.4, 0.3), 0.9;
  d.B.resize(2, 2);
  d.B << 0.5, cplx(0, 0.2), cplx(0, -0.2), 0.1;
  d.rho.resize(2, 2);
  d.rho << 0.2, 0.15, 0.15, 0.7;
  d.a = 0.4; d.om = 0.9; d.c1 = -0.8; d.c2 = 0.4;
  auto ref = [&](double T) {
    ZMatrix rt = simpson([&](double t) { return ZMatrix(g.R(T, t) * d.L(t, T) + g.L(T, t) * d.R(T, t).adjoint()); }, 0, T);
    ZMatrix mix = simpson([&](double tau) {
      ZMatrix dvt = -double(xi) * d.TV(T, beta - tau).adjoint();
      return ZMatrix(-II * g.TV(T, tau) * dvt);
    }, 0, beta);
    return ZMatrix(-II * (rt + mix));
  };
  double Tphys = 1.6;
  int nt = (int)std::lround(Tphys / h) + 1;
  herm_matrix_hodlr G(nt, r, 1, 1e-14, NAO, NAO, xi, k), Dl(nt, r, 1, 1e-14, NAO, NAO, xi, k);
  Integration::Integrator I(k);
  dyson D(nt, NAO, k, dlr, true, false);
  auto fill = [&](herm_matrix_hodlr &X, const F &f, int T) {
    for (int tp = 0; tp <= T; tp++) {
      X.set_ret_curr(T, tp, f.R(T * h, tp * h));
      X.set_les_curr(tp, T, f.L(tp * h, T * h));
    }
    for (int i = 0; i < r; i++) X.set_tv(T, i, f.TV(T * h, dlr.it0B(i)));
  };
  for (int T = 0; T <= k; T++) {
    fill(G, g, T);
    fill(Dl, d, T);
  }
  GammaRes res;
  ZMatrix out(NAO, NAO);
  for (int T = 0; T <= k; T++) {
    D.gamma_integral(T, G, Dl, h, I, out.data());
    res.err.push_back((out - ref(T * h)).norm());
  }
  while (G.tstp() < nt - 1) {
    G.update_blocks(I);
    Dl.update_blocks(I);
    int T = G.tstp();
    fill(G, g, T);
    fill(Dl, d, T);
  }
  D.gamma_integral(nt - 1, G, Dl, h, I, out.data());
  res.err_last = (out - ref((nt - 1) * h)).norm();
  return res;
}

// Q2 relation for the free equilibrium G = Delta (nao = 2): gamma_integral(0) = xi * gamma_integral_mat
double gamma_vs_mat(int xi) {
  double beta = 4;
  int r = 0;
  dlr_info dlr(r, 50., 1e-13, beta, NAO, xi);
  int k = 2, nt = 10;
  herm_matrix_hodlr G(nt, r, 1, 1e-14, NAO, NAO, xi, k);
  Integration::Integrator I(k);
  dyson D(nt, NAO, k, dlr, true, false);
  ZMatrix H(NAO, NAO);
  H << 0.7, 0.3, 0.3, 1.4;
  D.green_from_H(G, xi == -1 ? 0.9 : 0.0, H, 0.05, k, true);
  ZMatrix a(NAO, NAO), b(NAO, NAO);
  D.gamma_integral(0, G, G, 0.05, I, a.data());
  D.gamma_integral_mat(G, G, 0.05, I, b.data());
  return (a - double(xi) * b).norm() / b.norm();
}
}  // namespace

class GammaIntegralFermion : public ::testing::TestWithParam<int> {};
TEST_P(GammaIntegralFermion, VsQuadratureAndOrder) {
  const int k = GetParam();
  GammaRes a = gamma_errors(-1, k, 0.05), b = gamma_errors(-1, k, 0.025);
  for (double e : a.err) EXPECT_LT(e, 1e-5);
  for (double e : b.err) EXPECT_LT(e, 1e-5);
  EXPECT_LT(a.err_last, 1e-5);
  EXPECT_GT(hn_test::rate(a.err_last, b.err_last), k + 1 - 0.3) << "err at T=nt-1: " << a.err_last << " -> " << b.err_last;
}
INSTANTIATE_TEST_SUITE_P(Gamma, GammaIntegralFermion, ::testing::Values(2, 4));

TEST(GammaIntegralFermion, EquilibriumVsMat) { EXPECT_LT(gamma_vs_mat(-1), 1e-12); }

TEST(GammaIntegralBoson, KnownBug_B2_VsQuadrature) {
  hn_test::run_isolated([] {
    GammaRes a = gamma_errors(+1, 2, 0.05);
    for (double e : a.err) EXPECT_LT(e, 1e-5);
    EXPECT_LT(a.err_last, 1e-5);
  });
}
TEST(GammaIntegralBoson, KnownBug_B2_EquilibriumVsMat) {
  hn_test::run_isolated([] { EXPECT_LT(gamma_vs_mat(+1), 1e-12); });
}
