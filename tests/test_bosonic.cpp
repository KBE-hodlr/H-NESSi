// Exact resolvent-identity tests for both statistics (doc 51 §1).  Ported from agent_scratch/51/t_resolvent.cpp.
//   G_e * G_n = (G_e - G_n)/(e - n)   for every Keldysh component (two free levels, same beta and xi).
// References use only the paper conventions (Table 1, Eqs. 2-4), with f = 1/(exp(beta e) - xi):
//   G^R = -i exp(-i e (t-t')),  G^< = -i xi f exp(-i e (t-t')),  G^tv(t,tau) = -i xi f exp(-i e t) exp(e tau),
//   G^M(tau) = -(1 + xi f) exp(-e tau).
// Checks: B1 (tv_it_conv), B2 (gamma_integral), Q2 (gamma_integral_mat = C^M(beta^-)), B10 (green_from_H_dm).
#include <gtest/gtest.h>

#include <cmath>

#include "h_nessi/dyson.hpp"
#include "common/test_util.hpp"

using namespace h_nessi;
typedef std::complex<double> C;

namespace {
const C II(0, 1);

struct Free {
  double e, beta;
  int xi;
  double f;
  Free(double e_, double b, int x) : e(e_), beta(b), xi(x) { f = 1. / (exp(beta * e) - xi); }
  C R(double t, double tp) const { return -II * exp(-II * e * (t - tp)); }
  C L(double t, double tp) const { return -II * double(xi) * f * exp(-II * e * (t - tp)); }
  C TV(double t, double tau) const { return -II * double(xi) * f * exp(-II * e * t) * exp(e * tau); }
  double M(double tau) const { return -(1. + xi * f) * exp(-e * tau); }
};

void fill(herm_matrix_hodlr &G, const Free &g, dlr_info &dlr, int r, int kmax, double h) {
  for (int i = 0; i < r; i++) {
    DMatrix m(1, 1);
    m(0, 0) = g.M(dlr.it0B(i));
    G.set_mat(i, m);
  }
  for (int t = 0; t <= kmax; t++)
    for (int i = 0; i < r; i++) {
      ZMatrix m(1, 1);
      m(0, 0) = g.TV(t * h, dlr.it0B(i));
      G.set_tv(t, i, m);
    }
  for (int t = 0; t <= kmax; t++)
    for (int tp = 0; tp <= t; tp++) {
      ZMatrix m(1, 1);
      m(0, 0) = g.R(t * h, tp * h);
      G.set_ret_curr(t, tp, m);
      m(0, 0) = g.L(tp * h, t * h);
      G.set_les_curr(tp, t, m);
    }
}

struct ResSetup {
  double beta = 4., lambda = 60., eps = 1e-13, h = 0.01, e = 0.7, n = 1.3;
  int k = 5, nt = 20, nlvl = 1, r = 0, xi;
  dlr_info dlr;
  Free ge, gn;
  herm_matrix_hodlr Ge, Gn;
  dyson D;
  Integration::Integrator I;
  explicit ResSetup(int xi_)
      : xi(xi_), dlr(r, lambda, eps, beta, 1, xi_), ge(e, beta, xi_), gn(n, beta, xi_),
        Ge(nt, r, nlvl, 1e-14, 1, 1, xi_, k), Gn(nt, r, nlvl, 1e-14, 1, 1, xi_, k), D(nt, 1, k, dlr), I(k) {
    fill(Ge, ge, dlr, r, k, h);
    fill(Gn, gn, dlr, r, k, h);
  }
};

// B1: int_0^beta Gn^tv(0,tb) Ge^M(tb - tau) dtb = [Gn*Ge]^tv(0,tau) = (Gn^tv - Ge^tv)(0,tau)/(n-e)
void check_tv_it_conv(int xi) {
  ResSetup s(xi);
  s.Ge.initGMConvTensor(s.dlr);
  std::vector<C> res(s.r, C(0.));
  s.D.tv_it_conv(0, s.Gn, s.Ge, res.data());
  double ep = 0, nr = 0;
  for (int a = 0; a < s.r; a++) {
    double tau = s.dlr.it0B(a);
    C ref = (s.gn.TV(0, tau) - s.ge.TV(0, tau)) / (s.n - s.e);
    ep = std::max(ep, std::abs(res[a] - ref));
    nr = std::max(nr, std::abs(ref));
  }
  EXPECT_LT(ep / nr, 1e-12) << "xi=" << xi;
}

// B2: gamma_integral(tstp) = -i [Ge*Gn]^<(t,t) = -xi (f_e - f_n)/(e - n), independent of t (equilibrium)
void check_gamma_integral(int xi) {
  ResSetup s(xi);
  double ref = -xi * (s.ge.f - s.gn.f) / (s.e - s.n);
  for (int tstp = 0; tstp <= s.k; tstp++) {
    C gi;
    s.D.gamma_integral(tstp, s.Ge, s.Gn, s.h, s.I, &gi);
    EXPECT_NEAR(gi.real(), ref, 1e-9) << "xi=" << xi << " tstp=" << tstp;
    EXPECT_NEAR(gi.imag(), 0.0, 1e-9) << "xi=" << xi << " tstp=" << tstp;
  }
}

// Q2: gamma_integral_mat = C^M(beta^-) = (Ge^M - Gn^M)(beta^-)/(e-n) for both xi, and in equilibrium
//     gamma_integral = xi * gamma_integral_mat  (-i C^<(t,t) = xi C^M(beta^-)).
void check_gamma_relation(int xi) {
  ResSetup s(xi);
  C gm, gi;
  s.D.gamma_integral_mat(s.Ge, s.Gn, s.h, s.I, &gm);
  s.D.gamma_integral(0, s.Ge, s.Gn, s.h, s.I, &gi);
  EXPECT_NEAR(std::abs(gi - double(xi) * gm), 0.0, 1e-9) << "xi=" << xi;
}

// B10: green_from_H_dm with rho = density matrix (= density_matrix(-1) = -G^M(beta^-)), decision D-1.
void check_green_from_H_dm(int xi) {
  ResSetup s(xi);
  herm_matrix_hodlr G2(s.nt, s.r, s.nlvl, 1e-14, 1, 1, xi, s.k);
  for (int i = 0; i < s.r; i++) {
    DMatrix m(1, 1);
    m(0, 0) = s.ge.M(s.dlr.it0B(i));
    G2.set_mat(i, m);
  }
  ZMatrix rho(1, 1);
  G2.density_matrix(-1, s.dlr, rho);
  EXPECT_NEAR(std::abs(rho(0, 0) - s.ge.f), 0.0, 1e-12) << "density_matrix(-1) must be f";
  ZMatrix H(1, 1);
  H(0, 0) = s.e;
  s.D.green_from_H_dm(G2, 0.0, H, rho, s.h, s.k);
  ZMatrix M(1, 1), r0(1, 1);
  double eL = 0, eC = 0;
  for (int t = 0; t <= s.k; t++)
    for (int tp = t; tp <= s.k; tp++) {
      G2.get_les_curr(t, tp, M);
      eL = std::max(eL, std::abs(M(0, 0) - s.ge.L(t * s.h, tp * s.h)));
    }
  for (int t = 0; t <= s.k; t++) {  // continuity G^tv(t,0) = G^<(t,0) = -conj(G^<(0,t))
    ZMatrix L(1, 1), TV(1, 1);
    G2.get_les_curr(0, t, L);
    G2.get_tv_tau(t, 0.0, s.dlr, TV);
    eC = std::max(eC, std::abs(TV(0, 0) + std::conj(L(0, 0))));
  }
  G2.density_matrix(0, s.dlr, r0);
  EXPECT_LT(eL, 1e-12) << "xi=" << xi << " max|G^< - exact|";
  EXPECT_LT(eC, 1e-10) << "xi=" << xi << " contour continuity";
  EXPECT_NEAR(std::abs(r0(0, 0) - s.ge.f), 0.0, 1e-12) << "xi=" << xi << " density_matrix(0)";
}

}  // namespace

TEST(ResolventFermion, TvItConv) { check_tv_it_conv(-1); }
TEST(ResolventFermion, GammaIntegralEquilibrium) { check_gamma_integral(-1); }
TEST(ResolventFermion, GammaIntegralVsMat) { check_gamma_relation(-1); }
TEST(ResolventFermion, GreenFromHdm) { check_green_from_H_dm(-1); }

// gamma_integral_mat is correct for both statistics (Q2 false positive).
TEST(ResolventBoth, GammaIntegralMatExact) {
  for (int xi : {-1, +1}) {
    ResSetup s(xi);
    C gm;
    s.D.gamma_integral_mat(s.Ge, s.Gn, s.h, s.I, &gm);
    double refMat = (-s.ge.f + s.gn.f) / (s.e - s.n);
    EXPECT_NEAR(gm.real(), refMat, 1e-10) << "xi=" << xi;
    EXPECT_NEAR(gm.imag(), 0.0, 1e-12) << "xi=" << xi;
  }
}

TEST(ResolventBoson, KnownBug_B1_TvItConv) {
  hn_test::run_isolated([] { check_tv_it_conv(+1); });
}
TEST(ResolventBoson, KnownBug_B2_GammaIntegralEquilibrium) {
  hn_test::run_isolated([] { check_gamma_integral(+1); });
}
TEST(ResolventBoson, KnownBug_B2_GammaIntegralVsMat) {
  hn_test::run_isolated([] { check_gamma_relation(+1); });
}
TEST(ResolventBoson, KnownBug_B10_GreenFromHdm) {
  hn_test::run_isolated([] { check_green_from_H_dm(+1); });
}
