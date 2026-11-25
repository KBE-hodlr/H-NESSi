// Exact resolvent-identity tests for both statistics (doc 51 §1).  Ported from agent_scratch/51/t_resolvent.cpp.
//   G_e * G_n = (G_e - G_n)/(e - n)   for every Keldysh component (two free levels, same beta and xi).
// References use only the paper conventions (Table 1, Eqs. 2-4), with f = 1/(exp(beta e) - xi):
//   G^R = -i exp(-i e (t-t')),  G^< = -i xi f exp(-i e (t-t')),  G^tv(t,tau) = -i xi f exp(-i e t) exp(e tau),
//   G^M(tau) = -(1 + xi f) exp(-e tau).
// Checks: B1 (tv_it_conv), B2 (gamma_integral), Q2 (gamma_integral_mat = C^M(beta^-)), B10 (green_from_H_dm).
// B1, B2 and B10 are fixed in WP3; their former KnownBug_* tests are ordinary tests now.
// End-to-end (WP3, bottom of file): system + bath of free modes, dyson_mat -> initGMConvTensor -> dyson_start (TTI)
// -> dyson_timestep against the exact noninteracting propagator, both statistics, order k+1.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
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

TEST(ResolventBoson, TvItConv) {
  hn_test::run_isolated([] { check_tv_it_conv(+1); });
}
TEST(ResolventBoson, GammaIntegralEquilibrium) {
  hn_test::run_isolated([] { check_gamma_integral(+1); });
}
TEST(ResolventBoson, GammaIntegralVsMat) {
  hn_test::run_isolated([] { check_gamma_relation(+1); });
}
TEST(ResolventBoson, GreenFromHdm) {
  hn_test::run_isolated([] { check_green_from_H_dm(+1); });
}

// =============================================================================================== end-to-end (WP3)
// Two system modes linearly coupled to two bath modes (quadratic H_full = h_full - mu, real symmetric 4x4, for bosons
// with mu below the spectrum so that K = h_full - mu > 0 and the Bose function is finite). Sigma = V g_bath V^T is
// exact (the bath is not driven). The system block of h is driven after t_on = 0.5 > k*h, so dyson_start (TTI) sees a
// time-independent problem, while dyson_timestep runs out of equilibrium. Exact, with n0 = (e^{beta K} - xi)^{-1}:
//   G^R = -i U U'^dag, G^< = -i xi U n0 U'^dag, G^tv(t,tau) = -i xi U(t) n0 e^{K tau}, G^M(tau) = -(1 + xi n0) e^{-K tau}
// (system block), U(t) the full propagator (RK4 with fine substeps).
// Pipeline as in a user program: dyson_mat -> initGMConvTensor -> dyson_start -> {update_blocks, extrapolate,
// set Sigma row, dyson_timestep}. B1 enters dyson_start (all of R, <, tv) and every dyson_timestep_tv.
namespace e2e {
typedef Eigen::MatrixXcd CM;
const int NO = 2, NF = 4;

struct Model {
  int xi;
  double beta = 5., mu, a = 0.5, ton = 0.5;
  Eigen::MatrixXd hfull, K0, Kb, V;
  CM A;
  explicit Model(int xi_) : xi(xi_) {
    hfull.resize(NF, NF);
    hfull << 0.3, 0.2, 0.5, 0.3, 0.2, -0.4, -0.2, 0.4, 0.5, -0.2, 1.0, 0.0, 0.3, 0.4, 0.0, -0.7;
    mu = xi == 1 ? -1.4 : 0.1;  // bosons: spectrum of hfull - mu is [0.41, 2.67]
    K0 = hfull - mu * Eigen::MatrixXd::Identity(NF, NF);
    Kb = K0.block(2, 2, 2, 2);
    V = hfull.block(0, 2, 2, 2);
    A.resize(2, 2);
    A << C(0.2, 0), C(0.6, 0.4), C(0.6, -0.4), C(-0.3, 0);
  }
  double drive(double t) const { return t > ton ? a * exp(-1.0 / (t - ton)) : 0.; }  // C^infinity switch-on at ton
  CM Himp(double t) const { return hfull.block(0, 0, 2, 2).cast<C>() + drive(t) * A; }
  CM Kfull(double t) const {
    CM K = K0.cast<C>();
    K.block(0, 0, 2, 2) += drive(t) * A;
    return K;
  }
};

CM stat(const Eigen::MatrixXd &K, double beta, int xi) {  // (e^{beta K} - xi)^{-1}
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  Eigen::VectorXd d(K.rows());
  for (int i = 0; i < K.rows(); i++) d(i) = 1. / (exp(beta * es.eigenvalues()(i)) - xi);
  return (es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose()).cast<C>();
}
CM expm_sym(const Eigen::MatrixXd &K, C s) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  Eigen::VectorXcd d(K.rows());
  for (int i = 0; i < K.rows(); i++) d(i) = exp(s * es.eigenvalues()(i));
  CM U = es.eigenvectors().cast<C>();
  return U * d.asDiagonal() * U.transpose();
}

struct Exact {
  const Model &M;
  double h;
  std::vector<CM> U;
  CM n0, nb;
  Exact(const Model &M_, int nt, double h_) : M(M_), h(h_) {
    n0 = stat(M.K0, M.beta, M.xi);
    nb = stat(M.Kb, M.beta, M.xi);
    U.resize(nt);
    CM u = CM::Identity(NF, NF);
    U[0] = u;
    const int ns = 400;
    double dt = h / ns;
    auto f = [&](double tt, const CM &x) -> CM { return -II * M.Kfull(tt) * x; };
    for (int n = 1; n < nt; n++) {
      for (int s = 0; s < ns; s++) {
        double t = (n - 1) * h + s * dt;
        CM k1 = f(t, u), k2 = f(t + dt / 2, u + dt / 2 * k1), k3 = f(t + dt / 2, u + dt / 2 * k2), k4 = f(t + dt, u + dt * k3);
        u += dt / 6 * (k1 + 2. * k2 + 2. * k3 + k4);
      }
      U[n] = u;
    }
  }
  double xi() const { return double(M.xi); }
  CM GR(int n, int m) const { return (-II * U[n] * U[m].adjoint()).block(0, 0, 2, 2); }
  CM GL(int m, int n) const { return (-II * xi() * U[m] * n0 * U[n].adjoint()).block(0, 0, 2, 2); }
  CM Gtv(int n, double tau) const { return (-II * xi() * U[n] * n0 * expm_sym(M.K0, tau)).block(0, 0, 2, 2); }
  Eigen::MatrixXd GM(double tau) const {
    CM one = CM::Identity(NF, NF);
    return (-(one + xi() * n0) * expm_sym(M.K0, -tau)).block(0, 0, 2, 2).real();
  }
  CM VgV(const CM &g) const { return M.V.cast<C>() * g * M.V.transpose().cast<C>(); }
  CM SR(int n, int l) const { return VgV(-II * expm_sym(M.Kb, -II * double(n - l) * h)); }
  CM SL(int l, int n) const { return VgV(-II * xi() * expm_sym(M.Kb, -II * double(l - n) * h) * nb); }
  CM Stv(int n, double tau) const { return VgV(-II * xi() * expm_sym(M.Kb, -II * double(n) * h) * nb * expm_sym(M.Kb, tau)); }
  Eigen::MatrixXd SM(double tau) const {
    CM one = CM::Identity(2, 2);
    return VgV(-(one + xi() * nb) * expm_sym(M.Kb, -tau)).real();
  }
};

struct Errs { double M = 0, R = 0, L = 0, tv = 0; };

Errs run(int xi, int k, double h, double tmax) {
  Model Mo(xi);
  int nt = int(std::lround(tmax / h)) + 1, r = 0, nlvl = 2;
  dlr_info dlr(r, 60., 1e-13, Mo.beta, NO, xi);
  Exact E(Mo, nt, h);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, 1e-13, NO, NO, xi, k), Sig(nt, r, nlvl, 1e-13, NO, NO, xi, k);
  function H(nt, NO, NO);
  for (int t = -1; t < nt; t++) H.get_map(t) = ZMatrix(Mo.Himp(t < 0 ? 0. : t * h));
  auto set_sigma_row = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      Sig.set_ret_curr(T, tp, ZMatrix(E.SR(T, tp)));
      Sig.set_les_curr(tp, T, ZMatrix(E.SL(tp, T)));
    }
    for (int j = 0; j < r; j++) Sig.set_tv(T, j, ZMatrix(E.Stv(T, dlr.it0B(j))));
  };
  for (int j = 0; j < r; j++) {
    DMatrix s = E.SM(dlr.it0B(j));
    Sig.set_mat(j, s);
  }
  for (int T = 0; T <= k; T++) set_sigma_row(T);
  for (int T = 0; T <= k; T++)  // Sigma^R(n, l > n): smooth continuation (what a TTI user has in the window)
    for (int tp = T + 1; tp <= k; tp++) Sig.set_ret_curr(T, tp, ZMatrix(E.SR(T, tp)));

  dyson D(nt, NO, k, dlr);
  D.dyson_mat(G, Mo.mu, H, Sig, false, 1.0);
  G.initGMConvTensor(dlr);
  D.dyson_start(G, Mo.mu, H, Sig, I, h);

  Errs e;
  DMatrix dm(NO, NO);
  for (int j = 0; j < r; j++) {
    G.get_mat(j, dm);
    e.M = std::max(e.M, (dm - E.GM(dlr.it0B(j))).norm());
  }
  ZMatrix m(NO, NO);
  auto row_err = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      G.get_ret_curr(T, tp, m);
      e.R = std::max(e.R, (CM(m) - E.GR(T, tp)).norm());
      G.get_les_curr(tp, T, m);
      e.L = std::max(e.L, (CM(m) - E.GL(tp, T)).norm());
    }
    for (int j = 0; j < r; j++) {
      G.get_tv(T, j, m);
      e.tv = std::max(e.tv, (CM(m) - E.Gtv(T, dlr.it0B(j))).norm());
    }
  };
  for (int T = 0; T <= k; T++) row_err(T);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    Sig.update_blocks(I);
    D.extrapolate(G, I);
    set_sigma_row(T);
    D.dyson_timestep(T, G, Mo.mu, H, Sig, I, h);
    row_err(T);
  }
  return e;
}

void check(int xi) {
  const double tmax = 1.6;
  for (int k : {2, 4}) {
    std::vector<double> hs = {0.1, 0.05, 0.025, 0.0125};
    std::vector<Errs> e;
    for (double h : hs) e.push_back(run(xi, k, h, tmax));
    for (size_t i = 0; i < hs.size(); i++)
      std::printf("[e2e] xi=%+d k=%d h=%.4f  errM=%.2e errR=%.2e errL=%.2e errTV=%.2e\n", xi, k, hs[i], e[i].M, e[i].R,
                  e[i].L, e[i].tv);
    EXPECT_LT(e[0].M, 1e-11) << "dyson_mat vs exact G^M";
    // order k+1 on the finest pair (0.025 -> 0.0125; coarser pairs are pre-asymptotic for k = 4 after the switch-on)
    const size_t i = hs.size() - 1;
    const double want = k + 1 - 0.35;
    EXPECT_GT(hn_test::rate(e[i - 1].R, e[i].R), want) << "xi=" << xi << " k=" << k << " R";
    EXPECT_GT(hn_test::rate(e[i - 1].L, e[i].L), want) << "xi=" << xi << " k=" << k << " <";
    EXPECT_GT(hn_test::rate(e[i - 1].tv, e[i].tv), want) << "xi=" << xi << " k=" << k << " tv";
    for (size_t j = 1; j < hs.size(); j++) {
      EXPECT_LT(e[j].R, e[j - 1].R / 4) << "xi=" << xi << " k=" << k << " h=" << hs[j];
      EXPECT_LT(e[j].L, e[j - 1].L / 4) << "xi=" << xi << " k=" << k << " h=" << hs[j];
      EXPECT_LT(e[j].tv, e[j - 1].tv / 4) << "xi=" << xi << " k=" << k << " h=" << hs[j];
    }
    EXPECT_LT(e.back().R, 2e-5);
    EXPECT_LT(e.back().L, 2e-5);
    EXPECT_LT(e.back().tv, 2e-5);
  }
}
}  // namespace e2e

TEST(EndToEndFermion, MatStartTimestepVsExact) { e2e::check(-1); }
TEST(EndToEndBoson, MatStartTimestepVsExact) { e2e::check(+1); }
