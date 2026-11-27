// Time stepping T > k (docs 20, 21).
//  * R and tv step (dyson_timestep_ret / _tv) vs a dense evaluation of the documented discrete equations
//    (agent_scratch/20/rettv.cpp mode A) and convergence order vs an exact dot+bath model (mode B).
//  * Lesser step (dyson_timestep_les, _nobc, _2leg; RHO_DIAGONAL and RHO_HORIZONTAL) vs a dense evaluation of the
//    documented discrete system, les_lesadv_int_0_tstp region decomposition, and accuracy/order vs the exact
//    solution of a quadratic dot+bath (or closed) model (agent_scratch/21/les.cpp).
//  * B14 (_nobc boundary sign), B28 (dyson timing column 5).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "common/test_util.hpp"
#include "h_nessi/dlr.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/integration.hpp"
#define private public  // only to read dyson::Q_ (no library change)
#include "h_nessi/dyson.hpp"
#undef private

using namespace h_nessi;
namespace {
const cplx II(0, 1);
}

// =============================================================================================== R / tv (doc 20)
namespace rettv {
struct ResA { double maxR, maxTV, maxErr2, gnorm; };

// Mode A: nao=2, fixed analytic Sigma (few poles, time-dependent phases), time-dependent Hermitian H(t).
ResA modeA(int nt, int nlvl, int k, double tol) {
  double beta = 5.0, lambda = 50., epsdlr = 1e-12;
  int xi = -1;
  const int nao = 2;
  double h = 0.05, mu = 0.3, gamS = 0.0;
  int r = 0;
  dlr_info dlr(r, lambda, epsdlr, beta, nao, xi);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, tol, nao, nao, xi, k), S(nt, r, nlvl, tol, nao, nao, xi, k);
  dyson D(nt, nao, k, dlr, true, false);
  const int P = 3;
  double ep[P] = {-1.3, 0.4, 2.1}, am[P] = {0.7, 0.3, 0.5}, Om[P] = {1.1, 0.6, 1.7};
  ZMatrix Ap[P];
  for (int p = 0; p < P; p++) {
    ZMatrix v(nao, 1);
    v(0) = cos(0.4 + p);
    v(1) = sin(0.4 + p) * std::exp(II * (0.7 * p + 0.2));
    Ap[p] = 0.4 * v * v.adjoint();
  }
  auto phi = [&](int p, int t) { return am[p] * sin(Om[p] * t * h); };
  auto SR = [&](int t, int tp) {
    ZMatrix m = ZMatrix::Zero(nao, nao);
    double d = (t - tp) * h;
    for (int p = 0; p < P; p++) m += -II * Ap[p] * std::exp(-II * (ep[p] * d + phi(p, t) - phi(p, tp))) * exp(-gamS * d * d);
    return m;
  };
  auto SRext = [&](int a, int b) -> ZMatrix { return a >= b ? SR(a, b) : ZMatrix(-SR(b, a).adjoint()); };
  auto STV = [&](int t, int m) {
    ZMatrix x(nao, nao);
    double tau = dlr.it0B(m);
    for (int i = 0; i < nao; i++)
      for (int j = 0; j < nao; j++)
        x(i, j) = 0.2 * std::exp(-II * (0.3 + 0.1 * i - 0.2 * j) * (t * h)) * std::exp(-(0.5 + 0.3 * i + 0.1 * j) * tau) * (1.0 + 0.5 * i - 0.3 * j);
    return x;
  };
  function H(nt, nao, nao);
  for (int t = -1; t < nt; t++) {
    double tt = std::max(t, 0) * h;
    ZMatrix x(nao, nao);
    x << 0.5 + 0.2 * sin(tt), cplx(0.3, 0.1 * cos(tt)), cplx(0.3, -0.1 * cos(tt)), -0.4 + 0.1 * tt;
    if (t == -1) x = x.real().cast<cplx>();  // h(-1) must be real (G^M is real; WP6b: green_from_H(inc_mat) checks it)
    H.get_map(t) = x;
  }
  auto eps = [&](int t) { return ZMatrix(H.get_map(t) - mu * ZMatrix::Identity(nao, nao)); };
  D.green_from_H(G, mu, H, h, k, true);
  G.initGMConvTensor(dlr);
  auto fillS = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y = SR(T, tp);
      S.set_ret_curr(T, tp, y);
    }
    for (int t = 0; t <= T; t++) {
      ZMatrix y = ZMatrix::Zero(nao, nao);
      S.set_les_curr(t, T, y);
    }
    for (int m = 0; m < r; m++) {
      ZMatrix y = STV(T, m);
      S.set_tv(T, m, y);
    }
  };
  for (int T = 0; T <= k; T++) fillS(T);
  ResA res{0, 0, 0, 0};
  std::vector<ZMatrix> g(nt), gt(r);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    S.update_blocks(I);
    fillS(T);
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y = ZMatrix::Constant(nao, nao, cplx(0.1, 0.2));
      G.set_ret_curr(T, tp, y);
    }
    G.set_les_tstp_zero(T);
    // dense reference of ret row T: g[n] = G^R(T, T-n)
    g[0] = -II * ZMatrix::Identity(nao, nao);
    {
      ZMatrix A = ZMatrix::Zero(k * nao, k * nao), B = ZMatrix::Zero(nao, k * nao);
      for (int n = 1; n <= k; n++) {
        ZMatrix bn = -(II / h * I.poly_diff(n, 0) * g[0] - h * I.gregory_weights(n, 0) * g[0] * SRext(T, T - n));
        B.block(0, (n - 1) * nao, nao, nao) = bn;
        for (int l = 1; l <= k; l++) {
          ZMatrix a = II / h * I.poly_diff(n, l) * ZMatrix::Identity(nao, nao) - h * I.gregory_weights(n, l) * SRext(T - l, T - n);
          if (l == n) a -= eps(T - n);
          A.block((l - 1) * nao, (n - 1) * nao, nao, nao) = a;
        }
      }
      ZMatrix X = A.transpose().fullPivLu().solve(B.transpose()).transpose();
      for (int l = 1; l <= k; l++) g[l] = X.block(0, (l - 1) * nao, nao, nao);
    }
    for (int n = k + 1; n <= T; n++) {
      ZMatrix rhs = ZMatrix::Zero(nao, nao);
      for (int l = 0; l < n; l++) rhs += h * I.gregory_weights(n, l) * g[l] * SR(T - l, T - n);
      for (int l = 1; l <= k + 1; l++) rhs -= II / h * I.bd_weights(l) * g[n - l];
      ZMatrix Mt = II / h * I.bd_weights(0) * ZMatrix::Identity(nao, nao) - eps(T - n) - h * I.gregory_weights(n, n) * SR(T - n, T - n);
      g[n] = rhs * Mt.inverse();
    }
    // dense reference of tv row T (uses the code's earlier tv rows and the verified tv_it_conv)
    std::vector<cplx> Cv(r * nao * nao, 0.0);
    D.tv_it_conv(T, S, G, Cv.data());
    {
      ZMatrix Mt = (II / h * I.bd_weights(0) + mu) * ZMatrix::Identity(nao, nao) - H.get_map(T) - h * I.gregory_weights(T, T) * SR(T, T);
      for (int m = 0; m < r; m++) {
        ZMatrix rhs = ZMatrixMap(Cv.data() + m * nao * nao, nao, nao);
        for (int t = 0; t < T; t++) rhs += h * I.gregory_weights(T, t) * SR(T, t) * ZMatrixMap(G.tvptr(t, m), nao, nao);
        for (int l = 1; l <= k + 1; l++) rhs -= II / h * I.bd_weights(l) * ZMatrixMap(G.tvptr(T - l, m), nao, nao);
        gt[m] = Mt.fullPivLu().solve(rhs);
      }
    }
    D.dyson_timestep_ret(T, G, mu, H.ptr(0), S, I, h);
    double e2 = D.dyson_timestep_ret(T, G, mu, H.ptr(0), S, I, h);  // linear in G for fixed Sigma: 2nd call err ~ 0
    D.dyson_timestep_tv(T, G, mu, H.ptr(0), S, I, h);
    res.maxErr2 = std::max(res.maxErr2, e2);
    for (int n = 0; n <= T; n++) {
      double d = (ZMatrixMap(G.curr_timestep_ret_ptr(T, T - n), nao, nao) - g[n]).norm();
      res.maxR = std::max(res.maxR, d);
      res.gnorm = std::max(res.gnorm, g[n].norm());
    }
    for (int m = 0; m < r; m++) {
      res.maxTV = std::max(res.maxTV, (ZMatrixMap(G.tvptr(T, m), nao, nao) - gt[m]).norm());
    }
  }
  return res;
}

// Mode B: nao=1 level coupled to one bath level, Sigma = V^2 g_bath (exact); exact G from the 2x2 problem.
struct ResB { double eR, eTV; };
ResB modeB(double tmax, double h, int nlvl, int k) {
  double beta = 5.0, lambda = 50., epsdlr = 1e-12;
  int xi = -1;
  int nt = (int)std::lround(tmax / h) + 1;
  double tol = 1e-13;
  const int nao = 1;
  double mu = 0.0, ed = 0.4, eb = -0.7, V = 0.6;
  int r = 0;
  dlr_info dlr(r, lambda, epsdlr, beta, nao, xi);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, tol, nao, nao, xi, k), S(nt, r, nlvl, tol, nao, nao, xi, k);
  dyson D(nt, nao, k, dlr, true, false);
  ZMatrix H2(2, 2);
  H2 << ed, V, V, eb;
  Eigen::SelfAdjointEigenSolver<ZMatrix> es(H2);
  ZMatrix U = es.eigenvectors();
  Eigen::VectorXd w = es.eigenvalues();
  auto fermi = [&](double e) { return 1.0 / (1.0 + exp(beta * e)); };
  auto GR2 = [&](double d) {
    ZMatrix x = ZMatrix::Zero(2, 2);
    for (int a = 0; a < 2; a++) x += -II * std::exp(-II * w(a) * d) * U.col(a) * U.col(a).adjoint();
    return x;
  };
  auto GTV2 = [&](double t, double tau) {
    ZMatrix x = ZMatrix::Zero(2, 2);
    for (int a = 0; a < 2; a++) x += II * std::exp(-II * w(a) * t) * exp(w(a) * tau) * fermi(w(a)) * U.col(a) * U.col(a).adjoint();
    return x;
  };
  auto GM2 = [&](double tau) {
    ZMatrix x = ZMatrix::Zero(2, 2);
    for (int a = 0; a < 2; a++) x += -exp(-w(a) * tau) * (1 - fermi(w(a))) * U.col(a) * U.col(a).adjoint();
    return x;
  };
  auto gbR = [&](double d) { return -II * std::exp(-II * eb * d); };
  auto gbTV = [&](double t, double tau) { return II * std::exp(-II * eb * t) * exp(eb * tau) * fermi(eb); };
  auto gbM = [&](double tau) { return -exp(-eb * tau) * (1 - fermi(eb)); };
  function H(nt, nao, nao);
  ZMatrix e1(1, 1);
  e1(0, 0) = ed;
  H.set_constant(e1);
  for (int m = 0; m < r; m++) {
    double tau = dlr.it0B(m);
    G.matptr(m)[0] = GM2(tau)(0, 0).real();
    S.matptr(m)[0] = (V * V * gbM(tau));
  }
  G.initGMConvTensor(dlr);
  auto fillS = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y(1, 1);
      y(0, 0) = V * V * gbR((T - tp) * h);
      S.set_ret_curr(T, tp, y);
    }
    for (int t = 0; t <= T; t++) {
      ZMatrix y = ZMatrix::Zero(1, 1);
      S.set_les_curr(t, T, y);
    }
    for (int m = 0; m < r; m++) {
      ZMatrix y(1, 1);
      y(0, 0) = V * V * gbTV(T * h, dlr.it0B(m));
      S.set_tv(T, m, y);
    }
  };
  auto fillGexact = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y(1, 1);
      y(0, 0) = GR2((T - tp) * h)(0, 0);
      G.set_ret_curr(T, tp, y);
    }
    for (int m = 0; m < r; m++) {
      ZMatrix y(1, 1);
      y(0, 0) = GTV2(T * h, dlr.it0B(m))(0, 0);
      G.set_tv(T, m, y);
    }
    for (int t = 0; t <= T; t++) {
      ZMatrix y = ZMatrix::Zero(1, 1);
      G.set_les_curr(t, T, y);
    }
  };
  for (int T = 0; T <= k; T++) {
    fillS(T);
    fillGexact(T);
  }
  ResB res{0, 0};
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    S.update_blocks(I);
    fillS(T);
    G.set_les_tstp_zero(T);
    D.dyson_timestep_ret(T, G, mu, H.ptr(0), S, I, h);
    D.dyson_timestep_tv(T, G, mu, H.ptr(0), S, I, h);
    for (int tp = 0; tp <= T; tp++) res.eR = std::max(res.eR, std::abs(G.curr_timestep_ret_ptr(T, tp)[0] - GR2((T - tp) * h)(0, 0)));
    for (int m = 0; m < r; m++) res.eTV = std::max(res.eTV, std::abs(G.tvptr(T, m)[0] - GTV2(T * h, dlr.it0B(m))(0, 0)));
  }
  return res;
}
}  // namespace rettv

class RetTvDense : public ::testing::TestWithParam<std::tuple<int, int, int>> {};
TEST_P(RetTvDense, MatchesDocumentedDiscreteEquations) {
  auto [nt, nlvl, k] = GetParam();
  rettv::ResA r = rettv::modeA(nt, nlvl, k, 1e-12);
  EXPECT_LT(r.maxR, 1e-12) << "|G^R code - dense|  (max|G| " << r.gnorm << ")";
  EXPECT_LT(r.maxTV, 1e-13) << "|G^tv code - dense|";
  EXPECT_LT(r.maxErr2, 1e-13) << "second dyson_timestep_ret call must not change G";
}
INSTANTIATE_TEST_SUITE_P(Timestep, RetTvDense,
                         ::testing::Values(std::make_tuple(37, 3, 2), std::make_tuple(97, 3, 3), std::make_tuple(40, 4, 5),
                                           std::make_tuple(33, 5, 3), std::make_tuple(40, 3, 1)));

TEST(RetTvDense, CompressionToleranceControlsError) {
  rettv::ResA a = rettv::modeA(97, 3, 3, 1e-8);
  EXPECT_LT(a.maxR, 1e-9);
  EXPECT_LT(a.maxTV, 1e-13);
}

// Convergence order k+1 of R and tv vs the exact solution (tmax = 10, h = 0.1 -> 0.05).
class RetTvOrder : public ::testing::TestWithParam<int> {};
TEST_P(RetTvOrder, OrderKPlus1) {
  const int k = GetParam();
  rettv::ResB a = rettv::modeB(10.0, 0.1, 3, k), b = rettv::modeB(10.0, 0.05, 3, k);
  EXPECT_GT(hn_test::rate(a.eR, b.eR), k + 1 - 0.25) << a.eR << " -> " << b.eR;
  EXPECT_GT(hn_test::rate(a.eTV, b.eTV), k + 1 - 0.25) << a.eTV << " -> " << b.eTV;
}
INSTANTIATE_TEST_SUITE_P(Timestep, RetTvOrder, ::testing::Values(1, 2, 3, 4, 5));

// Full sweep h = 0.1 ... 0.0125 (slow).
class RetTvOrderSweep : public ::testing::TestWithParam<int> {};
TEST_P(RetTvOrderSweep, Slow_OrderKPlus1) {
  const int k = GetParam();
  std::vector<rettv::ResB> e;
  for (double h : {0.1, 0.05, 0.025, 0.0125}) e.push_back(rettv::modeB(10.0, h, 3, k));
  for (size_t i = 1; i < e.size(); i++) {
    if (e[i].eR < 1e-12) continue;  // roundoff floor
    EXPECT_GT(hn_test::rate(e[i - 1].eR, e[i].eR), k + 1 - 0.25) << "pair " << i;
    EXPECT_GT(hn_test::rate(e[i - 1].eTV, e[i].eTV), k + 1 - 0.25) << "pair " << i;
  }
}
INSTANTIATE_TEST_SUITE_P(Timestep, RetTvOrderSweep, ::testing::Values(1, 2, 3, 4, 5));

// =============================================================================================== lesser (doc 21)
namespace les {
// variant 0 = dyson_timestep, 1 = dyson_timestep_nobc, 2 = dyson_timestep_2leg (uncorrelated initial state),
// 3 = closed system (no Sigma, G = whole 5x5 problem).
int nao = 2;
const int nd = 2, nb = 3, nf = nd + nb;
double hstep, beta = 4.0, mu = 0.3;
int drive = 1, variant = 0, xi = -1;  // xi = +1: bosonic dot+bath (B14 test), mu shifted below the spectrum
std::vector<ZMatrix> U;
ZMatrix K0, rho, Vc;
DMatrix Kb;
ZMatrix Kfull(double s) {
  ZMatrix K = ZMatrix::Zero(nf, nf);
  double e1 = -0.4, e2 = 0.5;
  cplx w = 0.35;
  if (drive) {
    e1 += 0.6 * sin(1.3 * s);
    w = 0.35 * std::exp(II * 0.8 * sin(0.9 * s));
  }
  K(0, 0) = e1;
  K(1, 1) = e2;
  K(0, 1) = w;
  K(1, 0) = std::conj(w);
  double eb[nb] = {-1.1, 0.2, 1.4};
  for (int b = 0; b < nb; b++) {
    K(nd + b, nd + b) = eb[b];
    for (int i = 0; i < nd; i++) {
      K(i, nd + b) = Vc(i, b);
      K(nd + b, i) = std::conj(Vc(i, b));
    }
  }
  return K - mu * ZMatrix::Identity(nf, nf);
}
ZMatrix expmH(const ZMatrix &Hm, cplx fac) {
  Eigen::SelfAdjointEigenSolver<ZMatrix> es(Hm);
  ZMatrix D = ZMatrix::Zero(Hm.rows(), Hm.rows());
  for (int i = 0; i < Hm.rows(); i++) D(i, i) = std::exp(fac * es.eigenvalues()(i));
  return es.eigenvectors() * D * es.eigenvectors().adjoint();
}
ZMatrix fermi(const ZMatrix &K) {  // Fermi (xi = -1) or Bose (xi = +1) occupation
  Eigen::SelfAdjointEigenSolver<ZMatrix> es(K);
  ZMatrix D = ZMatrix::Zero(K.rows(), K.rows());
  for (int i = 0; i < K.rows(); i++) D(i, i) = 1. / (std::exp(beta * es.eigenvalues()(i)) - double(xi));
  return es.eigenvectors() * D * es.eigenvectors().adjoint();
}
ZMatrix dot(const ZMatrix &A) { return A.block(0, 0, nao, nao); }
ZMatrix exGR(int t, int tp) { return dot(-II * U[t] * U[tp].adjoint()); }
ZMatrix exGL(int t, int tp) { return dot(-II * double(xi) * U[t] * rho * U[tp].adjoint()); }
ZMatrix exGTV(int t, double tau) { return dot(-II * double(xi) * U[t] * rho * expmH(K0, tau)); }
ZMatrix exGM(double tau) { return dot(-expmH(K0, -tau) * (ZMatrix::Identity(nf, nf) + double(xi) * rho)); }
ZMatrix fb() {
  ZMatrix f = ZMatrix::Zero(nb, nb);
  for (int b = 0; b < nb; b++) f(b, b) = 1. / (std::exp(beta * Kb(b, b)) - double(xi));
  return f;
}
ZMatrix eb(double s) {
  ZMatrix e = ZMatrix::Zero(nb, nb);
  for (int b = 0; b < nb; b++) e(b, b) = std::exp(-II * Kb(b, b) * s);
  return e;
}
ZMatrix ebr(double s) {
  ZMatrix e = ZMatrix::Zero(nb, nb);
  for (int b = 0; b < nb; b++) e(b, b) = std::exp(Kb(b, b) * s);
  return e;
}
ZMatrix SR(int t, int tp) {
  if (variant == 3) return ZMatrix::Zero(nao, nao);
  return Vc * (-II * eb((t - tp) * hstep)) * Vc.adjoint();
}
ZMatrix SL(int t, int tp) {
  if (variant == 3) return ZMatrix::Zero(nao, nao);
  return Vc * (-II * double(xi) * eb(t * hstep) * fb() * eb(-tp * hstep)) * Vc.adjoint();
}
ZMatrix STV(int t, double tau) {
  if (variant == 3) return ZMatrix::Zero(nao, nao);
  return Vc * (-II * double(xi) * eb(t * hstep) * fb() * ebr(tau)) * Vc.adjoint();
}
ZMatrix SM(double tau) {
  if (variant == 3) return ZMatrix::Zero(nao, nao);
  return Vc * (-ebr(-tau) * (ZMatrix::Identity(nb, nb) + double(xi) * fb())) * Vc.adjoint();
}

struct Res {
  double worst_int = 0, worst_part = 0, worst_raw = 0, worst_solve = 0;  // dense checks
  double eL = 0, eR = 0, en = 0, ah = 0, drift = 0;                     // accuracy vs exact
  int max_iter = 0;
  std::vector<ZMatrix> GL;  // final G^<(t,t'), t <= t', row-major upper triangle
};

// dense: if true, every T > k is checked against the dense reimplementation (cost O(T^2) per step)
Res run(int nt, int nlvl, int k, double h, double tol, int rhov, int var, bool dense, int drv = 1, int sgn = -1) {
  nao = 2;
  xi = sgn;
  mu = 0.3;
  drive = drv;
  variant = var;
  hstep = h;
  if (variant == 3) nao = nf;
  Vc = ZMatrix(nd, nb);
  Vc << 0.3, 0.2, -0.25, 0.15, -0.3, 0.2;
  if (xi == 1) {  // bosons: h(0) - mu > 0 (whole system, hence also the bath levels)
    Eigen::SelfAdjointEigenSolver<ZMatrix> es0(Kfull(0) + mu * ZMatrix::Identity(nf, nf));
    mu = es0.eigenvalues()(0) - 0.4;
  }
  Kb = DMatrix::Zero(nb, nb);
  {
    ZMatrix K = Kfull(0);
    for (int b = 0; b < nb; b++) Kb(b, b) = K(nd + b, nd + b).real();
  }
  K0 = Kfull(0);
  if (variant == 2) {
    rho = ZMatrix::Zero(nf, nf);
    ZMatrix rd(nao, nao);
    rd << 0.7, 0.1 + 0.2 * II, 0.1 - 0.2 * II, 0.4;
    rho.block(0, 0, nao, nao) = rd;
    rho.block(nao, nao, nb, nb) = fb();
  } else
    rho = fermi(K0);
  // propagator: 4th-order Magnus, 40 substeps
  U.assign(nt, ZMatrix());
  U[0] = ZMatrix::Identity(nf, nf);
  int ns = 40;
  double dt = hstep / ns;
  for (int t = 1; t < nt; t++) {
    ZMatrix u = U[t - 1];
    for (int s = 0; s < ns; s++) {
      double s0 = (t - 1) * hstep + s * dt;
      double c1 = 0.5 - sqrt(3.) / 6, c2 = 0.5 + sqrt(3.) / 6;
      ZMatrix K1 = Kfull(s0 + c1 * dt), K2 = Kfull(s0 + c2 * dt);
      ZMatrix Hm = 0.5 * dt * (K1 + K2) - II * (sqrt(3.) * dt * dt / 12.) * (K2 * K1 - K1 * K2);
      u = expmH(Hm, -II) * u;
    }
    U[t] = u;
  }
  int r = 0;
  dlr_info dlr(r, 40., 1e-13, beta, nao, xi);
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, tol, nao, nao, xi, k), S(nt, r, nlvl, tol, nao, nao, xi, k);
  dyson D(nt, nao, k, dlr, rhov, false);
  function Hf(nt, nao, nao);
  for (int t = -1; t < nt; t++) {
    ZMatrix K = Kfull(std::max(t, 0) * hstep) + mu * ZMatrix::Identity(nf, nf);
    ZMatrix Hd = K.block(0, 0, nao, nao);
    Hf.set_value(t, Hd);
  }
  if (variant != 2) {
    for (int n = 0; n < r; n++) {
      DMatrix gm = exGM(dlr.it0B(n)).real(), sm = SM(dlr.it0B(n)).real();
      G.set_mat(n, gm);
      S.set_mat(n, sm);
    }
    G.initGMConvTensor(dlr);
    for (int t = 0; t < nt; t++)
      for (int n = 0; n < r; n++) {
        ZMatrix y = STV(t, dlr.it0B(n));
        S.set_tv(t, n, y);
      }
  }
  auto fillS = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y = SR(T, tp);
      S.set_ret_curr(T, tp, y);
    }
    for (int t = 0; t <= T; t++) {
      ZMatrix y = SL(t, T);
      S.set_les_curr(t, T, y);
    }
  };
  auto fillG = [&](int T) {
    for (int tp = 0; tp <= T; tp++) {
      ZMatrix y = exGR(T, tp);
      G.set_ret_curr(T, tp, y);
    }
    for (int t = 0; t <= T; t++) {
      ZMatrix y = exGL(t, T);
      G.set_les_curr(t, T, y);
    }
    if (variant != 2)
      for (int n = 0; n < r; n++) {
        ZMatrix y = exGTV(T, dlr.it0B(n));
        G.set_tv(T, n, y);
      }
  };
  for (int T = 0; T <= k; T++) {
    fillS(T);
    fillG(T);
  }
  Res res;
  ZMatrix M(nao, nao), M2(nao, nao);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    S.update_blocks(I);
    fillS(T);
    if (variant == 2) D.extrapolate_2leg(G, I);
    else D.extrapolate(G, I);
    double err = 1;
    int it;
    for (it = 0; it < 60 && err > 1e-13; it++) {
      if (variant == 0 || variant == 3) err = D.dyson_timestep(T, G, mu, Hf, S, I, hstep);
      else if (variant == 1) err = D.dyson_timestep_nobc(T, G, mu, Hf, S, I, hstep);
      else err = D.dyson_timestep_2leg(T, G, mu, Hf, S, I, hstep);
    }
    res.max_iter = std::max(res.max_iter, it);
    if (!dense) continue;
    int A = G.tstpmk();
    // (a) les_lesadv_int_0_tstp vs dense sum_tb w(T,tb) Sigma^<(t,tb) G^A(tb,T)
    std::vector<ZMatrix> GA(T + 1), densev(T + 1, ZMatrix::Zero(nao, nao));
    for (int tb = 0; tb <= T; tb++) {
      G.get_ret(T, tb, M);
      GA[tb] = M.adjoint();
    }
    for (int t = 0; t <= T; t++)
      for (int tb = 0; tb <= T; tb++) {
        S.get_les(t, tb, M);
        densev[t] += hstep * I.gregory_weights(T, tb) * M * GA[tb];
      }
    std::memset(D.Q_.data(), 0, sizeof(cplx) * (T + 1) * nao * nao);
    D.les_lesadv_int_0_tstp(T, S, G, hstep, I);
    double e1 = 0;
    for (int t = 0; t <= T; t++) {
      ZMatrix q = ZMatrixMap(D.Q_.data() + t * nao * nao, nao, nao);
      e1 = std::max(e1, (q.transpose() - densev[t]).norm());
    }
    // documented region sets (doc 21 §4), each evaluated densely with the documented weight
    auto seg = [&](int t) { return G.t_to_dirlvl(t); };
    std::vector<std::vector<ZMatrix>> reg(9, std::vector<ZMatrix>(T + 1, ZMatrix::Zero(nao, nao)));
    for (int t = 0; t <= T; t++)
      for (int tb = 0; tb <= T; tb++) {
        S.get_les(t, tb, M);
        ZMatrix c = hstep * M * GA[tb];
        double w = I.gregory_weights(T, tb);
        bool arch = (t < A && tb < A);
        if (arch && seg(t) == seg(tb)) reg[1][t] += c;
        if (arch && seg(t) != seg(tb) && tb > t) reg[3][t] += c;
        if (arch && seg(t) != seg(tb) && tb < t) reg[4][t] += c;
        if (arch && seg(t) == seg(tb) && tb <= k) reg[2][t] += (w - 1) * c;
        if (arch && tb < t && seg(t) != seg(tb) && tb <= k) reg[5][t] += (w - 1) * c;
        if (arch && tb > t && seg(t) != seg(tb) && tb <= k) reg[6][t] += (w - 1) * c;
        if (tb >= std::max(A, t)) reg[7][t] += w * c;
        if (t >= A && tb < t) reg[8][t] += w * c;
      }
    double ep = 0;
    for (int t = 0; t <= T; t++) {
      ZMatrix s = ZMatrix::Zero(nao, nao);
      for (int q = 1; q <= 8; q++) s += reg[q][t];
      ep = std::max(ep, (s - densev[t]).norm());
    }
    // raw-storage evaluation of regions 1,3,4,5 exactly as documented
    std::vector<std::vector<ZMatrix>> raw(9, std::vector<ZMatrix>(T + 1, ZMatrix::Zero(nao, nao)));
    int len = S.len_les_dir_square();
    for (int t = 0; t < A; t++)
      for (int tb = 0; tb < A; tb++)
        if (seg(t) == seg(tb)) {
          int s = seg(t), is = G.blkr1(s - 1), hs = G.blkdirheight(s);
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++)
              M(i, j) = S.les_dir_square()[(i * nao + j) * len + S.les_dir_square_first_index(s) + (tb - is) * hs + (t - is)];
          raw[1][t] += hstep * M * GA[tb];
        }
    for (int b = 0; b < S.built_blocks(); b++)
      for (int a = 0; a < S.blklen(b); a++)
        for (int c = 0; c < S.blkc2(b) - S.blkc1(b) + 1; c++) {
          int tl = S.blkr1(b) + a, te = S.blkc1(b) + c;
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              auto &B = S.les().data().blocks()[b][i][j];
              cplx v = 0;
              for (int q = 0; q < B.epsrank(); q++) v += B.U()(a, q) * B.S()(q) * std::conj(B.V()(c, q));
              M(i, j) = v;
            }
          raw[3][te] += hstep * M * GA[tl];
          raw[4][tl] += hstep * (-M.adjoint()) * GA[te];
        }
    if (T >= 2 * k + 1) {
      for (int t = G.blkr1(0); t < A; t++)
        for (int tb = 0; tb <= k; tb++) {
          for (int i = 0; i < nao; i++)
            for (int m = 0; m < nao; m++) M(i, m) = S.les_left_edge()[i * (k + 1) * nt * nao + t * nao * (k + 1) + m * (k + 1) + tb];
          raw[5][t] += hstep * M * GA[tb];
        }
    }
    double er = 0;
    for (int t = 0; t <= T; t++)
      for (int q : {1, 3, 4, 5}) {
        if (q == 5 && T < 2 * k + 1) continue;
        er = std::max(er, (raw[q][t] - reg[q][t]).norm());
      }
    res.worst_int = std::max(res.worst_int, e1);
    res.worst_part = std::max(res.worst_part, ep);
    res.worst_raw = std::max(res.worst_raw, er);
    // (b) full lesser step vs dense reimplementation of the documented discrete system
    std::vector<ZMatrix> old(T + 1);
    for (int t = 0; t <= T; t++) {
      G.get_les_curr(t, T, M);
      old[t] = M;
    }
    std::vector<cplx> lit((T + 1) * nao * nao, 0.);
    if (variant != 2) D.les_it_int(T, S, G, lit.data());
    if (variant == 0 || variant == 3) D.dyson_timestep_les(T, G, mu, Hf.ptr(0), S, I, hstep);
    else if (variant == 1) D.dyson_timestep_les_nobc(T, G, mu, Hf.ptr(0), S, I, hstep);
    else D.dyson_timestep_les_2leg(T, G, mu, Hf.ptr(0), S, I, hstep);
    std::vector<ZMatrix> Y(T + 1), Qd(T + 1);
    ZMatrix Id = ZMatrix::Identity(nao, nao);
    for (int t = 0; t <= T; t++) Qd[t] = densev[t] + ZMatrixMap(lit.data() + t * nao * nao, nao, nao).transpose();
    auto H = [&](int t) { return ZMatrixMap(Hf.ptr(t), nao, nao); };
    if (variant == 0 || variant == 3) {
      G.get_tv_tau(T, 0.0, dlr, M);
      Y[0] = -M.adjoint();
    } else {
      // boundary G^<(0,T) from the Hermitian conjugate of Eq. 8 (doc 51 §3 B14):
      //   i d_t' G^<(0,t') + G^<(0,t') eps + int G^< Sigma^A = +i int G^rceil Sigma^lceil = -(les_it_int(0,T,G,Sigma))^T
      std::vector<cplx> mix(nao * nao, 0.);
      if (variant == 1) D.les_it_int(0, T, G, S, mix.data());
      ZMatrix MIX = ZMatrixMap(mix.data(), nao, nao).transpose();
      ZMatrix rhs = ZMatrix::Zero(nao, nao);
      for (int l = 1; l <= k + 1; l++) {
        G.get_les(0, T - l, M);
        rhs -= II / hstep * I.bd_weights(l) * M;
      }
      for (int l = 0; l < T; l++) {
        G.get_les(0, l, M);
        S.get_ret(T, l, M2);
        rhs -= hstep * I.gregory_weights(T, l) * M * M2.adjoint();
      }
      S.get_ret(T, T, M2);
      ZMatrix L = II / hstep * I.bd_weights(0) * Id + H(T) - mu * Id + hstep * I.omega(0) * M2.adjoint();
      Y[0] = (rhs - MIX) * L.inverse();  // correct sign (B14: the code uses +MIX)
    }
    {
      ZMatrix Am = ZMatrix::Zero(k * nao, k * nao), b = ZMatrix::Zero(k * nao, nao);
      for (int m = 1; m <= k; m++) {
        ZMatrix rr = Qd[m];
        for (int l = 0; l <= k; l++) {
          S.get_ret(m, l, M);
          ZMatrix blk = II / hstep * I.poly_diff(m, l) * Id - hstep * I.gregory_weights(m, l) * M;
          if (m == l) blk += mu * Id - H(m);
          if (l == 0) rr -= blk * Y[0];
          else Am.block((m - 1) * nao, (l - 1) * nao, nao, nao) += blk;
        }
        b.block((m - 1) * nao, 0, nao, nao) = rr;
      }
      ZMatrix x = Am.fullPivLu().solve(b);
      for (int m = 1; m <= k; m++) Y[m] = x.block((m - 1) * nao, 0, nao, nao);
    }
    for (int m = k + 1; m <= T; m++) {
      ZMatrix rr = Qd[m];
      for (int tb = 0; tb < m; tb++) {
        S.get_ret(m, tb, M);
        rr += hstep * I.gregory_weights(m, tb) * M * Y[tb];
      }
      S.get_ret(m, m, M2);
      if (m < T || rhov == 0) {
        for (int l = 1; l <= k + 1; l++) rr -= II / hstep * I.bd_weights(l) * Y[m - l];
        ZMatrix L = (II / hstep * I.bd_weights(0) + mu) * Id - H(m) - hstep * I.omega(0) * M2;
        Y[m] = L.fullPivLu().solve(rr);
      } else {
        ZMatrix QB = rr + hstep * I.omega(0) * M2 * old[T] + (H(T) - mu * Id) * old[T];
        ZMatrix d = -II * (QB + QB.adjoint());
        for (int l = 1; l <= k + 1; l++) {
          G.get_les(T - l, T - l, M);
          d -= 1. / hstep * I.bd_weights(l) * M;
        }
        Y[m] = hstep / I.bd_weights(0) * d;
      }
    }
    double e2 = 0;
    for (int t = 0; t <= T; t++) {
      G.get_les_curr(t, T, M);
      e2 = std::max(e2, (M - Y[t]).norm());
    }
    res.worst_solve = std::max(res.worst_solve, e2);
  }
  // (c) accuracy vs exact solution
  int T = nt - 1;
  for (int t = 0; t <= T; t++) {
    for (int tp = t; tp <= T; tp++) {
      G.get_les(t, tp, M);
      res.eL = std::max(res.eL, (M - exGL(t, tp)).norm());
    }
    for (int tp = 0; tp <= t; tp++) {
      G.get_ret(t, tp, M);
      res.eR = std::max(res.eR, (M - exGR(t, tp)).norm());
    }
    G.get_les(t, t, M);
    res.en = std::max(res.en, std::abs((II * double(xi) * M).trace() - (II * double(xi) * exGL(t, t)).trace()));
    res.ah = std::max(res.ah, (M + M.adjoint()).norm());
  }
  {
    G.get_les(0, 0, M);
    cplx n0 = (II * double(xi) * M).trace();
    for (int t = 0; t <= T; t++) {
      G.get_les(t, t, M);
      res.drift = std::max(res.drift, std::abs((II * double(xi) * M).trace() - n0));
    }
  }
  for (int t = 0; t <= T; t++)
    for (int tp = t; tp <= T; tp++) {
      G.get_les(t, tp, M);
      res.GL.push_back(M);
    }
  return res;
}
}  // namespace les

// RHO_HORIZONTAL (0) / RHO_DIAGONAL (1)
class LesDense : public ::testing::TestWithParam<std::tuple<int, int, int, int>> {};
TEST_P(LesDense, MatchesDocumentedDiscreteSystem) {
  auto [nt, nlvl, k, rhov] = GetParam();
  les::Res r = les::run(nt, nlvl, k, 0.05, 1e-12, rhov, 0, true);
  EXPECT_LT(r.worst_int, 1e-14) << "les_lesadv_int_0_tstp vs dense";
  EXPECT_LT(r.worst_part, 1e-14) << "region sets vs dense";
  EXPECT_LT(r.worst_raw, 1e-14) << "raw storage regions vs sets";
  EXPECT_LT(r.worst_solve, 1e-13) << "lesser step vs dense discrete system";
  EXPECT_LT(r.max_iter, 20) << "fixed-Sigma iteration must converge";
  if (rhov == 1) EXPECT_LT(r.ah, 1e-14) << "RHO_DIAGONAL keeps G^<(t,t) anti-Hermitian";
  EXPECT_LT(r.eL, 1e-3);
  EXPECT_LT(r.eR, 1e-3);
}
INSTANTIATE_TEST_SUITE_P(Timestep, LesDense,
                         ::testing::Values(std::make_tuple(40, 3, 2, 1), std::make_tuple(40, 3, 2, 0), std::make_tuple(40, 4, 5, 1),
                                           std::make_tuple(24, 3, 5, 0), std::make_tuple(37, 3, 5, 1)));

TEST(LesDense, TwoLegBothRhoModes) {
  for (int rhov : {0, 1}) {
    les::Res r = les::run(60, 3, 3, 0.05, 1e-12, rhov, 2, true);
    EXPECT_LT(r.worst_int, 1e-14) << "rhov=" << rhov;
    EXPECT_LT(r.worst_solve, 1e-13) << "rhov=" << rhov;
    EXPECT_LT(r.eL, 1e-4) << "rhov=" << rhov;
    EXPECT_LT(r.eR, 1e-4) << "rhov=" << rhov;
  }
}

// Closed 5-level system (Sigma = 0): RHO_DIAGONAL conserves the particle number to roundoff; RHO_HORIZONTAL
// conserves it only to the discretisation error (doc 21 §6).
TEST(LesDense, ClosedSystemConservesN) {
  les::Res d = les::run(80, 3, 3, 0.05, 1e-12, 1, 3, false);
  EXPECT_LT(d.drift, 1e-11) << "RHO_DIAGONAL";
  EXPECT_LT(d.en, 1e-10) << "RHO_DIAGONAL";
  EXPECT_LT(d.eL, 1e-4) << "RHO_DIAGONAL";
  les::Res h = les::run(80, 3, 3, 0.05, 1e-12, 0, 3, false);
  EXPECT_LT(h.drift, 1e-3) << "RHO_HORIZONTAL";
  EXPECT_LT(h.eL, 1e-4) << "RHO_HORIZONTAL";
}

// Order k+1 of G^< and G^R vs exact (h = 0.05 -> 0.025), both rho modes.
class LesOrder : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(LesOrder, OrderKPlus1) {
  auto [k, rhov] = GetParam();
  les::Res a = les::run(40, 3, k, 0.05, 1e-12, rhov, 0, false), b = les::run(80, 4, k, 0.025, 1e-12, rhov, 0, false);
  EXPECT_GT(hn_test::rate(a.eL, b.eL), k + 1 - 0.3) << a.eL << " -> " << b.eL;
  EXPECT_GT(hn_test::rate(a.eR, b.eR), k + 1 - 0.3) << a.eR << " -> " << b.eR;
}
INSTANTIATE_TEST_SUITE_P(Timestep, LesOrder,
                         ::testing::Values(std::make_tuple(2, 0), std::make_tuple(2, 1), std::make_tuple(4, 0), std::make_tuple(4, 1)));

class LesOrderSweep : public ::testing::TestWithParam<std::tuple<int, int>> {};
TEST_P(LesOrderSweep, Slow_OrderKPlus1) {
  auto [k, rhov] = GetParam();
  les::Res a = les::run(80, 4, k, 0.025, 1e-12, rhov, 0, false), b = les::run(160, 5, k, 0.0125, 1e-12, rhov, 0, false);
  EXPECT_GT(hn_test::rate(a.eL, b.eL), k + 1 - 0.3) << a.eL << " -> " << b.eL;
  EXPECT_GT(hn_test::rate(a.eR, b.eR), k + 1 - 0.3) << a.eR << " -> " << b.eR;
}
INSTANTIATE_TEST_SUITE_P(Timestep, LesOrderSweep,
                         ::testing::Values(std::make_tuple(1, 0), std::make_tuple(2, 0), std::make_tuple(2, 1), std::make_tuple(3, 1),
                                           std::make_tuple(4, 0), std::make_tuple(4, 1), std::make_tuple(5, 0), std::make_tuple(5, 1)));

// Larger geometry with compression, a few dense checks (long: slow).
TEST(LesDense, Slow_LargeGeometryK5) {
  les::Res r = les::run(161, 4, 5, 0.025, 1e-12, 0, 0, true);
  EXPECT_LT(r.worst_int, 1e-13);
  EXPECT_LT(r.worst_solve, 1e-12);
  EXPECT_LT(r.eL, 1e-7);
}

// B14 (fixed in WP7): dyson_timestep_nobc must reproduce the exact solution like dyson_timestep does (and its boundary
// value G^<(0,T) must solve the conjugate KBE with -les_it_int, see the dense check).
TEST(LesNobc, AccuracyAndBoundary_B14) {
  les::Res r = les::run(60, 3, 3, 0.05, 1e-12, 1, 1, true);
  EXPECT_LT(r.eL, 1e-4) << "max|G^< - exact|";
  EXPECT_LT(r.worst_solve, 1e-13) << "lesser step vs dense system with the correct boundary sign";
}

// B14: a full _nobc run (Sigma^rceil != 0: dot+bath) agrees with the default dyson_timestep (boundary condition
// G^<(0,T) = -[G^rceil(T,0)]^dagger) at the level of the discretisation error, both statistics and both rho modes.
// Before WP7 the boundary error grew to 0.33 by T = 59 (fermions).
TEST(LesNobc, MatchesDefaultTimestep_B14) {
  for (int sgn : {-1, 1})
    for (int rhov : {1, 0}) {
      les::Res d = les::run(60, 3, 3, 0.05, 1e-12, rhov, 0, false, 1, sgn), n = les::run(60, 3, 3, 0.05, 1e-12, rhov, 1, false, 1, sgn);
      ASSERT_EQ(d.GL.size(), n.GL.size());
      double diff = 0;
      for (size_t i = 0; i < d.GL.size(); i++) diff = std::max(diff, (d.GL[i] - n.GL[i]).norm());
      EXPECT_LT(d.eL, 1e-4) << "xi=" << sgn << " rhov=" << rhov << " default";
      EXPECT_LT(n.eL, 1e-4) << "xi=" << sgn << " rhov=" << rhov << " _nobc";
      EXPECT_LT(diff, 2 * std::max(d.eL, n.eL)) << "xi=" << sgn << " rhov=" << rhov << " _nobc vs default";
      std::printf("B14 xi=%+d rhov=%d: eL default %.2e, _nobc %.2e, max|_nobc - default| %.2e\n", sgn, rhov, d.eL, n.eL, diff);
    }
}

// =============================================================================================== B28
// B28 (fixed in WP9a): dyson timing column 5 (intdircorr) must be accumulated (non-zero) when profiling is on.
TEST(DysonTiming, Column5Accumulated_B28) {
  hn_test::run_isolated([] {
    int nt = 60, nlvl = 3, k = 3, nao = 2;
    double h = 0.05, beta = 5.0;
    int r = 0;
    dlr_info dlr(r, 50., 1e-12, beta, nao, -1);
    Integration::Integrator I(k);
    herm_matrix_hodlr G(nt, r, nlvl, 1e-12, nao, nao, -1, k), S(nt, r, nlvl, 1e-12, nao, nao, -1, k);
    dyson D(nt, nao, k, dlr, true, true);
    ZMatrix H0(nao, nao);
    H0 << 0.3, 0.1, 0.1, -0.2;
    function H(nt, nao, nao);
    H.set_constant(H0);
    D.green_from_H(G, 0.0, H0, h, k, true);
    G.initGMConvTensor(dlr);
    auto fillS = [&](int T) {
      for (int tp = 0; tp <= T; tp++) S.set_ret_curr(T, tp, ZMatrix(-II * 0.1 * std::exp(-II * 0.5 * double(T - tp) * h) * ZMatrix::Identity(nao, nao)));
      for (int tp = 0; tp <= T; tp++) S.set_les_curr(tp, T, ZMatrix(II * 0.05 * std::exp(-II * 0.5 * double(tp - T) * h) * ZMatrix::Identity(nao, nao)));
    };
    for (int T = 0; T <= k; T++) fillS(T);
    for (int T = k + 1; T < nt; T++) {
      G.update_blocks(I);
      S.update_blocks(I);
      fillS(T);
      D.extrapolate(G, I);
      D.dyson_timestep(T, G, 0.0, H, S, I, h);
    }
    std::string fn = "dyson_timing_b28.h5";
    std::remove(fn.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      D.write_timing(out, "dt");
    }
    h5e::File in(fn, h5e::File::ReadOnly);
    DMatrix t = in.getDataSet("dt/timing").read<DMatrix>();
    ASSERT_GT(t.rows(), nt - 2);
    EXPECT_GT(t.col(5).maxCoeff(), 0.0) << "timing column 5 (intdircorr) never accumulated";
    EXPECT_GT(t.col(6).maxCoeff(), 0.0) << "sanity: column 6 is accumulated";
  });
}

// =============================================================================================== P1
// P1: the cached LU factorisations of the step matrices (dyson.hpp lu_*_; h(t), Sigma^R(t,t) of archived times are
// final, so each M(t') is factorised once): a long-lived dyson object (warm cache, corrector iterations) gives BITWISE
// the same G as a fresh object per call (cold cache), also when mu changes in the middle of the run (the cache must be
// cleared). In Debug every reuse is also checked against a fresh factorisation (DY-T12). Also: profile on and off give
// bitwise the same G, and the per-level block timing (timing_lvl_) sums to timing columns 3, 22 and is bounded by 15, 16.
namespace p1 {
struct Run {
  int nt = 48, nlvl = 3, k = 3, nao = 2;
  double h = 0.05, beta = 5.0;
};
// march T = k+1..nt-1 with two corrector iterations; fresh = a new dyson object for every call
void march(const Run &P, bool rho_version, bool profile, bool fresh, bool kick, herm_matrix_hodlr &G, dyson *&Dkeep,
           dlr_info &dlr) {
  Integration::Integrator I(P.k);
  herm_matrix_hodlr S(P.nt, dlr.r(), P.nlvl, 1e-12, P.nao, P.nao, -1, P.k);
  Dkeep = new dyson(P.nt, P.nao, P.k, dlr, rho_version, profile);
  ZMatrix H0(P.nao, P.nao);
  H0 << 0.3, 0.1, 0.1, -0.2;
  function H(P.nt, P.nao, P.nao);
  for (int t = -1; t < P.nt; t++) {
    ZMatrix Ht = H0 + 0.05 * std::cos(0.3 * t) * ZMatrix::Identity(P.nao, P.nao);
    H.set_value(t, Ht);
  }
  Dkeep->green_from_H(G, 0.0, H0, P.h, P.k, true);
  G.initGMConvTensor(dlr);
  auto fillS = [&](int T) {
    for (int tp = 0; tp <= T; tp++) S.set_ret_curr(T, tp, ZMatrix(-II * 0.1 * std::exp(-II * 0.5 * double(T - tp) * P.h) * ZMatrix::Identity(P.nao, P.nao)));
    for (int tp = 0; tp <= T; tp++) S.set_les_curr(tp, T, ZMatrix(II * 0.05 * std::exp(-II * 0.5 * double(tp - T) * P.h) * ZMatrix::Identity(P.nao, P.nao)));
  };
  for (int T = 0; T <= P.k; T++) fillS(T);
  for (int T = P.k + 1; T < P.nt; T++) {
    G.update_blocks(I);
    S.update_blocks(I);
    fillS(T);
    Dkeep->extrapolate(G, I);
    const double mu = (kick && T >= P.nt / 2) ? 0.1 : 0.0;   // a different mu must clear the cache
    for (int it = 0; it < 2; it++) {
      if (fresh) {
        dyson Df(P.nt, P.nao, P.k, dlr, rho_version, profile);
        Df.dyson_timestep(T, G, mu, H, S, I, P.h);
      } else {
        Dkeep->dyson_timestep(T, G, mu, H, S, I, P.h);
      }
    }
  }
}
double maxdiff(herm_matrix_hodlr &A, herm_matrix_hodlr &B, int nt, int nao) {
  double d = 0;
  ZMatrix a(nao, nao), b(nao, nao);
  for (int t = 0; t < nt; t++)
    for (int tp = 0; tp <= t; tp++) {
      A.get_ret(t, tp, a); B.get_ret(t, tp, b); d = std::max(d, (a - b).cwiseAbs().maxCoeff());
      A.get_les(tp, t, a); B.get_les(tp, t, b); d = std::max(d, (a - b).cwiseAbs().maxCoeff());
    }
  d = std::max(d, (Eigen::Map<ZMatrix>(A.tvptr(0, 0), nt, A.r() * nao * nao) - Eigen::Map<ZMatrix>(B.tvptr(0, 0), nt, B.r() * nao * nao)).cwiseAbs().maxCoeff());
  return d;
}
}  // namespace p1

TEST(DysonP1, LuCacheBitwiseVsFreshObject) {
  hn_test::run_isolated([] {
    p1::Run P;
    for (bool rhov : {false, true})
      for (bool kick : {false, true}) {
        int r = 0;
        dlr_info dlr(r, 50., 1e-12, P.beta, P.nao, -1);
        herm_matrix_hodlr Gw(P.nt, r, P.nlvl, 1e-12, P.nao, P.nao, -1, P.k), Gf(Gw);
        dyson *Dw = nullptr, *Df = nullptr;
        p1::march(P, rhov, true, false, kick, Gw, Dw, dlr);
        p1::march(P, rhov, true, true, kick, Gf, Df, dlr);
        // the warm object did reuse factorisations (slots of archived t' are valid)
        int nvalid = 0;
        for (unsigned char v : Dw->lu_valid_) nvalid += v;
        EXPECT_GT(nvalid, P.nt / 2);
        EXPECT_EQ(p1::maxdiff(Gw, Gf, P.nt, P.nao), 0.0) << "rho_version=" << rhov << " kick=" << kick;
        delete Dw;
        delete Df;
      }
  });
}

TEST(DysonP1, ProfileOffBitwiseAndLevelTiming) {
  hn_test::run_isolated([] {
    p1::Run P;
    int r = 0;
    dlr_info dlr(r, 50., 1e-12, P.beta, P.nao, -1);
    herm_matrix_hodlr Gp(P.nt, r, P.nlvl, 1e-12, P.nao, P.nao, -1, P.k), Gn(Gp);
    dyson *Dp = nullptr, *Dn = nullptr;
    p1::march(P, true, true, false, false, Gp, Dp, dlr);
    p1::march(P, true, false, false, false, Gn, Dn, dlr);
    EXPECT_EQ(p1::maxdiff(Gp, Gn, P.nt, P.nao), 0.0);
    EXPECT_EQ(Dn->timing.rows(), 0);
    EXPECT_EQ(Dn->timing_lvl_.rows(), 0);
    const int L = Dp->tl_lmax_;
    EXPECT_EQ(L, 5);   // floor(log2(48))
    ASSERT_EQ(Dp->timing_lvl_.rows(), P.nt);
    ASSERT_EQ(Dp->timing_lvl_.cols(), 4 * L);
    double tot[4] = {0, 0, 0, 0};
    for (int t = P.k + 1; t < P.nt; t++) {
      double g[4] = {0, 0, 0, 0};
      for (int q = 0; q < 4; q++) {
        for (int l = 0; l < L; l++) g[q] += Dp->timing_lvl_(t, q * L + l);
        // levels above nlvl hold nothing
        for (int l = P.nlvl; l < L; l++) EXPECT_EQ(Dp->timing_lvl_(t, q * L + l), 0.0);
        tot[q] += g[q];
      }
      EXPECT_NEAR(g[0], Dp->timing(t, 3), 1e-9 * (1 + Dp->timing(t, 3))) << t;
      EXPECT_NEAR(g[1], Dp->timing(t, 22), 1e-9 * (1 + Dp->timing(t, 22))) << t;
      EXPECT_LE(g[2], Dp->timing(t, 15) * (1 + 1e-12) + 1e-9) << t;
      EXPECT_LE(g[3], Dp->timing(t, 16) * (1 + 1e-12) + 1e-9) << t;
    }
    for (int q = 0; q < 4; q++) EXPECT_GT(tot[q], 0.0) << "group " << q;
    // written by write_timing and read back by the checkpoint constructor
    std::string fn = "dyson_timing_p1.h5";
    std::remove(fn.c_str());
    {
      h5e::File out(fn, h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      Dp->write_checkpoint_hdf5(out, "d");
      Dp->write_timing(out, "dt");
    }
    h5e::File in(fn, h5e::File::ReadOnly);
    DMatrix tl = in.getDataSet("dt/timing_level").read<DMatrix>();
    EXPECT_EQ((tl - Dp->timing_lvl_).cwiseAbs().maxCoeff(), 0.0);
    dyson Dc(in, "d", dlr);
    EXPECT_EQ((Dc.timing_lvl_ - Dp->timing_lvl_).cwiseAbs().maxCoeff(), 0.0);
    std::remove(fn.c_str());
    delete Dp;
    delete Dn;
  });
}

// P1: one dyson object used for two (G, Sigma, h) systems alternately (as the mpi_example does for several k-points per
// thread) must give bitwise the same G as one object per system: the step-matrix cache is cleared when the Sigma object
// or the h array changes (lu_params_). Before this check the Debug build stopped with DY-T12 in the mpi_example.
TEST(DysonP1, LuCacheSharedObjectTwoSystems) {
  hn_test::run_isolated([] {
    p1::Run P;
    int r = 0;
    dlr_info dlr(r, 50., 1e-12, P.beta, P.nao, -1);
    Integration::Integrator I(P.k);
    struct Sys {
      herm_matrix_hodlr G, S;
      function H;
      ZMatrix H0;
    };
    auto make = [&](double shift) {
      Sys *s = new Sys{herm_matrix_hodlr(P.nt, r, P.nlvl, 1e-12, P.nao, P.nao, -1, P.k),
                       herm_matrix_hodlr(P.nt, r, P.nlvl, 1e-12, P.nao, P.nao, -1, P.k), function(P.nt, P.nao, P.nao),
                       ZMatrix(P.nao, P.nao)};
      s->H0 << 0.3 + shift, 0.1, 0.1, -0.2 - shift;
      for (int t = -1; t < P.nt; t++) {
        ZMatrix Ht = s->H0 + 0.05 * std::cos(0.3 * t) * ZMatrix::Identity(P.nao, P.nao);
        s->H.set_value(t, Ht);
      }
      return s;
    };
    auto fillS = [&](Sys &s, int T, double g) {
      for (int tp = 0; tp <= T; tp++) s.S.set_ret_curr(T, tp, ZMatrix(-II * g * std::exp(-II * 0.5 * double(T - tp) * P.h) * ZMatrix::Identity(P.nao, P.nao)));
      for (int tp = 0; tp <= T; tp++) s.S.set_les_curr(tp, T, ZMatrix(II * 0.5 * g * std::exp(-II * 0.5 * double(tp - T) * P.h) * ZMatrix::Identity(P.nao, P.nao)));
    };
    // shared: one dyson for A and B; separate: one per system
    Sys *A1 = make(0.0), *B1 = make(0.2), *A2 = make(0.0), *B2 = make(0.2);
    dyson Dsh(P.nt, P.nao, P.k, dlr, true, false), DA(P.nt, P.nao, P.k, dlr, true, false), DB(P.nt, P.nao, P.k, dlr, true, false);
    Dsh.green_from_H(A1->G, 0.0, A1->H0, P.h, P.k, true);
    Dsh.green_from_H(B1->G, 0.0, B1->H0, P.h, P.k, true);
    DA.green_from_H(A2->G, 0.0, A2->H0, P.h, P.k, true);
    DB.green_from_H(B2->G, 0.0, B2->H0, P.h, P.k, true);
    for (Sys *s : {A1, B1, A2, B2}) s->G.initGMConvTensor(dlr);
    for (int T = 0; T <= P.k; T++) { fillS(*A1, T, 0.1); fillS(*A2, T, 0.1); fillS(*B1, T, 0.15); fillS(*B2, T, 0.15); }
    for (int T = P.k + 1; T < P.nt; T++) {
      for (Sys *s : {A1, B1, A2, B2}) { s->G.update_blocks(I); s->S.update_blocks(I); }
      fillS(*A1, T, 0.1); fillS(*A2, T, 0.1); fillS(*B1, T, 0.15); fillS(*B2, T, 0.15);
      Dsh.extrapolate(A1->G, I); Dsh.extrapolate(B1->G, I); DA.extrapolate(A2->G, I); DB.extrapolate(B2->G, I);
      for (int it = 0; it < 2; it++) {
        Dsh.dyson_timestep(T, A1->G, 0.0, A1->H, A1->S, I, P.h);
        Dsh.dyson_timestep(T, B1->G, 0.0, B1->H, B1->S, I, P.h);
        DA.dyson_timestep(T, A2->G, 0.0, A2->H, A2->S, I, P.h);
        DB.dyson_timestep(T, B2->G, 0.0, B2->H, B2->S, I, P.h);
      }
    }
    EXPECT_EQ(p1::maxdiff(A1->G, A2->G, P.nt, P.nao), 0.0);
    EXPECT_EQ(p1::maxdiff(B1->G, B2->G, P.nt, P.nao), 0.0);
    EXPECT_GT(p1::maxdiff(A1->G, B1->G, P.nt, P.nao), 1e-3);   // the two systems differ
    for (Sys *s : {A1, B1, A2, B2}) delete s;
  });
}
