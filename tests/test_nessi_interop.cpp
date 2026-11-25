// NESSi interoperability (WP-N; BUGS B8, decision D-5). Built only with -DINCLUDE_NESSI=ON; ctest label "nessi".
// Conventions and the storage mapping: agent_docs/05_nessi_herm_matrix.md.
//
//  1. Exact equilibrium: green_from_H in both libraries (constant 2x2 h, both statistics, N_o = 1, 2). The five
//     distance_norm2_* functions return roundoff for R and <, and the DLR interpolation error for tv and M.
//  2. Complex Hermitian h: R, <, tv still roundoff; distance_norm2_mat equals |Im G^M| of NESSi (ISSUES D1).
//  3. Perturbation detection: a single-entry perturbation of either object in one component is seen by exactly
//     the distance function(s) of that component (and of curr), and the others return bit-identical values.
//  4. Precondition checks (std::invalid_argument).
//  5. Time-dependent noninteracting run (driven complex-Hermitian 2x2 h) and
//  6. interacting run with a fixed bath self-energy (impurity + 2 bath modes), both statistics: the full
//     dyson_mat -> dyson_start -> dyson_timestep pipeline in both libraries. The distances are bounded by the sum of
//     both libraries' errors against the exact solution and converge with order ~k+1 in h.
//
// Include order: H-NESSi headers first. cntr.hpp defines global macros (II, cdmatrix, dvector, GREEN, CFUNC, ...).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "h_nessi/dlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/integration.hpp"
#include "common/test_util.hpp"

#include "cntr/cntr.hpp"

using namespace h_nessi;
typedef std::complex<double> C;
typedef Eigen::MatrixXcd CM;
typedef cntr::herm_matrix<double> NGF;
static const C IU(0., 1.);

namespace {

// ------------------------------------------------------------------------------------------------ helpers
CM const_h(int nao, bool complex_h) {
  CM H(nao, nao);
  if (nao == 1) H << 0.7;
  else if (complex_h) H << C(0.7, 0), C(0.3, 0.25), C(0.3, -0.25), C(1.4, 0);
  else H << 0.7, 0.3, 0.3, 1.4;
  return H;
}

// norm of NESSi's G^M over its tau grid (scale for the DLR error)
double nessi_mat_norm(NGF &g) {
  int es = g.size1() * g.size2();
  return ZMatrixMap(g.matptr(0), g.ntau() + 1, es).norm();
}
double nessi_tv_norm(NGF &g, int t) {
  int es = g.size1() * g.size2();
  return ZMatrixMap(g.tvptr(t, 0), g.ntau() + 1, es).norm();
}

// Both libraries' free equilibrium G (green_from_H, exact up to the e^{-i eps h} power and DLR)
struct EqPair {
  int xi, nao, k = 4, nt = 16, ntau = 200, r = 0;
  double beta = 5., mu = -0.2, h = 0.05;
  dlr_info *dlr = nullptr;
  herm_matrix_hodlr *G = nullptr;
  NGF *Gn = nullptr;
  EqPair(int xi_, int nao_, bool complex_h = false) : xi(xi_), nao(nao_) {
    dlr = new dlr_info(r, 60., 1e-13, beta, nao, xi);
    G = new herm_matrix_hodlr(nt, r, 2, 1e-12, nao, nao, xi, k);
    CM H = const_h(nao, complex_h);
    {
      dyson D(nt, nao, k, *dlr);
      D.green_from_H(*G, mu, ZMatrix(H), h, k, true);  // rows 0..k, tv rows 0..k, mat
    }
    Gn = new NGF(nt - 1, ntau, nao, xi);  // NESSi nt = last index
    cdmatrix Hn = H;
    cntr::green_from_H(*Gn, mu, Hn, beta, h);
  }
  ~EqPair() {
    delete Gn;
    delete G;
    delete dlr;
  }
};

}  // namespace

// ================================================================================================ 1. exact equilibrium
class NessiEq : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(NessiEq, GreenFromHDistancesAreRoundoff) {
  int xi = std::get<0>(GetParam()), nao = std::get<1>(GetParam());
  EqPair P(xi, nao);
  double dmat = distance_norm2_mat(*P.G, *P.Gn, *P.dlr);
  double smat = nessi_mat_norm(*P.Gn);
  std::printf("[eq] xi=%+d nao=%d  mat %.2e (|G^M| %.2e)\n", xi, nao, dmat, smat);
  EXPECT_LT(dmat, 1e-11 * smat);
  EXPECT_EQ(distance_norm2_curr(-1, *P.G, *P.Gn, *P.dlr), dmat);
  for (int t = 0; t <= P.k; t++) {
    double dr = distance_norm2_curr_ret(t, *P.G, *P.Gn), dl = distance_norm2_curr_les(t, *P.G, *P.Gn);
    double dtv = distance_norm2_tv(t, *P.G, *P.Gn, *P.dlr), stv = nessi_tv_norm(*P.Gn, t);
    double dc = distance_norm2_curr(t, *P.G, *P.Gn, *P.dlr);
    std::printf("[eq] xi=%+d nao=%d t=%d  ret %.2e les %.2e tv %.2e (|tv| %.2e) curr %.2e\n", xi, nao, t, dr, dl,
                dtv, stv, dc);
    EXPECT_LT(dr, 1e-13) << "t=" << t;
    EXPECT_LT(dl, 1e-13) << "t=" << t;
    EXPECT_LT(dtv, 1e-11 * stv) << "t=" << t;
    EXPECT_NEAR(dc, std::sqrt(dr * dr + dl * dl + dtv * dtv), 1e-15 + 1e-14 * dc);
  }
}

INSTANTIATE_TEST_SUITE_P(XiNao, NessiEq, ::testing::Combine(::testing::Values(-1, 1), ::testing::Values(1, 2)));

// The NESSi conventions used by the distance functions (doc 05), checked on the exact equilibrium pair.
TEST(NessiConventions, SameSignsAndDensityMatrix) {
  for (int xi : {-1, 1}) {
    EqPair P(xi, 2);
    int es = 4;
    // rho = -G^M(beta) = i xi G^<(t,t) in NESSi, and the same in H-NESSi
    cdmatrix rn(2, 2), rn0(2, 2);
    P.Gn->density_matrix(-1, rn);
    P.Gn->density_matrix(0, rn0);
    ZMatrix rh(2, 2);
    P.G->density_matrix(-1, *P.dlr, rh);
    EXPECT_LT((CM(rh) - rn).norm(), 1e-11) << "xi=" << xi;
    EXPECT_LT((rn0 - rn).norm(), 1e-13) << "xi=" << xi;
    // continuity G^tv(t, tau=0) = G^<(t, 0); G^<(t,0) is stored (t <= t') only for t = 0
    EXPECT_LT((ZMatrixMap(P.Gn->tvptr(0, 0), 1, es) - ZMatrixMap(P.Gn->lesptr(0, 0), 1, es)).norm(), 1e-13);
    // G^vt(tau,t) = -xi [G^tv(t, beta - tau)]^dagger in both (NESSi get_vt, H-NESSi get_vt at the DLR nodes)
    cdmatrix vt(2, 2), tv(2, 2);
    P.Gn->get_vt(10, 2, vt);
    P.Gn->get_tv(2, P.ntau - 10, tv);
    EXPECT_LT((vt - (-double(xi)) * tv.adjoint()).norm(), 1e-14);
    // G^>(t,t') = G^R + G^< for t >= t' (get_gtr), G^<(t,t') for t > t' = -[G^<(t',t)]^dagger
    cdmatrix les10(2, 2), les01(2, 2);
    P.Gn->get_les(3, 1, les10);
    P.Gn->get_les(1, 3, les01);
    EXPECT_LT((les10 + les01.adjoint()).norm(), 1e-15);
  }
}

// ================================================================================================ 2. complex h (D1)
TEST(NessiEqComplexH, MatDistanceIsImaginaryPartOfNessiGM) {
  EqPair P(-1, 2, true);
  int es = 4;
  for (int t = 0; t <= P.k; t++) {
    EXPECT_LT(distance_norm2_curr_ret(t, *P.G, *P.Gn), 1e-13);
    EXPECT_LT(distance_norm2_curr_les(t, *P.G, *P.Gn), 1e-13);
    EXPECT_LT(distance_norm2_tv(t, *P.G, *P.Gn, *P.dlr), 1e-11 * nessi_tv_norm(*P.Gn, t));
  }
  double im = ZMatrixMap(P.Gn->matptr(0), P.ntau + 1, es).imag().norm();
  double dmat = distance_norm2_mat(*P.G, *P.Gn, *P.dlr);
  std::printf("[complex h] mat distance %.6e, |Im G^M_nessi| %.6e\n", dmat, im);
  EXPECT_GT(im, 0.1);
  EXPECT_NEAR(dmat, im, 1e-10);
}

// ================================================================================================ 3. perturbations
class NessiPerturb : public ::testing::TestWithParam<int> {};

TEST_P(NessiPerturb, EachComponentDetectedOnlyByItsFunction) {
  int xi = GetParam();
  EqPair P(xi, 2);
  const C delta(1e-3, -2e-3);
  const double ad = std::abs(delta);
  const int k = P.k;
  struct D5 {
    double ret, les, tv, mat, curr, currm;
  };
  auto all = [&](herm_matrix_hodlr &g1, NGF &g2, int t) {
    return D5{distance_norm2_curr_ret(t, g1, g2), distance_norm2_curr_les(t, g1, g2),
              distance_norm2_tv(t, g1, g2, *P.dlr), distance_norm2_mat(g1, g2, *P.dlr),
              distance_norm2_curr(t, g1, g2, *P.dlr), distance_norm2_curr(-1, g1, g2, *P.dlr)};
  };
  for (int t : {0, 2, k}) {
    const D5 b = all(*P.G, *P.Gn, t);
    // --- NESSi side, one entry (orbital (0,1)) per component
    {  // R(t, t') -> only ret and curr
      NGF g(*P.Gn);
      g.retptr(t, t / 2)[1] += delta;
      D5 d = all(*P.G, g, t);
      EXPECT_NEAR(d.ret, ad, 1e-12) << "t=" << t;
      EXPECT_EQ(d.les, b.les);
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_EQ(d.mat, b.mat);
      EXPECT_EQ(d.currm, b.currm);
      EXPECT_NEAR(d.curr, std::sqrt(d.ret * d.ret + b.les * b.les + b.tv * b.tv), 1e-15);
      if (t > 0) EXPECT_EQ(distance_norm2_curr_ret(t - 1, *P.G, g), distance_norm2_curr_ret(t - 1, *P.G, *P.Gn));
    }
    {  // <(t', t) -> only les and curr
      NGF g(*P.Gn);
      g.lesptr(t / 2, t)[1] += delta;
      D5 d = all(*P.G, g, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_NEAR(d.les, ad, 1e-12) << "t=" << t;
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_EQ(d.mat, b.mat);
      EXPECT_EQ(d.currm, b.currm);
      if (t > 0) EXPECT_EQ(distance_norm2_curr_les(t - 1, *P.G, g), distance_norm2_curr_les(t - 1, *P.G, *P.Gn));
    }
    {  // tv(t, tau_m) -> only tv and curr
      NGF g(*P.Gn);
      g.tvptr(t, 37)[1] += delta;
      D5 d = all(*P.G, g, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_EQ(d.les, b.les);
      EXPECT_NEAR(d.tv, ad, 1e-10) << "t=" << t;
      EXPECT_EQ(d.mat, b.mat);
      EXPECT_EQ(d.currm, b.currm);
      EXPECT_EQ(distance_norm2_tv(t + 1, *P.G, g, *P.dlr), distance_norm2_tv(t + 1, *P.G, *P.Gn, *P.dlr));
    }
    {  // M(tau_m) -> only mat and curr(-1)
      NGF g(*P.Gn);
      g.matptr(P.ntau)[1] += delta;  // tau = beta endpoint
      D5 d = all(*P.G, g, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_EQ(d.les, b.les);
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_NEAR(d.mat, ad, 1e-10);
      EXPECT_EQ(d.currm, d.mat);
      EXPECT_EQ(d.curr, b.curr);
    }
    // --- H-NESSi side
    ZMatrix m(2, 2);
    {
      herm_matrix_hodlr g(*P.G);
      g.get_ret_curr(t, t / 2, m);
      m(0, 1) += delta;
      g.set_ret_curr(t, t / 2, m);
      D5 d = all(g, *P.Gn, t);
      EXPECT_NEAR(d.ret, ad, 1e-12);
      EXPECT_EQ(d.les, b.les);
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_EQ(d.mat, b.mat);
    }
    {
      herm_matrix_hodlr g(*P.G);
      g.get_les_curr(t / 2, t, m);
      m(0, 1) += delta;
      g.set_les_curr(t / 2, t, m);
      D5 d = all(g, *P.Gn, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_NEAR(d.les, ad, 1e-12);
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_EQ(d.mat, b.mat);
    }
    {  // tv at one DLR node: the interpolant changes on the whole grid -> only tv sees it
      herm_matrix_hodlr g(*P.G);
      g.get_tv(t, 3, m);
      m(0, 1) += delta;
      g.set_tv(t, 3, m);
      D5 d = all(g, *P.Gn, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_EQ(d.les, b.les);
      EXPECT_GT(d.tv, ad);  // the DLR interpolant through the perturbed node: >= |delta| at tau = node, summed
      EXPECT_EQ(d.mat, b.mat);
    }
    {
      herm_matrix_hodlr g(*P.G);
      DMatrix dm(2, 2);
      g.get_mat(3, dm);
      dm(0, 1) += 1e-3;
      g.set_mat(3, dm);
      D5 d = all(g, *P.Gn, t);
      EXPECT_EQ(d.ret, b.ret);
      EXPECT_EQ(d.les, b.les);
      EXPECT_EQ(d.tv, b.tv);
      EXPECT_GT(d.mat, 1e-3);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Xi, NessiPerturb, ::testing::Values(-1, 1));

// ================================================================================================ 4. preconditions
TEST(NessiPreconditions, ThrowOnMismatch) {
  EqPair P(-1, 2);
  NGF wrong_sig(P.nt - 1, P.ntau, 2, +1), wrong_size(P.nt - 1, P.ntau, 1, -1), no_tau(P.nt - 1, 0, 2, -1),
      short_nt(1, P.ntau, 2, -1);
  EXPECT_THROW(distance_norm2_curr_ret(0, *P.G, wrong_sig), std::invalid_argument);
  EXPECT_THROW(distance_norm2_curr_les(0, *P.G, wrong_size), std::invalid_argument);
  EXPECT_THROW(distance_norm2_mat(*P.G, no_tau, *P.dlr), std::invalid_argument);
  EXPECT_THROW(distance_norm2_curr_ret(P.k + 1, *P.G, *P.Gn), std::invalid_argument);  // outside the window
  EXPECT_THROW(distance_norm2_curr_les(-2, *P.G, *P.Gn), std::invalid_argument);
  EXPECT_THROW(distance_norm2_curr_ret(3, *P.G, short_nt), std::invalid_argument);     // NESSi nt too small
  EXPECT_THROW(distance_norm2_tv(P.nt, *P.G, *P.Gn, *P.dlr), std::invalid_argument);
  int r2 = 0;
  dlr_info other(r2, 60., 1e-13, P.beta, 2, +1);
  EXPECT_THROW(distance_norm2_tv(0, *P.G, *P.Gn, other), std::invalid_argument);
  EXPECT_NO_THROW(distance_norm2_curr(P.k, *P.G, *P.Gn, *P.dlr));
}

// ================================================================================================ 5./6. dynamics
namespace dyn {
// Impurity (2 orbitals) driven by a C^infinity switch-on at t_on = 0.5 with a complex Hermitian A (so h(t) is
// complex Hermitian for t > t_on, real symmetric in equilibrium), optionally coupled to 2 undriven bath modes. With
// the bath, Sigma = V g_bath V^T is set exactly in both libraries (same model as test_bosonic.cpp e2e).
struct Model {
  int xi;
  bool bath;
  double beta = 5., mu, a = 0.5, ton = 0.5;
  Eigen::MatrixXd hfull, K0, Kb, V;
  CM A;
  int nf;
  Model(int xi_, bool bath_) : xi(xi_), bath(bath_) {
    hfull.resize(4, 4);
    hfull << 0.3, 0.2, 0.5, 0.3, 0.2, -0.4, -0.2, 0.4, 0.5, -0.2, 1.0, 0.0, 0.3, 0.4, 0.0, -0.7;
    mu = xi == 1 ? -1.4 : 0.1;
    nf = bath ? 4 : 2;
    K0 = hfull.block(0, 0, nf, nf) - mu * Eigen::MatrixXd::Identity(nf, nf);
    Kb = (hfull - mu * Eigen::MatrixXd::Identity(4, 4)).block(2, 2, 2, 2);
    V = hfull.block(0, 2, 2, 2);
    A.resize(2, 2);
    A << C(0.2, 0), C(0.6, 0.4), C(0.6, -0.4), C(-0.3, 0);
  }
  double drive(double t) const { return t > ton ? a * exp(-1.0 / (t - ton)) : 0.; }
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
    CM u = CM::Identity(M.nf, M.nf);
    U[0] = u;
    const int ns = 200;
    double dt = h / ns;
    auto f = [&](double tt, const CM &x) -> CM { return -IU * M.Kfull(tt) * x; };
    for (int n = 1; n < nt; n++) {
      for (int s = 0; s < ns; s++) {
        double t = (n - 1) * h + s * dt;
        CM k1 = f(t, u), k2 = f(t + dt / 2, u + dt / 2 * k1), k3 = f(t + dt / 2, u + dt / 2 * k2),
           k4 = f(t + dt, u + dt * k3);
        u += dt / 6 * (k1 + 2. * k2 + 2. * k3 + k4);
      }
      U[n] = u;
    }
  }
  double xi() const { return double(M.xi); }
  CM GR(int n, int m) const { return (-IU * U[n] * U[m].adjoint()).block(0, 0, 2, 2); }
  CM GL(int m, int n) const { return (-IU * xi() * U[m] * n0 * U[n].adjoint()).block(0, 0, 2, 2); }
  CM Gtv(int n, double tau) const { return (-IU * xi() * U[n] * n0 * expm_sym(M.K0, tau)).block(0, 0, 2, 2); }
  CM GM(double tau) const {
    CM one = CM::Identity(M.nf, M.nf);
    return (-(one + xi() * n0) * expm_sym(M.K0, -tau)).block(0, 0, 2, 2);
  }
  CM VgV(const CM &g) const { return M.bath ? CM(M.V.cast<C>() * g * M.V.transpose().cast<C>()) : CM(CM::Zero(2, 2)); }
  CM SR(int n, int l) const { return VgV(-IU * expm_sym(M.Kb, -IU * double(n - l) * h)); }
  CM SL(int l, int n) const { return VgV(-IU * xi() * expm_sym(M.Kb, -IU * double(l - n) * h) * nb); }
  CM Stv(int n, double tau) const { return VgV(-IU * xi() * expm_sym(M.Kb, -IU * double(n) * h) * nb * expm_sym(M.Kb, tau)); }
  CM SM(double tau) const {
    CM one = CM::Identity(2, 2);
    return VgV(-(one + xi() * nb) * expm_sym(M.Kb, -tau));
  }
};

struct Res {
  double dM = 0, dR = 0, dL = 0, dTV = 0;          // max distances H-NESSi <-> NESSi (curr rows at every T)
  double eH = 0, eN = 0;                           // max |G - exact| per entry, H-NESSi / NESSi (R, <, tv rows)
  double eHM = 0, eNM = 0;                         // same for G^M
  double sTV = 0;                                  // max |NESSi tv row| (scale)
};

Res run(int xi, bool bath, int k, double h, double tmax, int ntau) {
  Model Mo(xi, bath);
  const int NO = 2, es = 4;
  int nt = int(std::lround(tmax / h)) + 1, r = 0;
  dlr_info dlr(r, 60., 1e-13, Mo.beta, NO, xi);
  Exact E(Mo, nt, h);
  // ---- H-NESSi
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, 2, 1e-13, NO, NO, xi, k), Sig(nt, r, 2, 1e-13, NO, NO, xi, k);
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
    DMatrix s = E.SM(dlr.it0B(j)).real();
    Sig.set_mat(j, s);
  }
  for (int T = 0; T <= k; T++) set_sigma_row(T);
  for (int T = 0; T <= k; T++)
    for (int tp = T + 1; tp <= k; tp++) Sig.set_ret_curr(T, tp, ZMatrix(E.SR(T, tp)));
  dyson D(nt, NO, k, dlr);
  D.dyson_mat(G, Mo.mu, H, Sig, false, 1.0);
  G.initGMConvTensor(dlr);
  D.dyson_start(G, Mo.mu, H, Sig, I, h);
  // ---- NESSi (same h(t), same Sigma on its own grids)
  NGF Gn(nt - 1, ntau, NO, xi), Sn(nt - 1, ntau, NO, xi);
  cntr::function<double> Hn(nt - 1, NO);
  for (int t = -1; t < nt; t++) {
    cdmatrix m = Mo.Himp(t < 0 ? 0. : t * h);
    Hn.set_value(t, m);
  }
  double dtau = Mo.beta / ntau;
  for (int m = 0; m <= ntau; m++) {
    cdmatrix s = E.SM(m * dtau);
    Sn.set_mat(m, s);
  }
  for (int n = 0; n < nt; n++) {
    for (int l = 0; l <= n; l++) {
      cdmatrix sr = E.SR(n, l), sl = E.SL(l, n);
      Sn.set_ret(n, l, sr);
      Sn.set_les(l, n, sl);
    }
    for (int m = 0; m <= ntau; m++) {
      cdmatrix s = E.Stv(n, m * dtau);
      Sn.set_tv(n, m, s);
    }
  }
  // NESSi's default fixed-point Matsubara solver; CNTR_MAT_CG (tol 1e-12, 40 iterations) leaves a 2.5e-9 error in the
  // fermionic bath case, which then dominates the < distance (doc 05 §7).
  cntr::dyson_mat(Gn, Mo.mu, Hn, Sn, Mo.beta, k, CNTR_MAT_FIXPOINT);
  cntr::dyson_start(Gn, Mo.mu, Hn, Sn, Mo.beta, h, k);
  for (int n = k + 1; n < nt; n++) cntr::dyson_timestep(n, Gn, Mo.mu, Hn, Sn, Mo.beta, h, k);

  Res res;
  res.dM = distance_norm2_mat(G, Gn, dlr);
  for (int m = 0; m <= ntau; m++)
    res.eNM = std::max(res.eNM, (ZMatrixMap(Gn.matptr(m), 2, 2) - E.GM(m * dtau)).cwiseAbs().maxCoeff());
  DMatrix dm(NO, NO);
  for (int j = 0; j < r; j++) {
    G.get_mat(j, dm);
    res.eHM = std::max(res.eHM, (dm.cast<C>() - E.GM(dlr.it0B(j))).cwiseAbs().maxCoeff());
  }
  auto compare_row = [&](int T) {
    res.dR = std::max(res.dR, distance_norm2_curr_ret(T, G, Gn));
    res.dL = std::max(res.dL, distance_norm2_curr_les(T, G, Gn));
    res.dTV = std::max(res.dTV, distance_norm2_tv(T, G, Gn, dlr));
    res.sTV = std::max(res.sTV, nessi_tv_norm(Gn, T));
    ZMatrix m(NO, NO);
    for (int tp = 0; tp <= T; tp++) {
      G.get_ret_curr(T, tp, m);
      res.eH = std::max(res.eH, (CM(m) - E.GR(T, tp)).cwiseAbs().maxCoeff());
      G.get_les_curr(tp, T, m);
      res.eH = std::max(res.eH, (CM(m) - E.GL(tp, T)).cwiseAbs().maxCoeff());
      res.eN = std::max(res.eN, (ZMatrixMap(Gn.retptr(T, tp), 2, 2) - E.GR(T, tp)).cwiseAbs().maxCoeff());
      res.eN = std::max(res.eN, (ZMatrixMap(Gn.lesptr(tp, T), 2, 2) - E.GL(tp, T)).cwiseAbs().maxCoeff());
    }
    for (int j = 0; j < r; j++) {
      G.get_tv(T, j, m);
      res.eH = std::max(res.eH, (CM(m) - E.Gtv(T, dlr.it0B(j))).cwiseAbs().maxCoeff());
    }
    for (int q = 0; q <= ntau; q++)
      res.eN = std::max(res.eN, (ZMatrixMap(Gn.tvptr(T, q), 2, 2) - E.Gtv(T, q * dtau)).cwiseAbs().maxCoeff());
  };
  for (int T = 0; T <= k; T++) compare_row(T);
  for (int T = k + 1; T < nt; T++) {
    G.update_blocks(I);
    Sig.update_blocks(I);
    D.extrapolate(G, I);
    set_sigma_row(T);
    D.dyson_timestep(T, G, Mo.mu, H, Sig, I, h);
    compare_row(T);
  }
  (void)es;
  return res;
}

// distances vs h; bounded by the libraries' own errors (Frobenius norm over a row of up to nt (R, <) or ntau+1
// (tv) entries of max-norm error e: <= 2 sqrt(#entries) (eH + eN)); order ~k+1 in h for R and <.
void check(int xi, bool bath) {
  const int k = 4, ntau = 400;
  const double tmax = 1.6;
  std::vector<double> hs = {0.05, 0.025, 0.0125};
  std::vector<Res> R;
  for (double h : hs) R.push_back(run(xi, bath, k, h, tmax, ntau));
  for (size_t i = 0; i < hs.size(); i++)
    std::printf("[dyn] xi=%+d bath=%d k=%d h=%.4f  dist: M %.2e R %.2e L %.2e TV %.2e (|tv| %.2e) | err H-NESSi "
                "%.2e (M %.2e)  NESSi %.2e (M %.2e)\n",
                xi, int(bath), k, hs[i], R[i].dM, R[i].dR, R[i].dL, R[i].dTV, R[i].sTV, R[i].eH, R[i].eHM, R[i].eN,
                R[i].eNM);
  for (size_t i = 0; i < hs.size(); i++) {
    int nt = int(std::lround(tmax / hs[i])) + 1;
    double bRL = 2. * 2. * std::sqrt(double(nt)) * (R[i].eH + R[i].eN) + 1e-12;
    double bTV = 2. * 2. * std::sqrt(double(ntau + 1)) * (R[i].eH + R[i].eN) + 1e-12;
    double bM = 2. * 2. * std::sqrt(double(ntau + 1)) * (R[i].eHM + R[i].eNM) + 1e-10;
    EXPECT_LT(R[i].dR, bRL) << "h=" << hs[i];
    EXPECT_LT(R[i].dL, bRL) << "h=" << hs[i];
    EXPECT_LT(R[i].dTV, bTV) << "h=" << hs[i];
    EXPECT_LT(R[i].dM, bM) << "h=" << hs[i];
  }
  // convergence of the distance (both libraries converge to the same exact G)
  const size_t f = hs.size() - 1;
  EXPECT_GT(hn_test::rate(R[f - 1].dR, R[f].dR), k - 0.5) << "R";
  EXPECT_GT(hn_test::rate(R[f - 1].dL, R[f].dL), k - 0.5) << "<";
  EXPECT_LT(R[f].dR, 1e-6);
  EXPECT_LT(R[f].dL, 1e-6);
  // mat and tv: both libraries are converged in tau (NESSi ntau = 400, order k; DLR eps 1e-13), and the tv rows of
  // both propagations agree to ~1e-12 (observed 2e-13 .. 4e-12): the distance is at the DLR/tau-grid error level.
  for (size_t i = 0; i < hs.size(); i++) {
    EXPECT_LT(R[i].dM, 1e-10) << "h=" << hs[i];
    EXPECT_LT(R[i].dTV, 1e-10 * R[i].sTV) << "h=" << hs[i];
  }
}
}  // namespace dyn

TEST(NessiDynamics, NoninteractingDrivenFermion) { dyn::check(-1, false); }
TEST(NessiDynamics, NoninteractingDrivenBoson) { dyn::check(+1, false); }
TEST(NessiDynamics, FixedBathSigmaFermion) { dyn::check(-1, true); }
TEST(NessiDynamics, FixedBathSigmaBoson) { dyn::check(+1, true); }
