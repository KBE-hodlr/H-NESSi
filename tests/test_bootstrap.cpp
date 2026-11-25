// Bootstrap T <= k (doc 22): dyson_start (TTI), dyson_start_ntti, _ntti_nobc, _2leg.
//  (a) library vs an independent dense (Kronecker-vectorised, untransposed) reimplementation of the CORRECT
//      discrete systems (agent_scratch/22/start_test.cpp).  Unlike the scratch program, the reference uses the
//      smooth continuation Sigma^R(n,l>n) = -[Sigma^R(l,n)]^dagger (B12) and solves the right-acting imp_tp0
//      equation X*A = Q (B13).
//  (b) error vs the exact solution of an impurity+bath model, convergence order k+1.
//  (c) agent_scratch/51/t_start*.cpp: order of dyson_start_ret_ntti with the slot (n,l>n) left zero (B12) and
//      with imp_tp0 (B13); (d) B25: dyson_start_2leg(function&) must forward imp_tp0; (e) B31: imp_tp0 error.
//  B12, B13, B25 and B31 are fixed in WP1; their former KnownBug_* tests are ordinary tests now.
//  (f) B34 (WP7): with RHO_DIAGONAL the lesser row solves of dyson_start_ntti / _ntti_nobc treat G^<(m,m) as known
//      (the diagonal value before the call) and the diagonal redo evaluates B with the diagonal values, so the
//      converged bootstrap diagonal solves its own discrete diagonal equation (as the time stepper's does).  The
//      dense reference implements this independently (column m moved to the right-hand side, equation n = m dropped).
//      _2leg keeps the old scheme (not part of B34's fix).  Closed-system tests (Sigma = 0, both xi): residual of the
//      diagonal equation, anti-Hermiticity and particle number over the bootstrap window, order k+1.
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <cstdio>
#include <unsupported/Eigen/MatrixFunctions>

#include "common/test_util.hpp"
#include "h_nessi/dlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/function.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/integration.hpp"

using namespace h_nessi;
typedef std::complex<double> C;
typedef Eigen::MatrixXcd CM;
typedef Eigen::VectorXcd CV;

namespace bs {
const C II(0, 1);
const int NO = 2, NF = 4;

struct Model {
  double beta = 5., mu = 0.1, a = 0.5, w = 2.0;
  Eigen::MatrixXd K0;
  CM Aimp;
  Eigen::MatrixXd V;
  Model() {
    Eigen::MatrixXd h(NF, NF);
    h << 0.3, 0.2, 0.5, 0.3, 0.2, -0.4, -0.2, 0.4, 0.5, -0.2, 1.0, 0.0, 0.3, 0.4, 0.0, -0.7;
    V = h.block(0, 2, 2, 2);
    K0 = h - mu * Eigen::MatrixXd::Identity(NF, NF);
    Aimp.resize(2, 2);
    Aimp << C(0.2, 0), C(0.6, 0.4), C(0.6, -0.4), C(-0.3, 0);
  }
  CM Himp(double t) const {
    CM H = (K0.block(0, 0, 2, 2) + mu * Eigen::MatrixXd::Identity(2, 2)).cast<C>();
    return H + a * sin(w * t) * Aimp;
  }
  CM Kfull(double t) const {
    CM K = K0.cast<C>();
    K.block(0, 0, 2, 2) += a * sin(w * t) * Aimp;
    return K;
  }
};

CM fermi(const Eigen::MatrixXd &K, double beta) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  Eigen::VectorXd d(K.rows());
  for (int i = 0; i < K.rows(); i++) d(i) = 1. / (exp(beta * es.eigenvalues()(i)) + 1.);
  return (es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose()).cast<C>();
}
CM expm_sym(const Eigen::MatrixXd &K, C s) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  CV d(K.rows());
  for (int i = 0; i < K.rows(); i++) d(i) = exp(s * es.eigenvalues()(i));
  CM U = es.eigenvectors().cast<C>();
  return U * d.asDiagonal() * U.transpose();
}

struct Exact {
  int k;
  double h;
  bool twoleg;
  std::vector<CM> U;
  CM n0;
  const Model &M;
  Eigen::MatrixXd Kb;
  CM fb;
  Exact(const Model &M_, int k_, double h_, bool tl) : k(k_), h(h_), twoleg(tl), M(M_) {
    Kb = M.K0.block(2, 2, 2, 2);
    fb = fermi(Kb, M.beta);
    if (!twoleg) n0 = fermi(M.K0, M.beta);
    else {
      n0 = CM::Zero(NF, NF);
      n0.block(0, 0, 2, 2) = fermi(M.K0.block(0, 0, 2, 2), M.beta);
      n0.block(2, 2, 2, 2) = fb;
    }
    U.resize(k + 1);
    CM u = CM::Identity(NF, NF);
    U[0] = u;
    int ns = 4000;
    double dt = h / ns;
    for (int n = 1; n <= k; n++) {
      for (int s = 0; s < ns; s++) {
        double t = (n - 1) * h + s * dt;
        auto f = [&](double tt, const CM &x) -> CM { return -II * M.Kfull(tt) * x; };
        CM k1 = f(t, u), k2 = f(t + dt / 2, u + dt / 2 * k1), k3 = f(t + dt / 2, u + dt / 2 * k2), k4 = f(t + dt, u + dt * k3);
        u += dt / 6 * (k1 + 2. * k2 + 2. * k3 + k4);
      }
      U[n] = u;
    }
  }
  CM GR(int n, int m) const { return (-II * U[n] * U[m].adjoint()).block(0, 0, 2, 2); }
  CM GL(int m, int n) const { return (II * U[m] * n0 * U[n].adjoint()).block(0, 0, 2, 2); }
  CM Gtv(int n, double tau) const { return (II * U[n] * n0 * expm_sym(M.K0, tau)).block(0, 0, 2, 2); }
  CM GM(double tau) const {
    CM one = CM::Identity(NF, NF);
    return (-(one - n0) * expm_sym(M.K0, -tau)).block(0, 0, 2, 2);
  }
  CM SR(int n, int l) const { return M.V.cast<C>() * (-II * expm_sym(Kb, -II * double(n - l) * h)) * M.V.transpose().cast<C>(); }
  CM SL(int l, int n) const { return M.V.cast<C>() * (II * expm_sym(Kb, -II * double(l - n) * h) * fb) * M.V.transpose().cast<C>(); }
  CM Stv(int n, double tau) const {
    return M.V.cast<C>() * (II * expm_sym(Kb, -II * double(n) * h) * fb * expm_sym(Kb, tau)) * M.V.transpose().cast<C>();
  }
};

CM kron(const CM &P, const CM &Q) {
  CM R(P.rows() * Q.rows(), P.cols() * Q.cols());
  for (int i = 0; i < P.rows(); i++)
    for (int j = 0; j < P.cols(); j++) R.block(i * Q.rows(), j * Q.cols(), Q.rows(), Q.cols()) = P(i, j) * Q;
  return R;
}
CM ID() { return CM::Identity(NO, NO); }
CM Lop(const CM &A) { return kron(ID(), A); }
CM Rop(const CM &B) { return kron(B.transpose(), ID()); }
CV vec(const CM &Y) {
  CV v(NO * NO);
  for (int j = 0; j < NO; j++)
    for (int i = 0; i < NO; i++) v(i + j * NO) = Y(i, j);
  return v;
}
CM unvec(const CV &v, int off) {
  CM Y(NO, NO);
  for (int j = 0; j < NO; j++)
    for (int i = 0; i < NO; i++) Y(i, j) = v(off + i + j * NO);
  return Y;
}
CM getmap(cplx *p) { return ZMatrixMap(p, NO, NO); }
void putmap(cplx *p, const CM &Y) { ZMatrixMap(p, NO, NO) = Y; }

enum SlotMode { SLOT_ZERO = 0, SLOT_FIXEQUIV = 1 };
enum LesMode { LES_BC = 0, LES_NOBC = 1, LES_2LEG = 2 };

struct Result {
  std::vector<std::vector<CM>> R, L, tv;
};

// Dense reference of the CORRECT discrete systems.  Black boxes (verified in doc 03 / test_dlr_matsubara):
// tv_it_conv, les_it_int, get_tv_tau, get_mat_reversed.
Result reference(int k, double h, double mu, function &H, herm_matrix_hodlr &Sig, herm_matrix_hodlr &Gref, dyson &Dref,
                 dlr_info &dlr, Integration::Integrator &I, bool tti, LesMode lm, bool rhodiag, bool imp_tp0, const CM &GL00,
                 const std::vector<CM> &Ldiag0) {
  int r = dlr.r(), es = NO * NO;
  double beta = dlr.beta();
  int xi = -1;
  Result res;
  res.R.assign(k + 1, std::vector<CM>(k + 1, CM::Zero(NO, NO)));
  res.L = res.R;
  res.tv.assign(k + 1, std::vector<CM>(r, CM::Zero(NO, NO)));
  auto D = [&](int m, int l) { return I.poly_diff(m, l); };
  auto Iw = [&](int m, int n, int l) { return I.poly_integ(m, n, l); };
  auto SR = [&](int n, int l) { return getmap(Sig.curr_timestep_ret_ptr(n, l)); };  // n>=l
  auto SL = [&](int l, int n) { return getmap(Sig.curr_timestep_les_ptr(l, n)); };  // l<=n
  auto eps = [&](int n) { return CM(getmap(H.ptr(n)) - mu * ID()); };
  auto SRt = [&](int n, int l) -> CM { return l <= n ? CM(SR(n, l)) : CM(-SR(l, n).adjoint()); };
  auto SLt = [&](int l, int n) -> CM { return l <= n ? CM(SL(l, n)) : CM(-SL(n, l).adjoint()); };
  auto SA = [&](int l, int n) -> CM { return SRt(n, l).adjoint(); };

  bool do_tv = (lm != LES_2LEG);
  if (do_tv) {
    std::vector<double> rev(r * es);
    Gref.get_mat_reversed(dlr, rev.data());
    for (int j = 0; j < r; j++) {
      CM g0(NO, NO);
      for (int a = 0; a < NO; a++)
        for (int b = 0; b < NO; b++) g0(a, b) = double(xi) * II * rev[j * es + a * NO + b];
      res.tv[0][j] = g0;
    }
    for (int j = 0; j < r; j++) {
      CM A = CM::Zero(k * es, k * es);
      CV b = CV::Zero(k * es);
      for (int n = 1; n <= k; n++) {
        ZMatrix im = ZMatrix::Zero(NO, NO);
        Dref.tv_it_conv(j, n, Sig, Gref, im.data());
        CM rhs = CM(im) - II / h * D(n, 0) * res.tv[0][j] + h * Iw(0, n, 0) * SR(n, 0) * res.tv[0][j];
        b.segment((n - 1) * es, es) = vec(rhs);
        for (int l = 1; l <= k; l++) {
          CM blk = II / h * D(n, l) * CM::Identity(es, es) - h * Iw(0, n, l) * Lop(SRt(n, l));
          if (l == n) blk -= Lop(eps(n));
          A.block((n - 1) * es, (l - 1) * es, es, es) = blk;
        }
      }
      CV x = A.partialPivLu().solve(b);
      for (int n = 1; n <= k; n++) res.tv[n][j] = unvec(x, (n - 1) * es);
    }
    for (int n = 0; n <= k; n++)
      for (int j = 0; j < r; j++) Gref.set_tv(n, j, ZMatrix(res.tv[n][j]));
  }

  if (tti) {
    for (int n = 0; n <= k; n++) {
      ZMatrix g0(NO, NO), gb(NO, NO);
      Gref.get_tv_tau(n, 0., dlr, g0);
      Gref.get_tv_tau(n, beta, dlr, gb);
      CM grn = (n == 0) ? CM(-II * ID()) : CM(-CM(g0) + double(xi) * CM(gb));
      for (int l = 0; l + n <= k; l++) res.R[n + l][l] = grn;
      CM gl = -CM(g0).adjoint();
      for (int l = 0; l + n <= k; l++) res.L[l][l + n] = gl;
    }
    return res;
  }

  // retarded, column by column; for l > n the integrand continuation is Sigma^R(n,l) = -[Sigma^R(l,n)]^dagger
  for (int i = 0; i <= k; i++) res.R[i][i] = -II * ID();
  for (int m = 0; m < k; m++) {
    int K = k - m;
    CM A = CM::Zero(K * es, K * es);
    CV b = CV::Zero(K * es);
    for (int n = m + 1; n <= k; n++) {
      CM rhs = CM::Zero(NO, NO);
      for (int l = 0; l <= m; l++) {
        CM gc = -res.R[m][l].adjoint();
        rhs -= (II / h * D(n, l) * ID() - h * Iw(m, n, l) * SR(n, l)) * gc;
      }
      b.segment((n - m - 1) * es, es) = vec(rhs);
      for (int l = m + 1; l <= k; l++) {
        CM blk = II / h * D(n, l) * CM::Identity(es, es) - h * Iw(m, n, l) * Lop(SRt(n, l));
        if (l == n) blk -= Lop(eps(n));
        A.block((n - m - 1) * es, (l - m - 1) * es, es, es) = blk;
      }
    }
    CV x = A.partialPivLu().solve(b);
    for (int n = m + 1; n <= k; n++) res.R[n][m] = unvec(x, (n - m - 1) * es);
  }
  if (imp_tp0) {  // right-acting equation for G^R(m,0): X * (i/h D_kk - eps(0) - h I Sigma^R(0,0)) = Q
    for (int m = 1; m <= k; m++) {
      CM Q = CM::Zero(NO, NO);
      for (int i = 0; i < k - m; i++) Q += II / h * D(k, i) * res.R[k - i][m].adjoint();
      for (int i = k - m; i < k; i++) Q += -II / h * D(k, i) * res.R[m][k - i];
      for (int i = 0; i < k - m; i++) Q -= h * Iw(k - m, k, i) * res.R[k - i][m].adjoint() * SR(k - i, 0);
      for (int i = k - m; i < k; i++) Q += h * Iw(k - m, k, i) * res.R[m][k - i] * SR(k - i, 0);
      CM Rm = II / h * D(k, k) * ID() - eps(0) - h * Iw(k - m, k, k) * SR(0, 0);
      res.R[m][0] = Rm.transpose().partialPivLu().solve(Q.transpose()).transpose();
    }
  }

  // lesser
  auto Gtl = [&](int m, int l) -> CM { return l <= m ? CM(res.R[m][l]) : CM(-res.R[l][m].adjoint()); };
  auto mix = [&](int t, int tp) -> CM {
    if (lm == LES_2LEG) return CM::Zero(NO, NO);
    ZMatrix z = ZMatrix::Zero(NO, NO);
    Dref.les_it_int(t, tp, Gref, Sig, z.data());
    return CM(z).transpose();
  };
  // B34: RHO_DIAGONAL (ntti, nobc): in rows m >= 1 the diagonal G^<(m,m) = Ldiag0[m] is known; its column goes to
  // the right-hand side and the equation at the diagonal node n = m is dropped ((k-1) unknowns, (k-1) equations).
  const bool consistent = rhodiag && lm != LES_2LEG;
  auto solve_row = [&](int m, const CM &Gm0) {
    const int known = (consistent && m >= 1) ? m : -1;
    std::vector<int> idx;  // unknown nodes = equation nodes
    for (int n = 1; n <= k; n++)
      if (n != known) idx.push_back(n);
    const int K = idx.size();
    CM A = CM::Zero(K * es, K * es);
    CV b = CV::Zero(K * es);
    for (int a = 0; a < K; a++) {
      int n = idx[a];
      CM rhs = CM::Zero(NO, NO);
      for (int l = 0; l <= k; l++) rhs += h * Iw(0, m, l) * Gtl(m, l) * SLt(l, n);
      rhs += mix(m, n);
      rhs += II / h * D(n, 0) * Gm0 + h * Iw(0, n, 0) * Gm0 * SA(0, n);
      if (known > 0) {  // move the known column l = m to the right-hand side
        const CM &Gmm = Ldiag0[m];
        rhs += II / h * D(n, m) * Gmm + h * Iw(0, n, m) * Gmm * SA(m, n);
      }
      b.segment(a * es, es) = vec(rhs);
      for (int c = 0; c < K; c++) {
        int l = idx[c];
        CM blk = -II / h * D(n, l) * CM::Identity(es, es) - h * Iw(0, n, l) * Rop(SA(l, n));
        if (l == n) blk -= Rop(eps(n));
        A.block(a * es, c * es, es, es) = blk;
      }
    }
    CV x = A.partialPivLu().solve(b);
    std::vector<CM> Y(k + 1);
    Y[0] = Gm0;
    for (int a = 0; a < K; a++) Y[idx[a]] = unvec(x, a * es);
    if (known > 0) Y[m] = Ldiag0[m];
    return Y;
  };
  if (lm == LES_BC) {
    for (int tp = 0; tp <= k; tp++) {
      ZMatrix g0(NO, NO);
      Gref.get_tv_tau(tp, 0., dlr, g0);
      res.L[0][tp] = -CM(g0).adjoint();
    }
  } else {
    res.L[0][0] = GL00;
    auto Y = solve_row(0, GL00);
    for (int n = 1; n <= k; n++) res.L[0][n] = Y[n];
  }
  for (int m = 1; m <= k; m++) {
    auto Y = solve_row(m, CM(-res.L[0][m].adjoint()));
    for (int n = m; n <= k; n++) res.L[m][n] = Y[n];
  }
  if (rhodiag) {
    // consistent (B34): res.L[i][i] is the diagonal value before the call (the rows did not overwrite it), and the
    // l == i term of int G^< Sigma^A uses G^<(i,i) itself.  _2leg (old scheme): this call's horizontal value, and the
    // l == i term uses -G^<(i,i)^dagger.
    CM A = CM::Zero(k * es, k * es);
    CV b = CV::Zero(k * es);
    for (int i = 1; i <= k; i++) {
      CM B = res.L[i][i] * (getmap(H.ptr(i)).adjoint() - mu * ID());
      for (int l = 0; l <= k; l++) B += h * Iw(0, i, l) * Gtl(i, l) * SLt(l, i);
      for (int l = 0; l <= k; l++) {
        CM gl = (l < i || (l == i && !consistent)) ? CM(-res.L[l][i].adjoint()) : CM(res.L[i][l]);
        B += h * Iw(0, i, l) * gl * SA(l, i);
      }
      B += mix(i, i);
      B *= II;
      CM rhs = B - B.adjoint() - 1. / h * D(i, 0) * res.L[0][0];
      b.segment((i - 1) * es, es) = vec(rhs);
      for (int j = 1; j <= k; j++) A.block((i - 1) * es, (j - 1) * es, es, es) = 1. / h * D(i, j) * CM::Identity(es, es);
    }
    CV x = A.partialPivLu().solve(b);
    for (int i = 1; i <= k; i++) res.L[i][i] = unvec(x, (i - 1) * es);
  }
  return res;
}

// variant: 0 dyson_start, 1 dyson_start_ntti, 2 dyson_start_ntti_nobc, 3 dyson_start_2leg
struct Case { int variant; bool tti; LesMode lm; bool rhodiag; bool imp; SlotMode sm; };
struct Errs { double R = 0, L = 0, tv = 0, libref = 0, fixpt = 0; int iters = 1; };

Errs run_case(const Case &cs, int k, double h) {
  Model M;
  int xi = -1, r = 0;
  dlr_info dlr(r, 60., 1e-13, M.beta, NO, xi);
  Model Mc = M;
  Mc.a = cs.tti ? 0. : M.a;
  Exact E(Mc, k, h, cs.lm == LES_2LEG);
  int nt = k + 4, nlvl = 1;
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, nlvl, 1e-10, NO, NO, xi, k), Sig(nt, r, nlvl, 1e-10, NO, NO, xi, k), Gref(nt, r, nlvl, 1e-10, NO, NO, xi, k);
  function H(nt, NO, NO);
  for (int t = -1; t < nt; t++) H.get_map(t) = ZMatrix(Mc.Himp(t < 0 ? 0. : t * h));
  for (int j = 0; j < r; j++) {
    DMatrix d = E.GM(dlr.it0B()[j]).real();
    G.set_mat(j, d);
    Gref.set_mat(j, d);
  }
  G.initGMConvTensor(dlr);
  Gref.initGMConvTensor(dlr);
  for (int n = 0; n <= k; n++) {
    for (int l = 0; l <= n; l++) putmap(Sig.curr_timestep_ret_ptr(n, l), E.SR(n, l));
    for (int l = n + 1; l <= k; l++) putmap(Sig.curr_timestep_ret_ptr(n, l), (cs.sm == SLOT_ZERO) ? CM(CM::Zero(NO, NO)) : E.SR(l, n));
    for (int l = 0; l <= n; l++) putmap(Sig.curr_timestep_les_ptr(l, n), E.SL(l, n));
    for (int j = 0; j < r; j++) Sig.set_tv(n, j, ZMatrix(cs.lm == LES_2LEG ? CM(CM::Zero(NO, NO)) : E.Stv(n, dlr.it0B()[j])));
  }
  CM GL00 = E.GL(0, 0);
  putmap(G.curr_timestep_les_ptr(0, 0), GL00);
  dyson D(nt, NO, k, dlr, cs.rhodiag, false), Dref(nt, NO, k, dlr, cs.rhodiag, false);
  auto call = [&]() {
    if (cs.variant == 0) return D.dyson_start(G, Mc.mu, H, Sig, I, h);
    if (cs.variant == 1) return D.dyson_start_ntti(G, Mc.mu, H, Sig, I, h, cs.imp);
    if (cs.variant == 2) return D.dyson_start_ntti_nobc(G, Mc.mu, H, Sig, I, h, cs.imp);
    return D.dyson_start_2leg(G, Mc.mu, H, Sig, I, h, cs.imp);
  };
  auto diag = [&]() {
    std::vector<CM> d(k + 1);
    for (int i = 0; i <= k; i++) d[i] = getmap(G.curr_timestep_les_ptr(i, i));
    return d;
  };
  // the reference never reads the slot (n, l>n): it uses the continuation
  auto ref = [&](const std::vector<CM> &d0) { return reference(k, h, Mc.mu, H, Sig, Gref, Dref, dlr, I, cs.tti, cs.lm, cs.rhodiag, cs.imp, GL00, d0); };
  Errs e;
  std::vector<CM> d0 = diag();
  double err = call();
  Result R = ref(d0);
  // B34: with RHO_DIAGONAL (ntti, nobc) one call is one fixed-point iteration of the diagonal (Sigma fixed here):
  // the first call is checked against the reference (initial guess G^<(i,i) = 0 for i >= 1), then iterate to
  // convergence and check the last call (its input diagonal) and the fixed point.
  if (cs.rhodiag && cs.lm != LES_2LEG && !cs.tti) {
    for (int n = 0; n <= k; n++)
      for (int m = 0; m <= n; m++)
        e.libref = std::max(e.libref, (getmap(G.curr_timestep_les_ptr(m, n)) - R.L[m][n]).norm());
    while (err > 1e-13 && e.iters < 200) {
      d0 = diag();
      err = call();
      e.iters++;
    }
    R = ref(d0);
    Result Rfix = ref(diag());  // one more pass of the reference on the converged diagonal must reproduce it
    for (int i = 1; i <= k; i++) e.fixpt = std::max(e.fixpt, (Rfix.L[i][i] - getmap(G.curr_timestep_les_ptr(i, i))).norm());
  }
  for (int n = 0; n <= k; n++)
    for (int m = 0; m <= n; m++) {
      CM gr = getmap(G.curr_timestep_ret_ptr(n, m)), gl = getmap(G.curr_timestep_les_ptr(m, n));
      e.libref = std::max(e.libref, (gr - R.R[n][m]).norm());
      e.libref = std::max(e.libref, (gl - R.L[m][n]).norm());
      e.R = std::max(e.R, (gr - E.GR(n, m)).norm());
      e.L = std::max(e.L, (gl - E.GL(m, n)).norm());
    }
  if (cs.lm != LES_2LEG)
    for (int n = 0; n <= k; n++)
      for (int j = 0; j < r; j++) {
        CM g = getmap(G.tvptr(n, j));
        e.libref = std::max(e.libref, (g - R.tv[n][j]).norm());
        e.tv = std::max(e.tv, (g - E.Gtv(n, dlr.it0B()[j])).norm());
      }
  return e;
}

// lib vs dense reference for k = 1..5 at h = h0, and convergence order k+1 vs exact from h = h0 -> h0/2
void check_case(const Case &cs, int kmax = 5, double h0 = 0.1) {
  for (int k = 1; k <= kmax; k++) {
    Errs a = run_case(cs, k, h0), b = run_case(cs, k, h0 / 2);
    EXPECT_LT(a.libref, 1e-13) << "k=" << k << " library vs dense reference";
    EXPECT_LT(b.libref, 1e-13) << "k=" << k << " library vs dense reference";
    EXPECT_LT(a.fixpt, 1e-13) << "k=" << k << " converged diagonal is a fixed point of the diagonal equation (B34)";
    EXPECT_LT(b.fixpt, 1e-13) << "k=" << k << " converged diagonal is a fixed point of the diagonal equation (B34)";
    EXPECT_LT(a.iters, 200) << "k=" << k << " bootstrap iteration must converge";
    double want = k + 1 - 0.5;  // h = 0.1 -> 0.05 is slightly pre-asymptotic for k = 5
    EXPECT_GT(hn_test::rate(a.R, b.R), want) << "k=" << k << " R " << a.R << " -> " << b.R;
    EXPECT_GT(hn_test::rate(a.L, b.L), want) << "k=" << k << " < " << a.L << " -> " << b.L;
    if (cs.lm != LES_2LEG) EXPECT_GT(hn_test::rate(a.tv, b.tv), want) << "k=" << k << " tv " << a.tv << " -> " << b.tv;
  }
}
void check_case_sweep(const Case &cs) {
  std::vector<double> hs = {0.4, 0.2, 0.1, 0.05, 0.025};
  for (int k = 1; k <= 5; k++) {
    std::vector<Errs> e;
    for (double h : hs) e.push_back(run_case(cs, k, h));
    for (size_t i = 3; i < hs.size(); i++) {  // pairs 0.1->0.05, 0.05->0.025 (coarser ones are pre-asymptotic)
      if (e[i].R < 1e-11) continue;
      if (cs.imp && k == 5 && i == 3) continue;  // imp_tp0, k=5: G^R(m,0) still pre-asymptotic at h=0.1 (rate 8.2, 4.3, 5.7)
      EXPECT_GT(hn_test::rate(e[i - 1].R, e[i].R), k + 1 - 0.5) << "k=" << k << " h=" << hs[i];
      EXPECT_GT(hn_test::rate(e[i - 1].L, e[i].L), k + 1 - 0.5) << "k=" << k << " h=" << hs[i];
    }
  }
}
}  // namespace bs

using bs::Case;
using bs::LES_2LEG;
using bs::LES_BC;
using bs::LES_NOBC;
using bs::SLOT_FIXEQUIV;
using bs::SLOT_ZERO;

// --- correct today: TTI, and non-TTI with the slot (n,l>n) holding the continuation of Sigma
TEST(Bootstrap, StartTTI) { bs::check_case({0, true, LES_BC, false, false, SLOT_ZERO}); }
TEST(Bootstrap, NttiHorizSlotFilled) { bs::check_case({1, false, LES_BC, false, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, NttiDiagSlotFilled) { bs::check_case({1, false, LES_BC, true, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, NobcHorizSlotFilled) { bs::check_case({2, false, LES_NOBC, false, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, NobcDiagSlotFilled) { bs::check_case({2, false, LES_NOBC, true, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, TwoLegHorizSlotFilled) { bs::check_case({3, false, LES_2LEG, false, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, TwoLegDiagSlotFilled) { bs::check_case({3, false, LES_2LEG, true, false, SLOT_FIXEQUIV}); }

// --- B12 (fixed in WP1): what every user has: the slot (n,l>n) is never written (zero); the code must use Sigma^R(l,n)
TEST(Bootstrap, NttiHorizSlotZero) { bs::check_case({1, false, LES_BC, false, false, SLOT_ZERO}); }
TEST(Bootstrap, NttiDiagSlotZero) { bs::check_case({1, false, LES_BC, true, false, SLOT_ZERO}); }
TEST(Bootstrap, TwoLegDiagSlotZero) { bs::check_case({3, false, LES_2LEG, true, false, SLOT_ZERO}); }
// --- B13 (fixed in WP1): imp_tp0 = true with N_o = 2 (right-acting system)
// (h = 0.05 -> 0.025: at k = 5 the imp_tp0 column G^R(m,0) is pre-asymptotic at h = 0.1, rate 4.3 for 0.1 -> 0.05)
TEST(Bootstrap, NttiImpTp0) { bs::check_case({1, false, LES_BC, false, true, SLOT_FIXEQUIV}, 5, 0.05); }

TEST(Bootstrap, Slow_StartTTISweep) { bs::check_case_sweep({0, true, LES_BC, false, false, SLOT_ZERO}); }
TEST(Bootstrap, Slow_NttiDiagSlotFilledSweep) { bs::check_case_sweep({1, false, LES_BC, true, false, SLOT_FIXEQUIV}); }
TEST(Bootstrap, Slow_TwoLegHorizSlotFilledSweep) { bs::check_case_sweep({3, false, LES_2LEG, false, false, SLOT_FIXEQUIV}); }
// WP1: the same sweeps as a user gets them (slot (n,l>n) of Sigma^R zero), for ntti, ntti_nobc and 2leg, and imp_tp0
TEST(Bootstrap, Slow_NttiHorizSlotZeroSweep) { bs::check_case_sweep({1, false, LES_BC, false, false, SLOT_ZERO}); }
TEST(Bootstrap, Slow_NobcDiagSlotZeroSweep) { bs::check_case_sweep({2, false, LES_NOBC, true, false, SLOT_ZERO}); }
TEST(Bootstrap, Slow_TwoLegHorizSlotZeroSweep) { bs::check_case_sweep({3, false, LES_2LEG, false, false, SLOT_ZERO}); }
TEST(Bootstrap, Slow_NttiImpTp0SlotZeroSweep) { bs::check_case_sweep({1, false, LES_BC, false, true, SLOT_ZERO}); }

// =============================================================================================== B34 (WP7)
// Closed driven system (Sigma = 0), N_o = 3, both statistics, RHO_DIAGONAL, dyson_start_ntti and _ntti_nobc iterated
// to convergence.  Over the bootstrap window t <= k: the diagonal solves its own discrete diagonal equation
//   (1/h) sum_j D_ij G^<(j,j) = B_i - B_i^dagger,  B_i = i G^<(i,i) (h(i) - mu)
// to roundoff (B34: before WP7 the right-hand side used the horizontal row values, residual ~1e-5 at k = 3, h = 0.05),
// G^<(t,t) is anti-Hermitian and N(t) = tr(xi i G^<(t,t)) is conserved to roundoff, and the order is k+1.
namespace cl {
const C II(0, 1);
const int NO = 3;
CM getmap(cplx *p) { return ZMatrixMap(p, NO, NO); }
void putmap(cplx *p, const CM &Y) { ZMatrixMap(p, NO, NO) = Y; }
struct Out { double res = 0, ah = 0, dn = 0, eL = 0, eR = 0; int iters = 0; };
Out run(int xi, int k, double h, bool nobc) {
  Eigen::MatrixXd H0(NO, NO);
  H0 << 0.4, 0.3, -0.2, 0.3, -0.5, 0.25, -0.2, 0.25, 0.9;
  CM A(NO, NO);
  A << C(0.3, 0), C(0.2, 0.35), C(-0.1, 0.2), C(0.2, -0.35), C(-0.2, 0), C(0.15, -0.1), C(-0.1, -0.2), C(0.15, 0.1), C(0.1, 0);
  const double beta = 2., a = 0.8, w = 2.5;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es0(H0);
  const double mu = (xi == 1) ? es0.eigenvalues()(0) - 0.5 : 0.1;  // bosons: h - mu > 0
  Eigen::MatrixXd K = H0 - mu * Eigen::MatrixXd::Identity(NO, NO);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(K);
  auto fK = [&](std::function<double(double)> f) {
    Eigen::VectorXd d(NO);
    for (int i = 0; i < NO; i++) d(i) = f(es.eigenvalues()(i));
    return CM((es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose()).cast<C>());
  };
  CM rho = fK([&](double e) { return 1. / (std::exp(beta * e) - double(xi)); });
  auto Ht = [&](double t) { return CM(H0.cast<C>() + a * std::sin(w * t) * A); };
  std::vector<CM> U(k + 1);
  U[0] = CM::Identity(NO, NO);
  const int ns = 2000;
  const double dt = h / ns;
  for (int n = 1; n <= k; n++) {
    CM u = U[n - 1];
    for (int q = 0; q < ns; q++) {
      double t = (n - 1) * h + q * dt;
      auto f = [&](double tt, const CM &x) -> CM { return -II * (Ht(tt) - mu * CM::Identity(NO, NO)) * x; };
      CM k1 = f(t, u), k2 = f(t + dt / 2, u + dt / 2 * k1), k3 = f(t + dt / 2, u + dt / 2 * k2), k4 = f(t + dt, u + dt * k3);
      u += dt / 6 * (k1 + 2. * k2 + 2. * k3 + k4);
    }
    U[n] = u;
  }
  auto exGL = [&](int n, int m) { return CM(-II * double(xi) * U[n] * rho * U[m].adjoint()); };
  auto exGR = [&](int n, int m) { return CM(-II * U[n] * U[m].adjoint()); };
  int r = 0;
  dlr_info dlr(r, 40., 1e-13, beta, NO, xi);
  int nt = k + 4;
  Integration::Integrator I(k);
  herm_matrix_hodlr G(nt, r, 1, 1e-12, NO, NO, xi, k), Sig(nt, r, 1, 1e-12, NO, NO, xi, k);
  function H(nt, NO, NO);
  for (int t = -1; t < nt; t++) H.get_map(t) = ZMatrix(Ht(t < 0 ? 0. : t * h));
  for (int j = 0; j < r; j++) {
    double tau = dlr.it0B()[j];
    CM gm = -fK([&](double e) { return std::exp(-tau * e); }) * (CM::Identity(NO, NO) + double(xi) * rho);
    DMatrix d = gm.real();
    G.set_mat(j, d);
    DMatrix z = DMatrix::Zero(NO, NO);
    Sig.set_mat(j, z);
  }
  G.initGMConvTensor(dlr);
  for (int n = 0; n <= k; n++) {
    for (int l = 0; l <= k; l++) putmap(Sig.curr_timestep_ret_ptr(n, l), CM::Zero(NO, NO));
    for (int l = 0; l <= k; l++) putmap(Sig.curr_timestep_les_ptr(l, n), CM::Zero(NO, NO));
    for (int j = 0; j < r; j++) Sig.set_tv(n, j, ZMatrix(CM::Zero(NO, NO)));
  }
  if (nobc) putmap(G.curr_timestep_les_ptr(0, 0), exGL(0, 0));
  dyson D(nt, NO, k, dlr, true, false);
  Out o;
  double err = 1;
  for (o.iters = 0; o.iters < 200 && err > 1e-13; o.iters++)
    err = nobc ? D.dyson_start_ntti_nobc(G, mu, H, Sig, I, h) : D.dyson_start_ntti(G, mu, H, Sig, I, h);
  auto GL = [&](int i, int j) { return CM(getmap(G.curr_timestep_les_ptr(i, j))); };
  auto N = [&](const CM &g) { return (II * double(xi) * g).trace(); };
  for (int i = 0; i <= k; i++) {
    CM g = GL(i, i);
    o.ah = std::max(o.ah, (g + g.adjoint()).norm());
    o.dn = std::max(o.dn, std::abs(N(g) - N(GL(0, 0))));
    if (i > 0) {
      CM B = II * g * (Ht(i * h) - mu * CM::Identity(NO, NO));
      CM lhs = CM::Zero(NO, NO);
      for (int j = 0; j <= k; j++) lhs += I.poly_diff(i, j) / h * GL(j, j);
      o.res = std::max(o.res, (lhs - (B - B.adjoint())).norm());
    }
    for (int j = 0; j <= i; j++) {
      o.eL = std::max(o.eL, (GL(j, i) - exGL(j, i)).norm());
      o.eR = std::max(o.eR, (CM(getmap(G.curr_timestep_ret_ptr(i, j))) - exGR(i, j)).norm());
    }
  }
  return o;
}
void check(int xi, bool nobc) {
  for (int k = 2; k <= 5; k++) {
    Out a = run(xi, k, 0.1, nobc), b = run(xi, k, 0.05, nobc);
    for (const Out *o : {&a, &b}) {
      EXPECT_LT(o->iters, 200) << "xi=" << xi << " k=" << k;
      EXPECT_LT(o->res, 1e-12) << "xi=" << xi << " k=" << k << " diagonal-equation residual (B34)";
      EXPECT_LT(o->ah, 1e-12) << "xi=" << xi << " k=" << k << " anti-Hermiticity of G^<(t,t)";
      EXPECT_LT(o->dn, 1e-13) << "xi=" << xi << " k=" << k << " |N(t) - N(0)|";
    }
    EXPECT_GT(hn_test::rate(a.eL, b.eL), k + 1 - 0.5) << "xi=" << xi << " k=" << k << " < " << a.eL << " -> " << b.eL;
    EXPECT_GT(hn_test::rate(a.eR, b.eR), k + 1 - 0.5) << "xi=" << xi << " k=" << k << " R " << a.eR << " -> " << b.eR;
  }
}
}  // namespace cl

TEST(BootstrapRhoDiagonal, ClosedSystemFermionNtti) { cl::check(-1, false); }
TEST(BootstrapRhoDiagonal, ClosedSystemBosonNtti) { cl::check(1, false); }
TEST(BootstrapRhoDiagonal, ClosedSystemFermionNobc) { cl::check(-1, true); }
TEST(BootstrapRhoDiagonal, ClosedSystemBosonNobc) { cl::check(1, true); }

// =============================================================================================== agent_scratch/51/t_start
// Exact system (nao orbitals) + bath, full 4x4 complex Hermitian H; G^R and Sigma^R exact.
namespace ts {
const C II(0, 1);
struct Err { double all = 0, col0 = 0; };
// slot_filled: slot (n,l>n) holds Sigma^R(l,n); imp: imp_tp0
Err ret_ntti_error(int nao, int k, double h, bool slot_filled, bool imp) {
  ZMatrix Hf(4, 4);
  Hf << 0.3, 0.4, 0.5, 0.2, 0.4, -0.6, 0.1, 0.7, 0.5, 0.1, 1.1, 0.3, 0.2, 0.7, 0.3, -0.9;
  Hf(0, 1) += C(0, 0.25);
  Hf(1, 0) -= C(0, 0.25);
  int nb = 4 - nao;
  ZMatrix Hs = Hf.block(0, 0, nao, nao), V = Hf.block(0, nao, nao, nb), Hb = Hf.block(nao, nao, nb, nb);
  auto GRex = [&](double dt) { return ZMatrix((-II * ZMatrix((-II * dt * Hf).exp())).block(0, 0, nao, nao)); };
  auto SRex = [&](double dt) { return ZMatrix(V * (-II * ZMatrix((-II * dt * Hb).exp())) * V.adjoint()); };
  int r = 0;
  dlr_info dlr(r, 20., 1e-10, 1., nao, -1);
  int nt = 2 * k + 4;
  herm_matrix_hodlr G(nt, r, 1, 1e-14, nao, nao, -1, k), S(nt, r, 1, 1e-14, nao, nao, -1, k);
  for (int n = 0; n <= k; n++)
    for (int l = 0; l <= n; l++) S.set_ret_curr(n, l, SRex((n - l) * h));
  if (slot_filled)
    for (int n = 0; n <= k; n++)
      for (int l = n + 1; l <= k; l++) S.set_ret_curr(n, l, SRex((l - n) * h));
  std::vector<C> H((k + 1) * nao * nao);
  for (int l = 0; l <= k; l++) ZMatrixMap(H.data() + l * nao * nao, nao, nao) = Hs;
  dyson D(nt, nao, k, dlr, true, false);
  Integration::Integrator I(k);
  D.dyson_start_ret_ntti(G, 0.0, H.data(), S, I, h, imp);
  ZMatrix M(nao, nao);
  Err e;
  for (int n = 0; n <= k; n++)
    for (int m = 0; m <= n; m++) {
      G.get_ret_curr(n, m, M);
      double d = (M - GRex((n - m) * h)).norm();
      e.all = std::max(e.all, d);
      if (m == 0) e.col0 = std::max(e.col0, d);
    }
  return e;
}
void check_order(int nao, bool slot_filled, bool imp, int kmin, int kmax) {
  for (int k = kmin; k <= kmax; k++) {
    Err a = ret_ntti_error(nao, k, 0.05, slot_filled, imp), b = ret_ntti_error(nao, k, 0.025, slot_filled, imp);
    double ra = imp ? hn_test::rate(a.col0, b.col0) : hn_test::rate(a.all, b.all);
    EXPECT_GT(ra, k + 1 - 0.3) << "nao=" << nao << " k=" << k << (imp ? " (column 0)" : "");
  }
}
}  // namespace ts

TEST(BootstrapRetNtti, OrderSlotFilled) { ts::check_order(2, true, false, 2, 5); }
TEST(BootstrapRetNtti, OrderImpTp0SingleOrbital) { ts::check_order(1, true, true, 2, 5); }
TEST(BootstrapRetNtti, OrderSlotZero) { ts::check_order(2, false, false, 2, 5); }                 // B12
TEST(BootstrapRetNtti, OrderImpTp0TwoOrbitals) { ts::check_order(2, true, true, 3, 5); }          // B13
TEST(BootstrapRetNtti, OrderImpTp0TwoOrbitalsSlotZero) { ts::check_order(2, false, true, 2, 5); }  // B12 + B13

// B31: the imp_tp0 error estimate must compare all es = N_o^2 entries of the old and new G^R(m,0).
// A second call on unchanged input reproduces every column bitwise; column 0 is re-solved (not counted), the
// columns m >= 1 are unchanged, so the returned error is exactly sum_m ||delta_m||_F for a perturbation delta_m
// put into G^R(m,0) before the second call.
TEST(BootstrapRetNtti, ImpTp0ErrorEstimateTwoOrbitals) {
  const int nao = 2, k = 4;
  const double h = 0.05;
  ZMatrix Hf(4, 4);
  Hf << 0.3, 0.4, 0.5, 0.2, 0.4, -0.6, 0.1, 0.7, 0.5, 0.1, 1.1, 0.3, 0.2, 0.7, 0.3, -0.9;
  Hf(0, 1) += C(0, 0.25);
  Hf(1, 0) -= C(0, 0.25);
  ZMatrix Hs = Hf.block(0, 0, nao, nao), V = Hf.block(0, nao, nao, 2), Hb = Hf.block(nao, nao, 2, 2);
  auto SRex = [&](double dt) { return ZMatrix(V * (-ts::II * ZMatrix((-ts::II * dt * Hb).exp())) * V.adjoint()); };
  int r = 0;
  dlr_info dlr(r, 20., 1e-10, 1., nao, -1);
  int nt = 2 * k + 4;
  herm_matrix_hodlr G(nt, r, 1, 1e-14, nao, nao, -1, k), S(nt, r, 1, 1e-14, nao, nao, -1, k);
  for (int n = 0; n <= k; n++)
    for (int l = 0; l <= n; l++) S.set_ret_curr(n, l, SRex((n - l) * h));
  std::vector<C> H((k + 1) * nao * nao);
  for (int l = 0; l <= k; l++) ZMatrixMap(H.data() + l * nao * nao, nao, nao) = Hs;
  dyson D(nt, nao, k, dlr, true, false);
  Integration::Integrator I(k);
  double e1 = D.dyson_start_ret_ntti(G, 0.0, H.data(), S, I, h, true);
  EXPECT_TRUE(std::isfinite(e1));
  EXPECT_EQ(D.dyson_start_ret_ntti(G, 0.0, H.data(), S, I, h, true), 0.0) << "repeat call must reproduce G^R";
  double want = 0;
  for (int m = 1; m <= k; m++) {
    ZMatrix d(nao, nao);
    d << C(1e-3 * m, 2e-4), C(-3e-4, 5e-4 * m), C(7e-4, -1e-4), C(2e-4 * m, 9e-4);  // all es entries nonzero
    ZMatrixMap(G.curr_timestep_ret_ptr(m, 0), nao, nao) += d;
    want += d.norm();
  }
  double e2 = D.dyson_start_ret_ntti(G, 0.0, H.data(), S, I, h, true);
  EXPECT_TRUE(std::isfinite(e2));
  EXPECT_NEAR(e2, want, 1e-12 * want);
}

// =============================================================================================== B25
// dyson_start_2leg(function&, ..., imp_tp0) must forward imp_tp0: it must agree with the pointer overload.
TEST(Bootstrap2leg, FunctionOverloadForwardsImpTp0) {  // B25
  {
    bs::Model M;
    int xi = -1, r = 0, k = 3, nao = bs::NO;
    double h = 0.05;
    dlr_info dlr(r, 60., 1e-13, M.beta, nao, xi);
    bs::Exact E(M, k, h, true);
    int nt = k + 4;
    Integration::Integrator I(k);
    function H(nt, nao, nao);
    for (int t = -1; t < nt; t++) H.get_map(t) = ZMatrix(M.Himp(t < 0 ? 0. : t * h));
    herm_matrix_hodlr Sig(nt, r, 1, 1e-10, nao, nao, xi, k);
    for (int n = 0; n <= k; n++) {
      for (int l = 0; l <= n; l++) bs::putmap(Sig.curr_timestep_ret_ptr(n, l), E.SR(n, l));
      for (int l = n + 1; l <= k; l++) bs::putmap(Sig.curr_timestep_ret_ptr(n, l), E.SR(l, n));
      for (int l = 0; l <= n; l++) bs::putmap(Sig.curr_timestep_les_ptr(l, n), E.SL(l, n));
    }
    herm_matrix_hodlr G1(nt, r, 1, 1e-10, nao, nao, xi, k), G2(nt, r, 1, 1e-10, nao, nao, xi, k);
    bs::putmap(G1.curr_timestep_les_ptr(0, 0), E.GL(0, 0));
    bs::putmap(G2.curr_timestep_les_ptr(0, 0), E.GL(0, 0));
    dyson D1(nt, nao, k, dlr, false, false), D2(nt, nao, k, dlr, false, false);
    D1.dyson_start_2leg(G1, M.mu, H, Sig, I, h, true);         // function& overload
    D2.dyson_start_2leg(G2, M.mu, H.ptr(0), Sig, I, h, true);  // pointer overload
    double d = 0;
    for (int n = 0; n <= k; n++)
      for (int m = 0; m <= n; m++) {
        d = std::max(d, (ZMatrixMap(G1.curr_timestep_ret_ptr(n, m), nao, nao) - ZMatrixMap(G2.curr_timestep_ret_ptr(n, m), nao, nao)).norm());
        d = std::max(d, (ZMatrixMap(G1.curr_timestep_les_ptr(m, n), nao, nao) - ZMatrixMap(G2.curr_timestep_les_ptr(m, n), nao, nao)).norm());
      }
    EXPECT_EQ(d, 0.0) << "function& and pointer overloads differ with imp_tp0=true";
  }
}
