// WP6c (ISSUES H27): the orbital-transposed cache tv_trans was removed. Every reader of tv that needed the transpose
// (les_it_int (both overloads), tv_it_conv (both overloads), get_vt (all overloads), Bubble1) now contracts on the tv
// layout directly with Eigen Tensor. This file
//  (1) keeps the OLD matrix-product kernels as a test-only reference (copied from dyson_it_integrals.cpp and
//      herm_matrix_hodlr_getit.cpp at 11aba98), operating on an explicitly built transposed copy of tv, and compares
//      every new reader with it on non-symmetric complex data, n in {1,2,3,4,8,16}, both statistics, several DLR
//      ranks r and T up to 3000 (normwise max|new-ref|/max|ref| <= 1e-14);
//  (2) replaces the former H27 "stale cache" hazard: writes through map_tv / tvptr / set_tstp_zero / Bubble1 are seen
//      by the next get_vt, Bubble1, les_it_int and tv_it_conv call;
//  (3) checks that get_memory_usage counts tv once.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "h_nessi/bubble.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/utils.hpp"

using namespace h_nessi;
using C = std::complex<double>;

namespace {

double hashu(std::uint64_t x) {  // deterministic pseudo-random in [-0.5, 0.5)
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return (double)(x >> 11) * (1.0 / 9007199254740992.0) - 0.5;
}

// fill tv rows [t0, t1) with non-symmetric complex values (no orbital symmetry: X_ij != X_ji, != conj)
void fill_tv(herm_matrix_hodlr &X, int t0, int t1, std::uint64_t seed) {
  const std::size_t row = (std::size_t)X.r() * X.size1() * X.size2();
  for (int t = t0; t < t1; t++) {
    C *p = X.tvptr(t, 0);
    for (std::size_t q = 0; q < row; q++) {
      std::uint64_t idx = 2 * ((std::uint64_t)t * row + q);
      p[q] = C(hashu(seed * 1000003ULL + idx), hashu(seed * 1000003ULL + idx + 1));
    }
  }
}

void fill_mat(herm_matrix_hodlr &X, std::uint64_t seed) {
  const int len = X.r() * X.size1() * X.size2();
  for (int q = 0; q < len; q++) X.matptr(0)[q] = hashu(seed * 7919ULL + q);
}

// explicitly built orbital-transposed copy of tv rows [t0, t1): out[(t-t0)][a][j][i] = tv[t][a][i][j] (the old tv_trans)
std::vector<C> transposed_rows(herm_matrix_hodlr &X, int t0, int t1) {
  const int r = X.r(), n = X.size1(), es = n * n;
  std::vector<C> out((std::size_t)(t1 - t0) * r * es);
  for (int t = t0; t < t1; t++)
    for (int a = 0; a < r; a++)
      ZMatrixMap(out.data() + ((std::size_t)(t - t0) * r + a) * es, n, n) = ZMatrixMap(X.tvptr(t, a), n, n).transpose();
  return out;
}

// ---------------------------------------------------------------- test-only reference: the OLD kernels (11aba98)
// les_it_int(tstp, Sigma, G, res): res_t += X(rn x n)^T * Sigma.tv_trans[t](rn x n), t = 0..tstp
void ref_les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, dlr_info &dlr, C *res) {
  const int r = G.r(), nao = G.size1(), es = nao * nao;
  std::vector<C> St = transposed_rows(Sigma, 0, tstp + 1), Gt = transposed_rows(G, tstp, tstp + 1);
  std::vector<C> NTauTmp(r * es), NTauTmp2(r * es);
  C mimxi = C(0, G.sig());
  ZMatrixMap(NTauTmp.data(), r, es).noalias() = mimxi * DMatrixConstMap(dlr.it2itr(), r, r).transpose() * ZMatrixMap(Gt.data(), r, es).conjugate();
  ZMatrixMap(NTauTmp2.data(), r, es).noalias() = DMatrixConstMap(dlr.ipmat(), r, r).transpose() * ZMatrixMap(NTauTmp.data(), r, es);
  for (int t = 0; t <= tstp; t++)
    ZMatrixMap(res + t * es, nao, nao).noalias() += ZMatrixMap(NTauTmp2.data(), r * nao, nao).transpose() * ZMatrixMap(St.data() + (std::size_t)t * r * es, r * nao, nao);
}

// les_it_int(t, tstp, Sigma, G, res): one row
void ref_les_it_int_row(int t, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, dlr_info &dlr, C *res) {
  const int r = G.r(), nao = G.size1(), es = nao * nao;
  std::vector<C> St = transposed_rows(Sigma, t, t + 1), Gt = transposed_rows(G, tstp, tstp + 1);
  std::vector<C> NTauTmp(r * es), NTauTmp2(r * es);
  C mimxi = C(0, G.sig());
  ZMatrixMap(NTauTmp.data(), r, es) = mimxi * DMatrixConstMap(dlr.it2itr(), r, r).transpose() * ZMatrixMap(Gt.data(), r, es).conjugate();
  ZMatrixMap(NTauTmp2.data(), r, es) = DMatrixConstMap(dlr.ipmat(), r, r).transpose() * ZMatrixMap(NTauTmp.data(), r, es);
  ZMatrixMap(res, nao, nao) += ZMatrixMap(NTauTmp2.data(), r * nao, nao).transpose() * ZMatrixMap(St.data(), r * nao, nao);
}

// tv_it_conv(tstp, Sigma, G, res): I_{aji} = C_{ajbk} Sigma.tv_trans_{bki}; res_a += I_a^T
void ref_tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, C *res) {
  const int r = G.r(), nao = G.size1(), es = nao * nao;
  std::vector<C> St = transposed_rows(Sigma, tstp, tstp + 1), tmp(r * es);
  ZMatrixMap(tmp.data(), r * nao, nao).noalias() = DMatrixMap(G.GMConvTens(), r * nao, r * nao) * ZMatrixMap(St.data(), r * nao, nao);
  for (int i = 0; i < r; i++) ZMatrixMap(res + i * es, nao, nao) += ZMatrixMap(tmp.data() + i * es, nao, nao).transpose();
}

// tv_it_conv(m, tstp, Sigma, G, res)
void ref_tv_it_conv_m(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, C *res) {
  const int r = G.r(), nao = G.size1();
  std::vector<C> St = transposed_rows(Sigma, tstp, tstp + 1);
  ZMatrixMap(res, nao, nao).noalias() += (DMatrixMap(G.GMConvTens() + m * nao * nao * r, nao, r * nao) * ZMatrixMap(St.data(), r * nao, nao)).transpose();
}

// get_vt(tstp, dlr, M): M = -xi (it2itr^T * tv_trans row)^*
void ref_get_vt(int tstp, herm_matrix_hodlr &G, dlr_info &dlr, C *M) {
  const int r = G.r(), es = G.size1() * G.size2();
  std::vector<C> Gt = transposed_rows(G, tstp, tstp + 1);
  ZMatrixMap(M, r, es) = -G.sig() * (DMatrixConstMap(dlr.it2itr(), r, r).transpose() * ZMatrixMap(Gt.data(), r, es)).conjugate();
}

// get_vt(tstp, dlr, M, i, j): strided column (i,j) of the tv_trans row
void ref_get_vt_ij(int tstp, herm_matrix_hodlr &G, dlr_info &dlr, ZMatrix &M, int i, int j) {
  const int r = G.r(), es = G.size1() * G.size2();
  std::vector<C> Gt = transposed_rows(G, tstp, tstp + 1);
  Eigen::Map<const ZMatrix, 0, Eigen::InnerStride<>> col(Gt.data() + i * G.size2() + j, r, 1, Eigen::InnerStride<>(es));
  ZMatrixMap(M.data(), r, 1) = -G.sig() * (DMatrixConstMap(dlr.it2itr(), r, r).transpose() * col).conjugate();
}

double maxabs(const C *a, std::size_t n) {
  double m = 0;
  for (std::size_t q = 0; q < n; q++) m = std::max(m, std::abs(a[q]));
  return m;
}
double g_maxrel = 0;  // largest reldiff seen in the current test (printed at its end)
// normwise relative difference max|a-b| / max|b|
double reldiff(const C *a, const C *b, std::size_t n) {
  double d = 0;
  for (std::size_t q = 0; q < n; q++) d = std::max(d, std::abs(a[q] - b[q]));
  double s = maxabs(b, n);
  double rel = s == 0 ? d : d / s;
  if (rel < 1e-3) g_maxrel = std::max(g_maxrel, rel);   // (the deliberate before/after difference is not recorded)
  return rel;
}
struct PrintMax {
  const char *what;
  explicit PrintMax(const char *w) : what(w) { g_maxrel = 0; }
  ~PrintMax() { std::printf("[ max rel ] %s: %.3e\n", what, g_maxrel); }
};
double reldiff(const std::vector<C> &a, const std::vector<C> &b) { return reldiff(a.data(), b.data(), b.size()); }

int nlvl_for(int nt) {  // HODLR levels with segments of ~8 rows (keeps the direct stores small for large nt)
  int l = 1;
  while ((2 << l) * 8 <= nt) l++;
  return l;
}

const double TOL = 1e-14;

// One (n, xi, Lambda) case: Sigma, G with tv rows 0..Tmax, random G^M (conv tensor built from it).
struct Case {
  int nao, xi, r = 0, nt, k = 3;
  dlr_info dlr;
  herm_matrix_hodlr S, G;
  dyson D;
  Case(int nao_, int xi_, double lambda, double eps, int Tmax)
      : nao(nao_), xi(xi_), nt(Tmax + 1), dlr(r, lambda, eps, 5., nao_, xi_),
        S(nt, r, nlvl_for(nt), 1e-10, nao, nao, xi, k), G(nt, r, nlvl_for(nt), 1e-10, nao, nao, xi, k),
        D(nt, nao, k, dlr, true, false) {
    fill_tv(S, 0, nt, 11 + nao + 100 * (xi + 1));
    fill_tv(G, 0, nt, 23 + nao + 100 * (xi + 1));
    fill_mat(G, 5 + nao);
    G.initGMConvTensor(dlr);
  }
};

struct Grid {
  int nao, xi;
  double lambda, eps;
  std::vector<int> Ts;  // les_it_int time steps; the last one is Tmax
};

std::vector<Grid> grid() {
  std::vector<Grid> g;
  for (int xi : {-1, +1}) {
    g.push_back({1, xi, 20., 1e-10, {0, 1, 7, 100, 3000}});
    g.push_back({1, xi, 1000., 1e-13, {0, 5, 3000}});
    g.push_back({2, xi, 20., 1e-10, {0, 3, 3000}});
    g.push_back({2, xi, 500., 1e-12, {0, 1, 100, 2500}});
    g.push_back({3, xi, 200., 1e-12, {0, 2, 1500}});
    g.push_back({4, xi, 20., 1e-10, {0, 1, 2000}});
    g.push_back({4, xi, 1000., 1e-13, {0, 7, 600}});
    g.push_back({8, xi, 100., 1e-12, {0, 3, 300}});
    g.push_back({16, xi, 20., 1e-10, {0, 2, 80}});
    g.push_back({16, xi, 200., 1e-12, {1, 40}});
  }
  return g;
}

std::string name(const Grid &c, int r) {
  return "n=" + std::to_string(c.nao) + " xi=" + std::to_string(c.xi) + " r=" + std::to_string(r);
}

}  // namespace

// ------------------------------------------------------------------------------------------------ les_it_int
TEST(TvContraction, LesItIntMatchesOldKernel) {
  PrintMax pm("LesItIntMatchesOldKernel");
  for (const Grid &c : grid()) {
    Case P(c.nao, c.xi, c.lambda, c.eps, c.Ts.back());
    const int es = c.nao * c.nao;
    for (int T : c.Ts) {
      // batched overload, incremented into a non-zero res (res VIA INCREMENT)
      std::vector<C> res((std::size_t)(T + 1) * es), ref;
      for (std::size_t q = 0; q < res.size(); q++) res[q] = C(0.25 * hashu(q), -0.5 * hashu(q + 7));
      ref = res;
      P.D.les_it_int(T, P.S, P.G, res.data());
      ref_les_it_int(T, P.S, P.G, P.dlr, ref.data());
      EXPECT_LE(reldiff(res, ref), TOL) << name(c, P.r) << " T=" << T;
      // single-row overload, a few rows (with the arguments swapped as the bootstrap does as well)
      for (int t : {0, T / 2, T}) {
        std::vector<C> a(es, C(0.3, -0.1)), b = a;
        P.D.les_it_int(t, T, P.S, P.G, a.data());
        ref_les_it_int_row(t, T, P.S, P.G, P.dlr, b.data());
        EXPECT_LE(reldiff(a, b), TOL) << name(c, P.r) << " row t=" << t << " T=" << T;
        std::fill(a.begin(), a.end(), C(0));
        std::fill(b.begin(), b.end(), C(0));
        P.D.les_it_int(t, T, P.G, P.S, a.data());
        ref_les_it_int_row(t, T, P.G, P.S, P.dlr, b.data());
        EXPECT_LE(reldiff(a, b), TOL) << name(c, P.r) << " swapped row t=" << t << " T=" << T;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ tv_it_conv
TEST(TvContraction, TvItConvMatchesOldKernel) {
  PrintMax pm("TvItConvMatchesOldKernel");
  for (const Grid &c : grid()) {
    Case P(c.nao, c.xi, c.lambda, c.eps, c.Ts.back());
    const int es = c.nao * c.nao, r = P.r;
    for (int T : {0, c.Ts.back() / 2, c.Ts.back()}) {
      std::vector<C> res((std::size_t)r * es), ref;
      for (std::size_t q = 0; q < res.size(); q++) res[q] = C(hashu(3 * q), hashu(3 * q + 1));
      ref = res;
      P.D.tv_it_conv(T, P.S, P.G, res.data());
      ref_tv_it_conv(T, P.S, P.G, ref.data());
      EXPECT_LE(reldiff(res, ref), TOL) << name(c, r) << " T=" << T;
      for (int m : {0, r / 2, r - 1}) {
        std::vector<C> a(es, C(-0.2, 0.4)), b = a;
        P.D.tv_it_conv(m, T, P.S, P.G, a.data());
        ref_tv_it_conv_m(m, T, P.S, P.G, b.data());
        EXPECT_LE(reldiff(a, b), TOL) << name(c, r) << " m=" << m << " T=" << T;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ get_vt
TEST(TvContraction, GetVtMatchesOldKernel) {
  PrintMax pm("GetVtMatchesOldKernel");
  for (const Grid &c : grid()) {
    Case P(c.nao, c.xi, c.lambda, c.eps, c.Ts.back());
    const int es = c.nao * c.nao, r = P.r;
    // get_vt reads computed rows only (row <= tstpmk+k): move the window to reach the last rows as well
    for (int T : {0, P.k, c.Ts.back()}) {
      if (T > P.k) P.G.set_tstpmk(T - P.k);
      std::vector<C> a((std::size_t)r * es), b(a.size());
      P.G.get_vt(T, P.dlr, a.data());
      ref_get_vt(T, P.G, P.dlr, b.data());
      EXPECT_LE(reldiff(a, b), TOL) << name(c, r) << " T=" << T;
      ZMatrix Z(r * es, 1);
      P.G.get_vt(T, P.dlr, Z);
      EXPECT_LE(reldiff(Z.data(), b.data(), b.size()), TOL) << name(c, r) << " ZMatrix overload T=" << T;
      for (int i = 0; i < c.nao; i++)
        for (int j = 0; j < c.nao; j++) {
          if (c.nao > 4 && (i + 3 * j) % 5 != 0) continue;
          ZMatrix e(r, 1), f(r, 1);
          P.G.get_vt(T, P.dlr, e, i, j);
          ref_get_vt_ij(T, P.G, P.dlr, f, i, j);
          EXPECT_LE(reldiff(e.data(), f.data(), r), TOL) << name(c, r) << " (i,j)=(" << i << "," << j << ") T=" << T;
          // the element overload is the (i,j) slice of the full one
          for (int n = 0; n < r; n++) EXPECT_LE(std::abs(e(n, 0) - a[n * es + i * c.nao + j]), TOL * maxabs(a.data(), a.size()));
        }
    }
  }
}

// ------------------------------------------------------------------------------------------------ Bubble1
// C^tv_{c1c2}(t,tau) = i A^tv_{a1a2}(t,tau) B^vt_{b2b1}(tau,t), with B^vt from the old kernel (transposed copy)
TEST(TvContraction, Bubble1MatchesOldKernel) {
  for (int nao : {1, 2, 3, 4, 8, 16})
    for (int xi : {-1, +1}) {
      int r = 0, nt = 16, k = 3;
      dlr_info dlr(r, nao > 4 ? 50. : 200., 1e-12, 5., nao, xi);
      herm_matrix_hodlr A(nt, r, 1, 1e-10, nao, nao, xi, k), B(nt, r, 1, 1e-10, nao, nao, xi, k),
          Cc(nt, r, 1, 1e-10, nao, nao, -xi * xi, k);
      fill_tv(A, 0, nt, 3 + nao);
      fill_tv(B, 0, nt, 4 + nao);
      for (int tstp : {0, k}) {
        for (int c1 = 0; c1 < nao; c1++)
          for (int c2 = 0; c2 < nao; c2++) {
            if (nao > 3 && (c1 + c2) % 3 != 0) continue;
            int a1 = (c1 + 1) % nao, a2 = c2, b1 = c2, b2 = (c1 + 2 * c2 + 1) % nao;
            Bubble1(tstp, Cc, c1, c2, A, a1, a2, B, b1, b2, dlr);
            ZMatrix Bvt(r, 1);
            ref_get_vt_ij(tstp, B, dlr, Bvt, b2, b1);
            double d = 0, s = 0;
            for (int tau = 0; tau < r; tau++) {
              C ref = C(0, 1) * A.map_tv(tstp, tau)(a1, a2) * Bvt(tau, 0);
              d = std::max(d, std::abs(Cc.map_tv(tstp, tau)(c1, c2) - ref));
              s = std::max(s, std::abs(ref));
            }
            EXPECT_LE(d, TOL * s) << "n=" << nao << " xi=" << xi << " tstp=" << tstp << " c=(" << c1 << "," << c2 << ")";
          }
      }
    }
}

// ------------------------------------------------------------------------------------------------ H27 hazard gone
// Formerly (ISSUES H27) writes through map_tv / tvptr / set_tstp_zero / Bubble1 outputs / an MPI get left tv_trans
// stale, and get_vt, Bubble1(..,B,..), les_it_int and tv_it_conv read the stale copy. There is no copy any more:
// every write is seen by the next read.
TEST(TvContraction, MapTvWritesAreSeenByAllReaders) {
  PrintMax pm("MapTvWritesAreSeenByAllReaders");
  for (int nao : {1, 2, 3}) {
    Case P(nao, -1, 100., 1e-12, 20);
    const int es = nao * nao, r = P.r, T = P.k;
    auto vt = [&](herm_matrix_hodlr &X, int t) {
      std::vector<C> v((std::size_t)r * es);
      X.get_vt(t, P.dlr, v.data());
      return v;
    };
    auto vt_ref = [&](herm_matrix_hodlr &X, int t) {
      std::vector<C> v((std::size_t)r * es);
      ref_get_vt(t, X, P.dlr, v.data());
      return v;
    };
    std::vector<C> before = vt(P.G, T);
    // (a) map_tv write, (b) raw tvptr write
    for (int a = 0; a < r; a++) P.G.map_tv(T, a) = ZMatrix::Constant(nao, nao, C(0.5, -0.25 * a)) + ZMatrix::Identity(nao, nao) * C(0, 1);
    P.G.tvptr(T, 0)[es - 1] = C(7., 3.);
    std::vector<C> after = vt(P.G, T);
    EXPECT_GT(reldiff(after, before), 1e-3) << "get_vt did not see the map_tv write";
    EXPECT_LE(reldiff(after, vt_ref(P.G, T)), TOL);
    // les_it_int sees new Sigma rows (map_tv) and the new G row T
    for (int t = 0; t <= T; t++)
      for (int a = 0; a < r; a++) P.S.map_tv(t, a)(0, nao - 1) += C(1., -2.);
    std::vector<C> L((std::size_t)(T + 1) * es, C(0)), Lr = L;
    P.D.les_it_int(T, P.S, P.G, L.data());
    ref_les_it_int(T, P.S, P.G, P.dlr, Lr.data());
    EXPECT_LE(reldiff(L, Lr), TOL);
    // tv_it_conv sees the new Sigma row
    std::vector<C> V((std::size_t)r * es, C(0)), Vr = V;
    P.D.tv_it_conv(T, P.S, P.G, V.data());
    ref_tv_it_conv(T, P.S, P.G, Vr.data());
    EXPECT_LE(reldiff(V, Vr), TOL);
    // (c) set_tstp_zero: get_vt of the zeroed row is zero
    P.G.set_tstpmk(0);
    P.G.set_tstp_zero(T);
    EXPECT_EQ(maxabs(vt(P.G, T).data(), (std::size_t)r * es), 0.0);
    // (d) Bubble1 output: C's tv row written by Bubble1 is seen by C.get_vt and by a following Bubble1 with B = C
    herm_matrix_hodlr Cc(P.nt, r, nlvl_for(P.nt), 1e-10, nao, nao, -1, P.k), E2(P.nt, r, nlvl_for(P.nt), 1e-10, nao, nao, -1, P.k);
    for (int c1 = 0; c1 < nao; c1++)
      for (int c2 = 0; c2 < nao; c2++) Bubble1(1, Cc, c1, c2, P.S, c1, c2, P.G, c2, c1, P.dlr);
    EXPECT_GT(maxabs(Cc.tvptr(1, 0), (std::size_t)r * es), 0.0);
    EXPECT_LE(reldiff(vt(Cc, 1), vt_ref(Cc, 1)), TOL);
    Bubble1(1, E2, 0, 0, P.S, 0, 0, Cc, 0, 0, P.dlr);
    ZMatrix Bvt(r, 1);
    ref_get_vt_ij(1, Cc, P.dlr, Bvt, 0, 0);
    double d = 0;
    for (int tau = 0; tau < r; tau++)
      d = std::max(d, std::abs(E2.map_tv(1, tau)(0, 0) - C(0, 1) * P.S.map_tv(1, tau)(0, 0) * Bvt(tau, 0)));
    EXPECT_LE(d, 1e-13);
  }
}

// ------------------------------------------------------------------------------------------------ memory rule
// get_memory_usage counts tv once (nt*r*es complex); before WP6c it also counted tv_trans (another nt*r*es).
// Two objects that differ only in r differ by the tv, mat and (empty) conv-tensor terms.
TEST(TvContraction, MemoryReportCountsTvOnce) {
  int nt = 64, nlvl = 2, k = 3, nao = 3, r1 = 5, r2 = 9;
  herm_matrix_hodlr G1(nt, r1, nlvl, 1e-10, nao, nao, -1, k), G2(nt, r2, nlvl, 1e-10, nao, nao, -1, k);
  const double D = (double)sizeof(double) / 1e9, Z = 2 * D;
  double d = G2.get_memory_usage(false) - G1.get_memory_usage(false);
  double gmct = D * (G2.GMConvTensMatrix().size() - G1.GMConvTensMatrix().size());
  double expected = Z * nt * (r2 - r1) * nao * nao + D * (r2 - r1) * nao * nao + gmct;
  EXPECT_NEAR(d, expected, 1e-9 * expected);
  EXPECT_EQ(G1.tv().data_vector().size(), (std::size_t)nt * r1 * nao * nao);
}
