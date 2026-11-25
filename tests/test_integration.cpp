// Integration weights (doc 02).  Ported from agent_scratch/02/t.cpp.
// Exactness on polynomials (interpolation, differentiation, integration, start, backward differentiation
// with k+2 terms, extrapolation, Gregory) and convergence orders on f(t) = exp(sin t).
#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>

#include "h_nessi/integration.hpp"
#include "common/test_util.hpp"

typedef long double LD;
static LD pw(LD x, int d) { return d == 0 ? 1 : powl(x, d); }
static LD dpw(LD x, int d) { return d == 0 ? 0 : d * pw(x, d - 1); }
static LD ipw(LD x, int d) { return pw(x, d + 1) / (d + 1); }

class IntegrationK : public ::testing::TestWithParam<int> {};

TEST_P(IntegrationK, PolynomialExactness) {
  const int k = GetParam();
  Integration::Integrator I(k);
  LD eInterp = 0, eD = 0, eInt = 0, eStart = 0;
  for (int a = 0; a <= k; a++)
    for (int d = 0; d <= k; d++) {
      LD s = 0;
      for (int l = 0; l <= k; l++) s += I.poly_interp(a, l) * pw(l, d);
      eInterp = std::max(eInterp, fabsl(s - (a == d)));
    }
  for (int d = 0; d <= k; d++) {
    for (int m = 0; m <= k; m++) {
      LD s = 0;
      for (int l = 0; l <= k; l++) s += I.poly_diff(m, l) * pw(l, d);
      eD = std::max(eD, fabsl(s - dpw(m, d)));
    }
    for (int m = 0; m <= k; m++)
      for (int n = 0; n <= k; n++) {
        LD s = 0;
        for (int l = 0; l <= k; l++) s += I.poly_integ(m, n, l) * pw(l, d);
        eInt = std::max(eInt, fabsl(s - (ipw(n, d) - ipw(m, d))));
      }
    for (int n = 0; n <= k; n++) {
      LD s = 0;
      for (int l = 0; l <= k; l++) s += I.start(n, l) * pw(l, d);
      eStart = std::max(eStart, fabsl(s - ipw(n, d)));
    }
  }
  EXPECT_LT((double)eInterp, 1e-10) << "poly_interp";
  EXPECT_LT((double)eD, 1e-10) << "poly_diff";
  EXPECT_LT((double)eInt, 1e-10) << "poly_integ";
  EXPECT_LT((double)eStart, 1e-10) << "start";
}

// bd_weights has k+2 terms and is exact up to degree k+1 (paper A.7 erratum P1); degree k+2 is not exact.
TEST_P(IntegrationK, BackwardDifferentiationKPlus2Terms) {
  const int k = GetParam();
  Integration::Integrator I(k);
  LD h = 0.5L, eBD = 0;
  for (int d = 0; d <= k + 1; d++)
    for (int n = k + 1; n <= k + 4; n++) {
      LD s = 0;
      for (int l = 0; l <= k + 1; l++) s += I.bd_weights(l) * pw((n - l) * h, d);
      s /= h;
      eBD = std::max(eBD, fabsl(s - dpw(n * h, d)) / std::max((LD)1, fabsl(dpw(n * h, d))));
    }
  EXPECT_LT((double)eBD, 1e-12);
  int d = k + 2, n = k + 3;
  LD s = 0;
  for (int l = 0; l <= k + 1; l++) s += I.bd_weights(l) * pw((n - l) * h, d);
  s /= h;
  EXPECT_GT((double)fabsl(s - dpw(n * h, d)), 0.1) << "degree k+2 unexpectedly exact";
}

TEST_P(IntegrationK, ExtrapolationExact) {
  const int k = GetParam();
  Integration::Integrator I(k);
  LD eEx = 0;
  for (int d = 0; d <= k; d++)
    for (int n = k; n <= k + 3; n++) {
      LD s = 0;
      for (int l = 0; l <= k; l++) s += I.ex_weights(l) * pw(n - l, d);
      eEx = std::max(eEx, fabsl(s - pw(n + 1, d)) / pw(n + 1, d));
    }
  EXPECT_LT((double)eEx, 1e-12);
}

TEST_P(IntegrationK, GregoryExactAndSymmetric) {
  const int k = GetParam();
  Integration::Integrator I(k);
  int ns[] = {0, 1, 2, k, k + 1, k + 2, 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3, 3 * k + 4, 20, 57, 200};
  LD eGk = 0;
  for (int d = 0; d <= k; d++)
    for (int n : ns) {
      LD s = 0;
      for (int j = 0; j <= std::max(n, k); j++) s += I.gregory_weights(n, j) * pw(j, d);
      eGk = std::max(eGk, fabsl(s - ipw(n, d)) / std::max((LD)1, ipw(n, d)));
    }
  EXPECT_LT((double)eGk, 1e-11);
  LD eSym = 0;
  for (int n = k + 1; n <= 3 * k + 5; n++)
    for (int j = 0; j <= n; j++) eSym = std::max(eSym, fabsl(I.gregory_weights(n, j) - I.gregory_weights(n, n - j)));
  EXPECT_LT((double)eSym, 1e-14);
}

INSTANTIATE_TEST_SUITE_P(Integration, IntegrationK, ::testing::Values(0, 1, 2, 3, 4, 5, 6));

// Convergence on a smooth function: bd O(h^{k+1}), Gregory >= O(h^{k+1}), ex O(h^{k+1}), start >= O(h^{k+1}),
// poly_diff at the last point O(h^k).
class IntegrationOrder : public ::testing::TestWithParam<int> {};

TEST_P(IntegrationOrder, ConvergenceRates) {
  const int k = GetParam();
  Integration::Integrator I(k);
  auto f = [](LD t) { return expl(sinl(t)); };
  auto fp = [](LD t) { return cosl(t) * expl(sinl(t)); };
  LD T = 2.0L, ref = 0;
  {
    long N = 2000000;
    LD hh = T / N;
    ref = f(0) + f(T);
    for (long i = 1; i < N; i++) ref += f(i * hh) * ((i % 2) ? 4 : 2);
    ref *= hh / 3;
  }
  struct E { double bd, greg, ex, start, diff; };
  std::vector<E> errs;
  for (int N : {20, 40, 80, 160}) {
    LD h = T / N;
    LD s = 0;
    for (int l = 0; l <= k + 1; l++) s += I.bd_weights(l) * f(T - l * h);
    s /= h;
    LD g = 0;
    for (int j = 0; j <= N; j++) g += I.gregory_weights(N, j) * f(j * h);
    g *= h;
    LD x = 0;
    for (int l = 0; l <= k; l++) x += I.ex_weights(l) * f(T - l * h);
    LD st = 0;
    for (int l = 0; l <= k; l++) st += I.start(k, l) * f(l * h);
    st *= h;
    LD exs = 0;
    {
      long M = 20000;
      LD hh = k * h / M;
      exs = f(0) + f(k * h);
      for (long i = 1; i < M; i++) exs += f(i * hh) * ((i % 2) ? 4 : 2);
      exs *= hh / 3;
    }
    LD dd = 0;
    for (int l = 0; l <= k; l++) dd += I.poly_diff(k, l) * f(l * h);
    dd /= h;
    errs.push_back({(double)fabsl(s - fp(T)), (double)fabsl(g - ref), (double)fabsl(x - f(T + h)), (double)fabsl(st - exs),
                    (double)fabsl(dd - fp(k * h))});
  }
  const double slack = 0.3;
  for (int p = 1; p < 3; p++) {  // pairs (40,80), (80,160): asymptotic regime
    EXPECT_GT(hn_test::rate(errs[p].bd, errs[p + 1].bd), k + 1 - slack) << "bd_weights pair " << p;
    EXPECT_GT(hn_test::rate(errs[p].greg, errs[p + 1].greg), k + 1 - slack) << "gregory pair " << p;
    EXPECT_GT(hn_test::rate(errs[p].ex, errs[p + 1].ex), k + 1 - slack) << "ex_weights pair " << p;
  }
  EXPECT_GT(hn_test::rate(errs[0].start, errs[1].start), k + 1 - slack) << "start";
  EXPECT_GT(hn_test::rate(errs[0].diff, errs[1].diff), k - slack) << "poly_diff(k,.)";
}

INSTANTIATE_TEST_SUITE_P(Integration, IntegrationOrder, ::testing::Values(1, 2, 3, 4, 5));
