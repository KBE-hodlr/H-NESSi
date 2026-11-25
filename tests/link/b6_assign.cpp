// B6 (fixed in WP4): herm_matrix_hodlr::operator= was declared (documented as a deep copy) but never defined; it is
// now defaulted (memberwise = deep copy).
// This program must link and the assignment must produce an independent deep copy.
#include <cstdio>
#include "h_nessi/herm_matrix_hodlr.hpp"
using namespace h_nessi;
int main() {
  int nt = 20, k = 2, r = 3;
  Integration::Integrator I(k);
  herm_matrix_hodlr A(nt, r, 2, 1e-10, 1, 1, -1, k), B(nt, r, 2, 1e-10, 1, 1, -1, k);
  auto f = [](int T, int tp) { return cplx(T, tp); };
  for (int T = 0; T < 12; T++) {
    if (T > k) A.update_blocks(I);
    for (int tp = 0; tp <= T; tp++) { ZMatrix Y(1, 1); Y(0, 0) = f(T, tp); A.set_ret_curr(T, tp, Y); }
  }
  B = A;
  if (B.tstpmk() != A.tstpmk()) { std::printf("tstpmk differs\n"); return 1; }
  ZMatrix M(1, 1);
  for (int t = 0; t < 12; t++) for (int tp = 0; tp <= t; tp++) {
    B.get_ret(t, tp, M);
    if (std::abs(M(0, 0) - f(t, tp)) > 1e-8) { std::printf("value differs at %d %d\n", t, tp); return 1; }
  }
  ZMatrix Z(1, 1); Z(0, 0) = 0.;
  A.set_ret_curr(11, 0, Z);  // modifying the source must not change the copy
  B.get_ret(11, 0, M);
  if (std::abs(M(0, 0) - f(11, 0)) > 1e-12) { std::printf("not a deep copy\n"); return 1; }
  std::printf("ok\n");
  return 0;
}
