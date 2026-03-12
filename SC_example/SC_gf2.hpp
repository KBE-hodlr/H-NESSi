#ifndef SC_GF2
#define SC_GF2

#include "hodlr/utils.hpp"
#include "hodlr/herm_matrix_hodlr.hpp"
#include "hodlr/function.hpp"
#include "hodlr/dlr.hpp"

namespace hodlr {

class SC_gf2 {
private:
  dlr_info &dlr_;
  ZMatrix sigma3_;;

  void Delta_mat(function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void Delta_tstp(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);

  void Sigma_mat(function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void Sigma_tstp(int tstp, function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  
public:
  SC_gf2(dlr_info &dlr);

  void solve_Delta(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void solve_Delta_LmR(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void solve_Sigma(int tstp, function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void solve_Sigma_Fock(int tstp, function &U, herm_matrix_hodlr &G, function &H);
};

} // namespace

#endif
