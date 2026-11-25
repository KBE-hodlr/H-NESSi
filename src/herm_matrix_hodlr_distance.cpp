#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"

#ifdef INCLUDE_NESSI
#include <cmath>
#include <stdexcept>
#include "cntr/cntr.hpp"
#endif

namespace h_nessi {


double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk() && tstp >= g2.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()) && tstp <= (g2.tstpmk()+g2.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_les_ptr(0, tstp), (tstp+1)*es, 1) - ZMatrixMap(g2.curr_timestep_les_ptr(0,tstp), (tstp+1)*es, 1)).norm();
}

double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(tstp >= g1.tstpmk() && tstp >= g2.tstpmk());
  assert(tstp <= (g1.tstpmk()+g1.k()) && tstp <= (g2.tstpmk()+g2.k()));

  int size1 = g1.size1(), size2 = g1.size2(), es = size1*size2;

  return (ZMatrixMap(g1.curr_timestep_ret_ptr(tstp,0), (tstp+1)*es, 1) - ZMatrixMap(g2.curr_timestep_ret_ptr(tstp,0), (tstp+1)*es, 1)).norm();
}

double distance_norm2_mat(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(g1.r() == g2.r());

  int size1 = g1.size1(), size2 = g1.size2(), r = g1.r(), es = size1*size2;

  return (DMatrixMap(g1.matptr(0), r*es, 1) - DMatrixMap(g2.matptr(0), r*es, 1)).norm();
}

double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  assert(g1.size1() == g2.size1());
  assert(g1.r() == g2.r());

  int size1 = g1.size1(), size2 = g1.size2(), r = g1.r(), es = size1*size2;

  return (ZMatrixMap(g1.tvptr(tstp,0), r*es, 1) - ZMatrixMap(g2.tvptr(tstp,0), r*es, 1)).norm();
}

double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2) {
  if(tstp == -1) return distance_norm2_mat(g1,g2);
  else {
    double err_R, err_L, err_TV;
    err_R = std::pow(distance_norm2_curr_ret(tstp, g1, g2),2);
    err_L = std::pow(distance_norm2_curr_les(tstp, g1, g2),2);
    err_TV = std::pow(distance_norm2_tv(tstp, g1, g2),2);

    return std::sqrt(err_R + err_L + err_TV);
  }
}


#ifdef INCLUDE_NESSI
// ------------------------------------------------------------------------------------------------------------------
// Distance to a NESSi cntr::herm_matrix<double> (BUGS B8, WP-N). Definitions: herm_matrix_hodlr.hpp; storage and
// convention mapping: agent_docs/05_nessi_herm_matrix.md.
//   R:  H-NESSi curr_timestep_ret_ptr(tstp,0..tstp) and NESSi retptr(tstp,0..tstp) are both one contiguous run of
//       (tstp+1)*es entries, G^R(tstp,t') with t' slow and (i,j) row-major fast.
//   <:  H-NESSi curr_timestep_les_ptr(0..tstp,tstp) and NESSi lesptr(0..tstp,tstp) are both one contiguous run,
//       G^<(t,tstp) with t slow and (i,j) row-major fast.
//   tv/M: NESSi stores tau_m = m*beta/ntau, m = 0..ntau, (i,j) row-major fast; H-NESSi is evaluated there by DLR.
namespace {

void nessi_check_common(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, const char *fn) {
  if (g1.size1() != g2.size1() || g1.size2() != g2.size2() || g1.size1() != g1.size2())
    throw std::invalid_argument(std::string(fn) + ": orbital sizes of herm_matrix_hodlr and cntr::herm_matrix differ "
                                "(or are not square)");
  if (g1.sig() != g2.sig())
    throw std::invalid_argument(std::string(fn) + ": sig (Bose/Fermi) of herm_matrix_hodlr and cntr::herm_matrix differ");
}

void nessi_check_window(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, const char *fn) {
  if (tstp < g1.tstpmk() || tstp > g1.tstp() || tstp < 0)
    throw std::invalid_argument(std::string(fn) + ": tstp is not in the current window [tstpmk, tstpmk+k] of the "
                                "herm_matrix_hodlr");
  if (tstp > g2.nt())
    throw std::invalid_argument(std::string(fn) + ": tstp > nt() of the cntr::herm_matrix");
}

void nessi_check_tau(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr, const char *fn) {
  if (g2.ntau() < 1)
    throw std::invalid_argument(std::string(fn) + ": cntr::herm_matrix needs ntau >= 1");
  if (dlr.r() != g1.r() || dlr.xi() != g1.sig() || dlr.size1() != g1.size1())
    throw std::invalid_argument(std::string(fn) + ": dlr_info (r, xi, size1) does not match the herm_matrix_hodlr");
}

// NESSi's equidistant tau grid tau_m = m*beta/ntau, m = 0..ntau (the last point is exactly beta)
DColVector nessi_tau_grid(int ntau, double beta) {
  DColVector taus(ntau + 1);
  for (int m = 0; m < ntau; m++) taus(m) = m * beta / ntau;
  taus(ntau) = beta;
  return taus;
}

}  // namespace

double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
  nessi_check_common(g1, g2, "distance_norm2_curr_ret");
  nessi_check_window(tstp, g1, g2, "distance_norm2_curr_ret");
  int es = g1.size1() * g1.size2();
  return (ZMatrixMap(g1.curr_timestep_ret_ptr(tstp, 0), (tstp + 1) * es, 1) -
          ZMatrixMap(g2.retptr(tstp, 0), (tstp + 1) * es, 1)).norm();
}

double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2) {
  nessi_check_common(g1, g2, "distance_norm2_curr_les");
  nessi_check_window(tstp, g1, g2, "distance_norm2_curr_les");
  int es = g1.size1() * g1.size2();
  return (ZMatrixMap(g1.curr_timestep_les_ptr(0, tstp), (tstp + 1) * es, 1) -
          ZMatrixMap(g2.lesptr(0, tstp), (tstp + 1) * es, 1)).norm();
}

double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr) {
  nessi_check_common(g1, g2, "distance_norm2_mat");
  nessi_check_tau(g1, g2, dlr, "distance_norm2_mat");
  int ntau = g2.ntau(), es = g1.size1() * g1.size2();
  DColVector taus = nessi_tau_grid(ntau, dlr.beta());
  DMatrix vals(ntau + 1, es);  // row m = g1^M(tau_m), (i,j) row-major
  g1.get_mat_tau_array(taus, dlr, vals);
  return (vals.cast<cplx>() - ZMatrixMap(g2.matptr(0), ntau + 1, es)).norm();
}

double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr) {
  nessi_check_common(g1, g2, "distance_norm2_tv");
  nessi_check_tau(g1, g2, dlr, "distance_norm2_tv");
  if (tstp < 0 || tstp >= g1.nt() || tstp > g2.nt())
    throw std::invalid_argument("distance_norm2_tv: tstp out of range (0 <= tstp < herm_matrix_hodlr::nt(), "
                                "tstp <= cntr::herm_matrix::nt())");
  int ntau = g2.ntau(), es = g1.size1() * g1.size2();
  DColVector taus = nessi_tau_grid(ntau, dlr.beta());
  ZMatrix vals(ntau + 1, es);  // row m = g1^tv(tstp, tau_m), (i,j) row-major
  g1.get_tv_tau_array(tstp, taus, dlr, vals);
  return (vals - ZMatrixMap(g2.tvptr(tstp, 0), ntau + 1, es)).norm();
}

double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr) {
  if (tstp == -1) return distance_norm2_mat(g1, g2, dlr);
  double err_R = std::pow(distance_norm2_curr_ret(tstp, g1, g2), 2);
  double err_L = std::pow(distance_norm2_curr_les(tstp, g1, g2), 2);
  double err_TV = std::pow(distance_norm2_tv(tstp, g1, g2, dlr), 2);
  return std::sqrt(err_R + err_L + err_TV);
}

#endif

} // namespace
