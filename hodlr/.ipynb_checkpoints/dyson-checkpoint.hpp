#ifndef DYSON_DECL
#define DYSON_DECL

#include <vector>
#include <iostream>

#include "herm_matrix_hodlr.hpp"
#include "integration.hpp"
#include "utils.hpp"
#include <chrono>

extern "C"
{
  #include "dlr_c/dlr_c.h"
}


namespace hodlr {

class dyson {
private:
  int ntau_;
  int nao_;
  int es_;
  int k_;
  int nt_;
  
  // libdlr params
  double beta_;
  double lambda_;
  double eps_;
  int r_; 
  int xi_;

  // How to do density matrix
  int rho_version_;

  // libdlr tensors
  double *dlrrf_;
  double *dlrit_;
  double *it0B_;
  double *it2cf_;
  double *it2itr_;
  int *it2cfp_;
  double *phi_;
  double *gmat_;
  double *ipmat_;

  // temporary storage for solving linear equations in timestepping routines
  double *DNTauTmp_;
  double *DNTauTmp2_;
  cplx *NTauTmp_;
  cplx *NTauTmp2_;
  cplx *Q_;
  cplx *M_;
  cplx *X_;
  cplx *epsnao_tmp_;
  cplx *epsnao_tmp_2_;
  cplx *iden_;
  cplx *bound_;

  bool use_dlr_;
  double *w_i_;

  DMatrix timing;
  int timing_count;

public:
  // Construct, destruct
//  dyson();
  ~dyson();
  dyson(int nt, int ntau, int &r, int nao, int k, double beta, double lambda, double eps, int xi, int rho_version=1, bool use_dlr_ = true);
  
  int r() {return r_;}
  int ntau() {return ntau_;}

  // Set up Matsubara convolution tensor
  double *dlrrf() { return dlrrf_; };
  double *dlrit() { return dlrit_; };
  double dlrit(int i) { return dlrit_[i]; };
  double *it0B() { return it0B_; };
  double *dlrit2cf() { return it2cf_; };
  double *dlrit2itr() { return it2itr_; };
  int *dlrit2cfp() { return it2cfp_; };
  double *dlrphi() { return phi_; };
  double *dlripmat() { return ipmat_; };
  int xi(){return xi_;};
  double *w_i(){return w_i_;}

  void write_to_hdf5(h5e::File &file);
  void write_timing(h5e::File &file);

  // Dipole Calc
  void dipole_step(int tstp, cntr::function<double> &dfield, herm_matrix_hodlr &Gu, herm_matrix_hodlr &Gd, double *dipole, double l, double n, double h, Integration::Integrator &I);

  // Free GF
  void green_from_H(herm_matrix_hodlr &G, double mu, cplx *H, double h, int tmax, bool inc_mat = true);
  void green_from_H_dm(herm_matrix_hodlr &G, double mu, cplx *H, cplx *rho, double h, int tmax, bool inc_mat = true);
  void green_from_H_mat(double *g0, double mu, cplx *H);
  void green_from_H_mat(double *g0, double mu, double *H);

  // Extrapolate
  void Extrapolate(herm_matrix_hodlr &G, Integration::Integrator &I);
  void Extrapolate_2leg(herm_matrix_hodlr &G, Integration::Integrator &I);

  // Perform a timestep
  double dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  std::vector<double> dyson_timestep_errinfo(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  // Matsubara timestep
  double dyson_mat(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, bool fixHam=false);
  double dyson_mat(herm_matrix_hodlr &G, double mu, DMatrix &hmf, herm_matrix_hodlr &Sigma, bool fixHam=false);

  // Retarded timestep
  double dyson_timestep_ret(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);


  // Mixed timestep
  double dyson_timestep_tv(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  void tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  void tv_it_conv(cplx *Sigma, cplx *res,double *GMConvTens);
  void tv_it_conv(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  void tv_it_conv(int m, cplx *Sigma, cplx *res,double *GMConvTens);
  void tv_ret_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, Integration::Integrator &I, double h);

  // Les timestep
  double dyson_timestep_les(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_les_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_timestep_les_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  void les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);
  void les_it_int(int t, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);
  void les_lesadv_int_0_tstp(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double h, Integration::Integrator &I);
  void les_it_int_edge(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);

  // Bootstrapping
  double dyson_start(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_ret(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_ret(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_tv(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_tv(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_les(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_les(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  // Bootstrapping_ntti using tv BC in les
  double dyson_start_ntti(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  // Bootstrapping_ntti without using tv BC
  double dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  // Bootstrapping_ntti without mat axis at all
  double dyson_start_2leg(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  double dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, cntr::function<double> &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  double dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);


  // Convolutions
  void matsubara_integral_1(double *C,double *A, double *B);
  void convolution_matsubara_tau_dispatch(double *C, herm_matrix_hodlr &A,double *f0, herm_matrix_hodlr &B);
  void convolution_density_matrix(int n, std::complex<double> *rho, herm_matrix_hodlr &A, herm_matrix_hodlr &Acc, cntr::function<double> &ft,herm_matrix_hodlr &B, herm_matrix_hodlr &Bcc, Integration::Integrator &I, double beta, double h,double *it2cf,int *it2cfp, double *dlrrf);
  void gamma_integral(int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res);
  void gamma_integral_mat(herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res);

};

} // namespace hodlr

#endif // header guard
