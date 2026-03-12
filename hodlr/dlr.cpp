#ifndef DLR_IMPL
#define DLR_IMPL

#include "dlr.hpp"

namespace hodlr {

void rel2abs_times_beta(int n, double beta, double *trel, double *tabs) {
  for( int i = 0; i < n; i++) {
    tabs[i] = trel[i] < 0. ? beta * (trel[i]+1) : beta * trel[i];
  }
}

dlr_info::dlr_info(int &r, double lambda, double eps, double beta, int nao, int xi, int ntaumax) {
  eps_ = eps;
  lambda_ = lambda;
  beta_ = beta;
  size1_ = nao;
  size2_ = nao;
  xi_ = xi;
  int input_r = ntaumax;
  dlrrf_.resize(input_r); // DLR real frequencies
  dlrit_.resize(input_r); // DLR imaginary time nodes
  c_dlr_it_build(&lambda_, &eps_, &input_r, dlrrf_.data(), dlrit_.data()); // Get imaginary time nodes and frequencies

  // input_r has been overwritten with the number of dlr nodes
  r = input_r;
  r_ = input_r;

  it0B_.resize(r);
  cf2it_.resize(r*r);  // coeficients [cf] to imaginary time [it]
  it2cf_.resize(r*r);  // Imaginary time [it] to coeficients [cf]
  it2cfp_.resize(r);         // Imaginary time to coeficients [pivots]
  it2itr_.resize(r*r); // Imaginary time to reversed imaginary time
  phi_.resize(r*r*r); // Convolution tensor
  ipmat_.resize(r*r);    // Inner product weights 

  // these functions call fortran code which is not threadsafe
  #pragma omp critical 
  { 
    c_dlr_it2cf_init(&r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data()); // Get it to cf
    c_dlr_convtens(&beta_, &xi_, &r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data(), phi_.data()); // Get convolution tensor
    c_dlr_ipmat(&beta_, &r, dlrit_.data(), dlrrf_.data(), it2cf_.data(), it2cfp_.data(), ipmat_.data());
    c_dlr_cf2it_init(&r,dlrrf_.data(),dlrit_.data(),cf2it_.data());
    c_dlr_it2itr_init(&r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data(), it2itr_.data()); // Imaginary time to reversed imaginary time
    rel2abs_times_beta(r, beta_, dlrit_.data(), it0B_.data()); // Convert from relative format in dlr to absolute values from 0 to beta
  

    Gijc_.resize(r*size1_*size2_);
    res_.resize(size1_*size2_);
    LU_.compute(DMatrixMap(cf2it_.data(), r, r).transpose());
  }
}



void dlr_info::eval_point(double tau, double *Gtij, double *Mij) {
  int es = size1_*size2_;
  int one = 1;
  
  double tau01 = tau/beta_;
  double taurel;
  c_abs2rel(&one, &tau01, &taurel);

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), &taurel, res_.data());
  DMatrixMap(Mij, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
}

void dlr_info::eval_point(double tau, double *Gtij, cplx *Mij) {
  int es = size1_*size2_;
  int one = 1;
  
  double tau01 = tau/beta_;
  double taurel;
  c_abs2rel(&one, &tau01, &taurel);

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), &taurel, res_.data());
  ZMatrixMap(Mij, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
}

void dlr_info::eval_point(double tau, cplx *Gtij, cplx *Mij) {
  int es = size1_*size2_;
  int one = 1;
  
  double tau01 = tau/beta_;
  double taurel;
  c_abs2rel(&one, &tau01, &taurel);

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).real()).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), &taurel, res_.data());
  ZMatrixMap(Mij, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).imag()).transpose();
  c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), &taurel, res_.data());
  ZMatrixMap(Mij, size1_, size2_) += cplx(0.,1.) * DMatrixMap(res_.data(), size1_, size2_);  
}

void dlr_info::eval_point(DColVector &tau, double *Gtij, double *Mtij) {
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  DColVector taurel;
  int ntau = tau.size();
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data(), res_.data());
    DMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::eval_point(DColVector &tau, double *Gtij, cplx *Mtij) {
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  DColVector taurel;
  int ntau = tau.size();
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data(), res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::eval_point(DColVector &tau, cplx *Gtij, cplx *Mtij) {
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  DColVector taurel;
  int ntau = tau.size();
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).real()).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data(), res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).imag()).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data(), res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) += cplx(0.,1.) * DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::write_to_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, label + std::string("/r"), r_);
  h5e::dump(out, label + std::string("/beta"), beta_);
  h5e::dump(out, label + std::string("/eps"), eps_);
  h5e::dump(out, label + std::string("/lambda"), lambda_);
  h5e::dump(out, label + std::string("/xi"), xi_);
  h5e::dump(out, label + std::string("/it0B"), DMatrixMap(it0B_.data(), r_, 1));
}
}
#endif
