#ifndef DLR_IMPL
#define DLR_IMPL

#include "h_nessi/dlr.hpp"
#include "h_nessi/checks.hpp"

#include <algorithm>
#include <cmath>

namespace h_nessi {

void rel2abs_times_beta(int n, double beta, double *trel, double *tabs) {
  for( int i = 0; i < n; i++) {
    tabs[i] = trel[i] < 0. ? beta * (trel[i]+1) : beta * trel[i];
  }
}

dlr_info::dlr_info(int &r, double lambda, double eps, double beta, int nao, int xi, int ntaumax) {
  // DL-1..DL-6 (tier 2). eps >= 1e-15 keeps the DLR rank well below libdlr's hard limit of 500 nodes (oidx(500) in
  // dlr_it_build, ISSUES L1/H5); the maintainer fixed the bound at 1e-15 (audit Q-7).
  HNESSI_REQUIRE(std::isfinite(lambda) && lambda > 0, "dlr_info::dlr_info", HNESSI_KV(lambda));
  HNESSI_REQUIRE(eps >= 1e-15 && eps < 1, "dlr_info::dlr_info", HNESSI_KV(eps),
                 detail::note{"eps below 1e-15 can exceed libdlr's limit of 500 DLR nodes (ISSUES H5, L1)"});
  HNESSI_REQUIRE(std::isfinite(beta) && beta > 0, "dlr_info::dlr_info", HNESSI_KV(beta));
  HNESSI_REQUIRE(nao >= 1, "dlr_info::dlr_info", HNESSI_KV(nao));
  HNESSI_REQUIRE(xi == -1 || xi == 1, "dlr_info::dlr_info", HNESSI_KV(xi));
  HNESSI_REQUIRE(ntaumax >= 1, "dlr_info::dlr_info", HNESSI_KV(ntaumax));

  eps_ = eps;
  lambda_ = lambda;
  beta_ = beta;
  size1_ = nao;
  size2_ = nao;
  xi_ = xi;

  // libdlr's dlr_it_build ignores the capacity passed in and can return up to 500 nodes (its internal bound), so the
  // node buffers are sized to that limit, not to the user's ntaumax (H5); they are shrunk to r below.
  const int libdlr_max_rank = 500;
  int input_r = std::max(ntaumax, libdlr_max_rank);
  dlrrf_.resize(input_r); // DLR real frequencies
  dlrit_.resize(input_r); // DLR imaginary time nodes

  // libdlr routines with SAVE'd locals are not thread safe: dlr_it_build is now inside the critical section too (H3),
  // so dlr_info objects may be constructed concurrently.
  #pragma omp critical(h_nessi_libdlr)
  {
    c_dlr_it_build(&lambda_, &eps_, &input_r, dlrrf_.data(), dlrit_.data()); // Get imaginary time nodes and frequencies
  }

  // input_r has been overwritten with the number of dlr nodes
  HNESSI_ENSURE(input_r >= 1 && input_r <= libdlr_max_rank, "dlr_info::dlr_info", HNESSI_KV(input_r), HNESSI_KV(lambda),
                HNESSI_KV(eps));
  HNESSI_REQUIRE(input_r <= ntaumax, "dlr_info::dlr_info", HNESSI_KV(input_r), HNESSI_KV(ntaumax),
                 detail::note{"the DLR rank for this lambda and eps exceeds the requested maximum ntaumax"});
  dlrrf_.resize(input_r);
  dlrit_.resize(input_r);
  r = input_r;
  r_ = input_r;

  it0B_.resize(r);
  cf2it_.resize(r*r);  // coeficients [cf] to imaginary time [it]
  it2cf_.resize(r*r);  // Imaginary time [it] to coeficients [cf]
  it2cfp_.resize(r);         // Imaginary time to coeficients [pivots]
  it2itr_.resize(r*r); // Imaginary time to reversed imaginary time
  phi_.resize(r*r*r); // Convolution tensor
  ipmat_.resize(r*r);    // Inner product weights 

  // these libdlr routines are reentrant (doc 04 §4.3-4.11); the critical section is kept as a safeguard
  #pragma omp critical(h_nessi_libdlr)
  { 
    c_dlr_it2cf_init(&r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data()); // Get it to cf
    c_dlr_convtens(&beta_, &xi_, &r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data(), phi_.data()); // Get convolution tensor
    c_dlr_ipmat(&beta_, &r, dlrit_.data(), dlrrf_.data(), it2cf_.data(), it2cfp_.data(), ipmat_.data());
    c_dlr_cf2it_init(&r,dlrrf_.data(),dlrit_.data(),cf2it_.data());
    c_dlr_it2itr_init(&r, dlrrf_.data(), dlrit_.data(), it2cf_.data(), it2cfp_.data(), it2itr_.data()); // Imaginary time to reversed imaginary time
  }
  rel2abs_times_beta(r, beta_, dlrit_.data(), it0B_.data()); // Convert from relative format in dlr to absolute values from 0 to beta

  Gijc_.resize(r*size1_*size2_);
  res_.resize(size1_*size2_);
  LU_.compute(DMatrixMap(cf2it_.data(), r, r).transpose());
}



void dlr_info::check_tau_(const char *where, double tau) const {
  HNESSI_REQUIRE(tau >= 0 && tau <= beta_ * (1 + 1e-14), where, HNESSI_KV(tau), HNESSI_KV(beta_),
                 detail::note{"the imaginary time must lie in [0, beta]"});
}

void dlr_info::eval_point(double tau, double *Gtij, double *Mij) {
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  check_tau_("dlr_info::eval_point", tau);
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
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  check_tau_("dlr_info::eval_point", tau);
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
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  check_tau_("dlr_info::eval_point", tau);
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
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  for(Eigen::Index i = 0; i < tau.size(); i++) check_tau_("dlr_info::eval_point", tau(i));   // DL-9
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  int ntau = tau.size();
  DColVector taurel(ntau);  // B3: c_abs2rel writes ntau entries
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data()+i, res_.data());
    DMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::eval_point(DColVector &tau, double *Gtij, cplx *Mtij) {
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  for(Eigen::Index i = 0; i < tau.size(); i++) check_tau_("dlr_info::eval_point", tau(i));   // DL-9
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  int ntau = tau.size();
  DColVector taurel(ntau);  // B3: c_abs2rel writes ntau entries
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(DMatrixMap(Gtij, r_, es)).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data()+i, res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::eval_point(DColVector &tau, cplx *Gtij, cplx *Mtij) {
  detail::owner_scope busy(owner_, "dlr_info::eval_point");   // ST-10
  for(Eigen::Index i = 0; i < tau.size(); i++) check_tau_("dlr_info::eval_point", tau(i));   // DL-9
  int es = size1_*size2_;
  int one = 1;
  
  DColVector tau01 = tau/beta_;
  int ntau = tau.size();
  DColVector taurel(ntau);  // B3: c_abs2rel writes ntau entries
  c_abs2rel(&ntau, tau01.data(), taurel.data());

  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).real()).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data()+i, res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) = DMatrixMap(res_.data(), size1_, size2_);  
  }
  DMatrixMap(Gijc_.data(), es, r_).noalias() = LU_.solve(ZMatrixMap(Gtij, r_, es).imag()).transpose();
  for(int i = 0; i < ntau; i++) {
    c_dlr_it_eval(&r_, &size1_, dlrrf_.data(), Gijc_.data(), taurel.data()+i, res_.data());
    ZMatrixMap(Mtij+i*es, size1_, size2_) += cplx(0.,1.) * DMatrixMap(res_.data(), size1_, size2_);  
  }
}

void dlr_info::write_to_hdf5(h5e::File &out, std::string label) {
  h5e::dump(out, h5_join({label, "r"}), r_);
  h5e::dump(out, h5_join({label, "beta"}), beta_);
  h5e::dump(out, h5_join({label, "eps"}), eps_);
  h5e::dump(out, h5_join({label, "lambda"}), lambda_);
  h5e::dump(out, h5_join({label, "xi"}), xi_);
  h5e::dump(out, h5_join({label, "it0B"}), DMatrixMap(it0B_.data(), r_, 1));
}
}
#endif
