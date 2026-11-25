#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"

namespace h_nessi {

// dlr pairing (tier 2, HM-I1..HM-I6): the dlr_info evaluates with its own r and orbital size, and G is indexed with
// r_, size1_, size2_; a mismatch reads or writes out of bounds.
void herm_matrix_hodlr::check_dlr_(const char *where, const dlr_info &dlr) const {
  HNESSI_REQUIRE(dlr.r() == r_ && dlr.size1() == size1_ && dlr.size2() == size2_, where, HNESSI_KV(dlr.r()),
                 HNESSI_KV(r_), HNESSI_KV(dlr.size1()), HNESSI_KV(size1_),
                 detail::note{"the dlr_info must have the DLR rank and orbital size of this object"});
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, DMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, double *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau", dlr);
  dlr.eval_point(tau, tvptr(tstp, 0), M.data());
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau", dlr);
  dlr.eval_point(tau, tvptr(tstp, 0), M);
}





void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, DMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, double *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau_array", dlr);
  dlr.eval_point(taus, tvptr(tstp,0), M.data());
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau_array", dlr);
  dlr.eval_point(taus, tvptr(tstp,0), M);
}



// return (r,size1,size2) array where data[n,i,j] = G^M[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, double *M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    DMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, std::complex<double> *M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    ZMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, DMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    DMatrixMap(M.data(), r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, DMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    int ij_flat = i * size2_ + j;
    
    Eigen::Map<const DMatrix, 0, Eigen::InnerStride<>> mat_vec(
        mat_.data() + ij_flat,
        r_,
        1,
        Eigen::InnerStride<>(size1_ * size2_)
    );
    
    DMatrixMap(M.data(), r_, 1) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * mat_vec;
}


// return (r,size1,size2) array where data[n,i,j] = G^tv[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, cplx *M) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
  ZMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr(tstp, 0), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
  ZMatrixMap(M.data(), r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr(tstp, 0), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
    int ij_flat = i * size2_ + j;
    
    Eigen::Map<const ZMatrix, 0, Eigen::InnerStride<>> tv_vec(
        tvptr(tstp, 0) + ij_flat,
        r_,
        1,
        Eigen::InnerStride<>(size1_ * size2_)
    );
    
    ZMatrixMap(M.data(), r_, 1) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * tv_vec;
}


// return (r,size1,size2) array where data[n,i,j] = G^vt[tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, cplx *M) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
  ZMatrixMap(M, r_, size1_*size2_) = -sig_ * (DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr_trans(tstp, 0), r_, size1_*size2_)).conjugate();
}
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, ZMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
  ZMatrixMap(M.data(), r_, size1_*size2_) = -sig_ * (DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr_trans(tstp, 0), r_, size1_*size2_)).conjugate();
}
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, ZMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
    int ij_flat = i * size2_ + j;
    
    Eigen::Map<const ZMatrix, 0, Eigen::InnerStride<>> tv_trans_vec(
        tvptr_trans(tstp, 0) + ij_flat,
        r_,
        1,
        Eigen::InnerStride<>(size1_ * size2_)
    );
    
    ZMatrixMap(M.data(), r_, 1) = -sig_ * (DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * tv_trans_vec).conjugate();
}

} // namespace
