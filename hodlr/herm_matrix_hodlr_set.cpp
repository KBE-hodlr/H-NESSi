#include <vector>
#include <iostream>
#include <string>

#include "herm_matrix_hodlr.hpp"


namespace hodlr {

void herm_matrix_hodlr::set_tstp_zero(int tstp) {
    assert( tstp == -1 || (tstp >= tstpmk_ && tstp <= tstpmk_+k_));
    if(tstp == -1) {
    std::fill(mat_.begin(), mat_.end(), 0.0);
    }
    else {
      memset(curr_timestep_les_ptr(0,tstp), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(curr_timestep_ret_ptr(tstp,0), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(tvptr(tstp,0), 0, sizeof(cplx) * r_ * size1_ * size2_);
    }
}

void herm_matrix_hodlr::set_mat_zero() {
    std::fill(mat_.begin(), mat_.end(), 0.0);
}

void herm_matrix_hodlr::set_tv_tstp_zero(int tstp) {
  assert(tstp >= tstpmk_ && tstp <= tstpmk_+k_);
  memset(tvptr(tstp,0), 0, sizeof(cplx) * r_ * size1_ * size2_);
}

void herm_matrix_hodlr::set_les_tstp_zero(int tstp) {
  assert(tstp >= tstpmk_ && tstp <= tstpmk_+k_);
  memset(curr_timestep_les_ptr(0,tstp), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
}

void herm_matrix_hodlr::set_ret_tstp_zero(int tstp) {
  assert(tstp >= tstpmk_ && tstp <= tstpmk_+k_);
  memset(curr_timestep_ret_ptr(tstp,0), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
}

void herm_matrix_hodlr::set_mat(int t1, double *M) {
  assert(t1 < r_ && t1 >=0);
    memcpy(mat_.data() + t1*size1_*size2_, M, size1_*size2_*sizeof(double));
}

void herm_matrix_hodlr::set_mat(int t1, DMatrix &M) {
  assert(t1 < r_ && t1 >=0);
    memcpy(mat_.data() + t1*size1_*size2_, M.data(), size1_*size2_*sizeof(double));
}

void herm_matrix_hodlr::set_tv(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(tvptr(t1,t2),M,size1_*size2_*sizeof(cplx));
    ZMatrixMap(tvptr_trans(t1,t2), size2_, size1_) = ZMatrixMap(tvptr(t1,t2), size1_, size2_).transpose();
}

void herm_matrix_hodlr::set_tv(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    set_tv(t1, t2, M.data());
}

void herm_matrix_hodlr::set_ret_curr(int t1, int t2, cplx *M){
    memcpy(curr_timestep_ret_ptr(t1,t2),M,size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::set_ret_curr(int t1, int t2, ZMatrix &M){
    set_ret_curr(t1, t2, M.data());
}

void herm_matrix_hodlr::set_les_curr(int t1, int t2, cplx *M){
    memcpy(curr_timestep_les_ptr(t1,t2),M,size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::set_les_curr(int t1, int t2, ZMatrix &M){
    set_les_curr(t1, t2, M.data());
}

} // namespace
