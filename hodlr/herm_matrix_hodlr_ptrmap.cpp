#include <vector>
#include <iostream>
#include <string>

#include "herm_matrix_hodlr.hpp"


namespace hodlr {

DMatrixMap herm_matrix_hodlr::map_mat(int i){
    return DMatrixMap(mat_.data()+i*size1_*size2_, size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_ret_curr(int t1, int t2){
    assert(t1 <= tstpmk_+k_);
    assert(t1 >= tstpmk_);
    return ZMatrixMap(curr_timestep_ret_ptr(t1,t2), size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_les_curr(int t1, int t2){
    assert(t2 <= tstpmk_+k_);
    assert(t2 >= tstpmk_);
    return ZMatrixMap(curr_timestep_les_ptr(t1,t2), size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_tv(int t1, int t2){
    return ZMatrixMap(tvptr(t1,t2), size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_tv_trans(int t1, int t2){
    return ZMatrixMap(tvptr_trans(t1,t2), size1_, size2_);
}

double* herm_matrix_hodlr::matptr(int i){
  assert(i < r_ && i >=0);
    return mat_.data() + i * size1_ * size2_;
}

cplx* herm_matrix_hodlr::tvptr(int t, int tau) {
  assert(t <= nt_ && tau < r_);
  return tv_.data() + t*r_*size1_*size2_ + tau*size1_*size2_;
}

cplx* herm_matrix_hodlr::tvptr_trans(int t, int tau) {
  assert(t <= nt_ && tau < r_);
  return tv_.data_trans() + t*r_*size1_*size2_ + tau*size1_*size2_;
}

cplx* herm_matrix_hodlr::retptr_col(int t1, int t2){
  return ret_.dirtricol()+time2direct_col(t1,t2)*size1_*size2_;
}

cplx* herm_matrix_hodlr::retptr_corr(int t1, int t2){
  return ret_corr_below_tri_.data() + (ret_corr_index_t_[t2] + (t1-r2_dir_[t2]-1))*size1_*size2_;
}

cplx* herm_matrix_hodlr::curr_timestep_ret_ptr(int t, int tp) {
  assert(t <= tstpmk_+k_);
  int index = t%(k_+1);
  return curr_timestep_ret_.data() + index * (nt_+1) * size1_ * size2_ + tp * size1_ * size2_;
}

cplx* herm_matrix_hodlr::curr_timestep_les_ptr(int t, int tp) {
  assert(tp <= tstpmk_+k_);
  int index = tp%(k_+1);
  return curr_timestep_les_.data() + index * (nt_ +1) * size1_ * size2_ + t * size1_ * size2_;
}

} // namespace
