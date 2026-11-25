#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"


namespace h_nessi {

// Raw pointers and maps: hot paths, so every index check here is tier 1 (HNESSI_ASSERT; doc 61 §2.2, LX-3, LX-9, LX-10).

DMatrixMap herm_matrix_hodlr::map_mat(int i){
    HNESSI_ASSERT(0 <= i && i < r_, "herm_matrix_hodlr::map_mat", HNESSI_KV(i), HNESSI_KV(r_));   // HM-A2
    return DMatrixMap(mat_.data()+i*size1_*size2_, size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_ret_curr(int t1, int t2){
    check_ret_curr_("herm_matrix_hodlr::map_ret_curr", t1, t2);   // HM-A10, LX-10: only t2 <= t1 is stored
    return ZMatrixMap(curr_timestep_ret_ptr(t1,t2), size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_les_curr(int t1, int t2){
    check_les_curr_("herm_matrix_hodlr::map_les_curr", t1, t2);   // HM-A11, LX-10: only t1 <= t2 is stored
    return ZMatrixMap(curr_timestep_les_ptr(t1,t2), size1_, size2_);
}

ZMatrixMap herm_matrix_hodlr::map_tv(int t1, int t2){
    return ZMatrixMap(tvptr(t1,t2), size1_, size2_);
}

double* herm_matrix_hodlr::matptr(int i){
    HNESSI_ASSERT(0 <= i && i < r_, "herm_matrix_hodlr::matptr", HNESSI_KV(i), HNESSI_KV(r_));   // HM-A1
    return mat_.data() + i * size1_ * size2_;
}

cplx* herm_matrix_hodlr::tvptr(int t, int tau) {
    check_tv_("herm_matrix_hodlr::tvptr", t, tau);   // HM-A3, LX-3: tv_ holds rows 0..nt-1
    return tv_.data() + t*r_*size1_*size2_ + tau*size1_*size2_;
}

cplx* herm_matrix_hodlr::retptr_col(int t1, int t2){
    return ret_.dirtricol()+time2direct_col(t1,t2)*size1_*size2_;   // HM-A6: checked in time2direct_col
}

cplx* herm_matrix_hodlr::retptr_corr(int t1, int t2){
    // HM-A7: (t1,t2) lies in the correction strip below the yellow triangle of column t2, at most k rows below t2
    HNESSI_ASSERT(0 <= t2 && t2 < nt_ && r2_dir_[t2] < t1 && t1 <= t2 + k_, "herm_matrix_hodlr::retptr_corr",
                  HNESSI_KV(t1), HNESSI_KV(t2), HNESSI_KV(k_), HNESSI_KV(nt_));
    return ret_corr_below_tri_.data() + (ret_corr_index_t_[t2] + (t1-r2_dir_[t2]-1))*size1_*size2_;
}

cplx* herm_matrix_hodlr::curr_timestep_ret_ptr(int t, int tp) {
    // HM-A8, LX-9: row t in the window (a row below the window would silently alias a newer row of the same slot);
    // the slot holds columns 0..nt
    HNESSI_ASSERT(tstpmk_ <= t && t <= tstpmk_+k_ && 0 <= tp && tp <= nt_, "herm_matrix_hodlr::curr_timestep_ret_ptr",
                  HNESSI_KV(t), HNESSI_KV(tp), HNESSI_KV(tstpmk_), HNESSI_KV(k_), HNESSI_KV(nt_));
    int index = t%(k_+1);
    return curr_timestep_ret_.data() + index * (nt_+1) * size1_ * size2_ + tp * size1_ * size2_;
}

cplx* herm_matrix_hodlr::curr_timestep_les_ptr(int t, int tp) {
    // HM-A9, LX-9: column tp in the window; the slot holds rows 0..nt
    HNESSI_ASSERT(tstpmk_ <= tp && tp <= tstpmk_+k_ && 0 <= t && t <= nt_, "herm_matrix_hodlr::curr_timestep_les_ptr",
                  HNESSI_KV(t), HNESSI_KV(tp), HNESSI_KV(tstpmk_), HNESSI_KV(k_), HNESSI_KV(nt_));
    int index = tp%(k_+1);
    return curr_timestep_les_.data() + index * (nt_ +1) * size1_ * size2_ + t * size1_ * size2_;
}

} // namespace
