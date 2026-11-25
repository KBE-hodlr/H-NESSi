#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"


namespace h_nessi {

// Setters: element-wise hot paths, tier 1 (doc 61 §2.3 HM-S*, LX-5, LX-26).
// ST-6 (DY-X3): a write into row tstp() between update_blocks and extrapolate overwrites the slot that still holds row
// tstp()-k-1, which extrapolate reads; every setter that writes row tstp() therefore clears can_extrap_ (extrapolate is
// then a no-op, doc 32 inv. 6). Writes through the maps/raw pointers cannot be tracked.

void herm_matrix_hodlr::set_tstp_zero(int tstp) {
    HNESSI_ASSERT(tstp == -1 || (tstp >= tstpmk_ && tstp <= tstpmk_+k_), "herm_matrix_hodlr::set_tstp_zero",
                  HNESSI_KV(tstp), HNESSI_KV(tstpmk_), HNESSI_KV(k_));   // HM-S1
    if(tstp == -1) {
    std::fill(mat_.begin(), mat_.end(), 0.0);
    }
    else {
      if(tstp == tstpmk_+k_) can_extrap_ = false;   // ST-6
      memset(curr_timestep_les_ptr(0,tstp), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(curr_timestep_ret_ptr(tstp,0), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
      memset(tvptr(tstp,0), 0, sizeof(cplx) * r_ * size1_ * size2_);
    }
}

void herm_matrix_hodlr::set_mat_zero() {
    std::fill(mat_.begin(), mat_.end(), 0.0);
}

void herm_matrix_hodlr::set_tv_tstp_zero(int tstp) {
  HNESSI_ASSERT(tstp >= tstpmk_ && tstp <= tstpmk_+k_, "herm_matrix_hodlr::set_tv_tstp_zero", HNESSI_KV(tstp),
                HNESSI_KV(tstpmk_), HNESSI_KV(k_));   // HM-S3
  if(tstp == tstpmk_+k_) can_extrap_ = false;   // ST-6
  memset(tvptr(tstp,0), 0, sizeof(cplx) * r_ * size1_ * size2_);
}

void herm_matrix_hodlr::set_les_tstp_zero(int tstp) {
  HNESSI_ASSERT(tstp >= tstpmk_ && tstp <= tstpmk_+k_, "herm_matrix_hodlr::set_les_tstp_zero", HNESSI_KV(tstp),
                HNESSI_KV(tstpmk_), HNESSI_KV(k_));   // HM-S3
  if(tstp == tstpmk_+k_) can_extrap_ = false;   // ST-6
  memset(curr_timestep_les_ptr(0,tstp), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
}

void herm_matrix_hodlr::set_ret_tstp_zero(int tstp) {
  HNESSI_ASSERT(tstp >= tstpmk_ && tstp <= tstpmk_+k_, "herm_matrix_hodlr::set_ret_tstp_zero", HNESSI_KV(tstp),
                HNESSI_KV(tstpmk_), HNESSI_KV(k_));   // HM-S3
  if(tstp == tstpmk_+k_) can_extrap_ = false;   // ST-6
  memset(curr_timestep_ret_ptr(tstp,0), 0, sizeof(cplx) * (tstp+1) * size1_ * size2_);
}

void herm_matrix_hodlr::set_mat(int t1, const double *M) {
    memcpy(matptr(t1), M, size1_*size2_*sizeof(double));   // HM-S4: index checked in matptr
}

void herm_matrix_hodlr::set_mat(int t1, const DMatrix &M) {
    // HM-S4, LX-26: memcpy reads size1*size2 entries of M
    HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::set_mat", HNESSI_KV(M.rows()),
                  HNESSI_KV(M.cols()), HNESSI_KV(size1_), HNESSI_KV(size2_));
    set_mat(t1, M.data());
}

void herm_matrix_hodlr::set_tv(int t1, int t2, const cplx* M){
    check_tv_("herm_matrix_hodlr::set_tv", t1, t2);   // HM-S6, LX-5
    if(t1 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    memcpy(tvptr(t1,t2),M,size1_*size2_*sizeof(cplx));
    ZMatrixMap(tvptr_trans(t1,t2), size2_, size1_) = ZMatrixMap(tvptr(t1,t2), size1_, size2_).transpose();
}

void herm_matrix_hodlr::set_tv(int t1, int t2, const ZMatrix &M){
    HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::set_tv", HNESSI_KV(M.rows()),
                  HNESSI_KV(M.cols()), HNESSI_KV(size1_), HNESSI_KV(size2_));   // LX-26
    set_tv(t1, t2, M.data());
}

void herm_matrix_hodlr::set_ret_curr(int t1, int t2, const cplx *M){
    check_ret_curr_("herm_matrix_hodlr::set_ret_curr", t1, t2);   // HM-S8
    if(t1 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    memcpy(curr_timestep_ret_ptr(t1,t2),M,size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::set_ret_curr(int t1, int t2, const ZMatrix &M){
    HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::set_ret_curr", HNESSI_KV(M.rows()),
                  HNESSI_KV(M.cols()), HNESSI_KV(size1_), HNESSI_KV(size2_));   // LX-26
    set_ret_curr(t1, t2, M.data());
}

void herm_matrix_hodlr::set_les_curr(int t1, int t2, const cplx *M){
    check_les_curr_("herm_matrix_hodlr::set_les_curr", t1, t2);   // HM-S9
    if(t2 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    memcpy(curr_timestep_les_ptr(t1,t2),M,size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::set_les_curr(int t1, int t2, const ZMatrix &M){
    HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::set_les_curr", HNESSI_KV(M.rows()),
                  HNESSI_KV(M.cols()), HNESSI_KV(size1_), HNESSI_KV(size2_));   // LX-26
    set_les_curr(t1, t2, M.data());
}

} // namespace
