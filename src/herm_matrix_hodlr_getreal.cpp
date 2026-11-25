#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"

namespace h_nessi {

// HM-R13, LX-7 (tier 2, once per call): -1 <= tstp <= last (row tstp computed and allocated)
void herm_matrix_hodlr::check_row_(const char *where, int tstp) const {
  HNESSI_REQUIRE(0 <= tstp && tstp <= last_row_(), where, HNESSI_KV(tstp), HNESSI_KV(tstpmk_ + k_), HNESSI_KV(nt_),
                 detail::note{"row tstp must be computed (0 <= tstp <= min(tstp(), nt()-1))"});
}

void herm_matrix_hodlr::density_matrix(int tstp, dlr_info &dlr, DMatrix &M){
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::density_matrix", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  if(tstp != -1) check_row_("herm_matrix_hodlr::density_matrix", tstp);
    if (tstp == -1) {
        check_dlr_("herm_matrix_hodlr::density_matrix", dlr);
        get_mat_tau(dlr.beta(),dlr,M);
        M *= (-1.0);
    } else {
        ZMatrix tmpM(size1_,size2_);
        get_les(tstp, tstp, tmpM);
        tmpM *= std::complex<double>(0.0, 1.0 * sig_);
        M=tmpM.real();
    }
}

void herm_matrix_hodlr::density_matrix(int tstp, dlr_info &dlr, ZMatrix &M){
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::density_matrix", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  if(tstp != -1) check_row_("herm_matrix_hodlr::density_matrix", tstp);
    if (tstp == -1) {
        check_dlr_("herm_matrix_hodlr::density_matrix", dlr);
        get_mat_tau(dlr.beta(),dlr,M);
        M *= -1.;
    } else {
        get_les(tstp, tstp, M);
        M *= std::complex<double>(0.0, 1.0 * sig_);
    }
}

void herm_matrix_hodlr::get_ret(int t1, int t2, cplx *M){
    // HM-R1, LX-6 (tier 1): a row above tstp() would read an aliased circular slot
    HNESSI_ASSERT(0 <= t1 && t1 <= last_row_() && 0 <= t2 && t2 <= last_row_(), "herm_matrix_hodlr::get_ret",
                  HNESSI_KV(t1), HNESSI_KV(t2), HNESSI_KV(tstpmk_ + k_), HNESSI_KV(nt_));
    if(t1 >= t2){
        if( t1 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_ret_ptr(t1,t2), size1_, size2_);
        }
        else if(t2>=c1_dir_[t1]){ // direct get
            ret_.get_direct_col(time2direct_col(t1,t2),M);
        }
        else if(t1-t2<=k_) { // ret_corr
          get_ret_corr(t1,t2,M);
        }
        else if(t2<=k_) { // ret_left_edge
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(ret_left_edge_.data() + (t1*(k_+1)+t2)*size1_*size2_, size1_, size2_);
        }
        else {               // compressed get
            int b=time2block(t1,t2);
            HNESSI_ASSERT(b < built_blocks_ && t1-blkr1_[b] < blklen_[b], "herm_matrix_hodlr::get_ret", HNESSI_KV(b),
                          HNESSI_KV(built_blocks_), HNESSI_KV(t1), HNESSI_KV(blklen_[b]));   // HM-R3: row archived
            ret_.get_compress(t1-blkr1_[b],t2-blkc1_[b],b,M);
        }
    }else{
        if( t2 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_ret_ptr(t2,t1), size1_, size2_); 
        }
        else if(t1>=c1_dir_[t2]){ // direct get
            ret_.get_direct_col(time2direct_col(t2,t1),M);
        }
        else if(t2-t1<=k_) { // ret_corr
          get_ret_corr(t2,t1,M);
        }
        else if(t1<=k_) { // ret_left_edge
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(ret_left_edge_.data() + (t2*(k_+1)+t1)*size2_*size1_, size1_, size2_);
        }
        else {               // compressed get
            int b=time2block(t2,t1);
            HNESSI_ASSERT(b < built_blocks_ && t2-blkr1_[b] < blklen_[b], "herm_matrix_hodlr::get_ret", HNESSI_KV(b),
                          HNESSI_KV(built_blocks_), HNESSI_KV(t2), HNESSI_KV(blklen_[b]));   // HM-R3
            ret_.get_compress(t2-blkr1_[b],t1-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);
    get_ret(t1, t2, M.data());
}

void herm_matrix_hodlr::get_ret_curr(int t1, int t2, cplx *M){
    // HM-R4, LX-8: the stored row is max(t1,t2)
    HNESSI_ASSERT(tstpmk_ <= std::max(t1,t2) && std::max(t1,t2) <= tstpmk_+k_ && std::min(t1,t2) >= 0,
                  "herm_matrix_hodlr::get_ret_curr", HNESSI_KV(t1), HNESSI_KV(t2), HNESSI_KV(tstpmk_), HNESSI_KV(k_));
    if(t1>=t2){
        memcpy(M,curr_timestep_ret_ptr(t1,t2),size1_*size2_*sizeof(cplx));
    }else{
        memcpy(M,curr_timestep_ret_ptr(t2,t1),size1_*size2_*sizeof(cplx));
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret_curr(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);
    get_ret_curr(t1, t2, M.data());
}

void herm_matrix_hodlr::get_ret_corr(int t1, int t2, cplx *M){
    if(t1>=t2){
        memcpy(M,retptr_corr(t1,t2),size1_*size2_*sizeof(cplx));
    }else{
        memcpy(M,retptr_corr(t2,t1),size1_*size2_*sizeof(cplx));
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret_corr(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);
    get_ret_corr(t1, t2, M.data());
}


void herm_matrix_hodlr::get_les(int t1, int t2, cplx* M){
    // HM-R6, LX-6 (tier 1)
    HNESSI_ASSERT(0 <= t1 && t1 <= last_row_() && 0 <= t2 && t2 <= last_row_(), "herm_matrix_hodlr::get_les",
                  HNESSI_KV(t1), HNESSI_KV(t2), HNESSI_KV(tstpmk_ + k_), HNESSI_KV(nt_));
    if(t2 >= t1){
        if( t2 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_).noalias() = ZMatrixMap(curr_timestep_les_ptr(t1,t2), size1_, size2_);
        }
        else if(t1>=c1_dir_[t2]){ // direct get
            int dirlvl = t_to_dirlvl_[t2];
            int first_index = les_dir_square_first_index_[dirlvl];
            int row = t2 - blkr1(dirlvl-1);
            int col = t1 - c1_dir_[t2];
            int width = blkdirheight_[dirlvl];
            HNESSI_ASSERT(0 <= row && row < width && 0 <= col && col < width, "herm_matrix_hodlr::get_les",
                          HNESSI_KV(row), HNESSI_KV(col), HNESSI_KV(width));   // HM-R8: inside the direct square
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t2,t1);
            HNESSI_ASSERT(b < built_blocks_ && t2-blkr1_[b] < blklen_[b], "herm_matrix_hodlr::get_les", HNESSI_KV(b),
                          HNESSI_KV(built_blocks_), HNESSI_KV(t2), HNESSI_KV(blklen_[b]));   // HM-R8 (blue)
            les_.get_compress(t2-blkr1_[b],t1-blkc1_[b],b,M);
        }
    }else{
        if( t1 >= tstpmk_) {
          ZMatrixMap(M, size1_, size2_) = ZMatrixMap(curr_timestep_les_ptr(t2,t1), size1_, size2_);
        }
        else if(t2>=c1_dir_[t1]){ // direct get
            int dirlvl = t_to_dirlvl_[t1];
            int first_index = les_dir_square_first_index_[dirlvl];
            int row = t1 - blkr1(dirlvl-1);
            int col = t2 - c1_dir_[t1];
            int width = blkdirheight_[dirlvl];
            HNESSI_ASSERT(0 <= row && row < width && 0 <= col && col < width, "herm_matrix_hodlr::get_les",
                          HNESSI_KV(row), HNESSI_KV(col), HNESSI_KV(width));   // HM-R8
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t1,t2);
            HNESSI_ASSERT(b < built_blocks_ && t1-blkr1_[b] < blklen_[b], "herm_matrix_hodlr::get_les", HNESSI_KV(b),
                          HNESSI_KV(built_blocks_), HNESSI_KV(t1), HNESSI_KV(blklen_[b]));   // HM-R8 (blue)
            les_.get_compress(t1-blkr1_[b],t2-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_les(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);

    get_les(t1, t2, M.data());
}

void herm_matrix_hodlr::get_les_curr(int t1, int t2, cplx *M){
    // HM-R9, LX-8: the stored column is max(t1,t2)
    HNESSI_ASSERT(tstpmk_ <= std::max(t1,t2) && std::max(t1,t2) <= tstpmk_+k_ && std::min(t1,t2) >= 0,
                  "herm_matrix_hodlr::get_les_curr", HNESSI_KV(t1), HNESSI_KV(t2), HNESSI_KV(tstpmk_), HNESSI_KV(k_));
    if(t1<=t2){
        memcpy(M,curr_timestep_les_ptr(t1,t2),size1_*size2_*sizeof(cplx));
    }else{
        memcpy(M,curr_timestep_les_ptr(t2,t1),size1_*size2_*sizeof(cplx));
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_les_curr(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);
    get_les_curr(t1, t2, M.data());
}

void herm_matrix_hodlr::get_tv(int t1, int t2, cplx* M){
    check_tv_("herm_matrix_hodlr::get_tv", t1, t2);   // HM-R10, LX-5
    memcpy(M,tvptr(t1,t2),size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::get_tv(int t1, int t2, ZMatrix &M){
    M.resize(size1_,size2_);
    get_tv(t1, t2, M.data());
}

void herm_matrix_hodlr::get_mat(int t1, DMatrix &M) {
  // HM-R12: memcpy writes size1*size2 entries into M, which is not resized
  HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::get_mat", HNESSI_KV(M.rows()),
                HNESSI_KV(M.cols()), HNESSI_KV(size1_), HNESSI_KV(size2_));
  memcpy(M.data(), matptr(t1), size1_*size2_*sizeof(double));
}

} // namespace 
