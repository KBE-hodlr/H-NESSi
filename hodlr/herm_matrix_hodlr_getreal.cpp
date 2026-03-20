#include <vector>
#include <iostream>
#include <string>

#include "herm_matrix_hodlr.hpp"

namespace hodlr {

void herm_matrix_hodlr::density_matrix(int tstp, dlr_info &dlr, DMatrix &M){
  assert(M.rows() == size1_ && M.cols() == size2_);
  assert(tstp >= -1 && tstp <= nt_);
    if (tstp == -1) {
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
  assert(M.rows() == size1_ && M.cols() == size2_);
  assert(tstp >= -1 && tstp <= nt_);
    if (tstp == -1) {
        get_mat_tau(dlr.beta(),dlr,M);
        M *= -1.;
    } else {
        get_les(tstp, tstp, M);
        M *= std::complex<double>(0.0, 1.0 * sig_);
    }
}

void herm_matrix_hodlr::get_ret(int t1, int t2, cplx *M){
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
            ret_.get_compress(t2-blkr1_[b],t1-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_ret(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<=nt_);
    M.resize(size1_,size2_);
    get_ret(t1, t2, M.data());
}

void herm_matrix_hodlr::get_ret_curr(int t1, int t2, cplx *M){
    assert(t1 <= tstpmk_+k_);
    assert(t1 >= tstpmk_);
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
    assert(t1 <= tstpmk_+k_);
    assert(t1 >= tstpmk_);
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
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t2,t1);
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
            int index = first_index + row*width + col;
            ZMatrixMap(M, size1_, size2_).noalias() = Eigen::Map<Eigen::Matrix<cplx, -1, -1, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(les_dir_square_.data() + index, size1_, size2_, Eigen::InnerStride<>(len_les_dir_square_));
        }else{               // compressed get
            int b=time2block(t1,t2);
            les_.get_compress(t1-blkr1_[b],t2-blkc1_[b],b,M);
        }
        ZMatrixMap MMap(M, size1_, size2_);
        MMap = MMap.adjoint().eval();
        MMap *= -1.0;
    }
}

void herm_matrix_hodlr::get_les(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<=nt_);
    M.resize(size1_,size2_);

    get_les(t1, t2, M.data());
}

void herm_matrix_hodlr::get_les_curr(int t1, int t2, cplx *M){
    assert(t2 <= tstpmk_+k_);
    assert(t2 >= tstpmk_);
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
    assert(t2 <= tstpmk_+k_);
    assert(t2 >= tstpmk_);
    M.resize(size1_,size2_);
    get_les_curr(t1, t2, M.data());
}

void herm_matrix_hodlr::get_tv(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(M,tvptr(t1,t2),size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::get_tv(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    M.resize(size1_,size2_);
    get_tv(t1, t2, M.data());
}

void herm_matrix_hodlr::get_tv_trans(int t1, int t2, cplx* M){
    assert(t1<=nt_ && t2<r_);
    memcpy(M,tvptr_trans(t1,t2),size1_*size2_*sizeof(cplx));
}

void herm_matrix_hodlr::get_tv_trans(int t1, int t2, ZMatrix &M){
    assert(t1<=nt_ && t2<r_);
    M.resize(size1_,size2_);
    get_tv_trans(t1, t2, M.data());
}

void herm_matrix_hodlr::get_mat(int t1, DMatrix &M) {
  assert(t1 < r_ && t1 >=0);
    memcpy(M.data(), mat_.data() + t1*size1_*size2_, size1_*size2_*sizeof(double));
}

} // namespace 
