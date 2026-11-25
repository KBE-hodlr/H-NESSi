#include <iostream>
#include "h_nessi/blocks_keldysh.hpp"
#include "h_nessi/checks.hpp"

namespace h_nessi {

/* #######################################################################################
#
#  Blocks list
#
########################################################################################*/

blocks_list::blocks_list() : blocks_() {
    nbox_=0;
    size1_=0;
    size2_=0;
}


blocks_list::blocks_list(int nbox, std::vector<int>& nrows, std::vector<int>& ncols,
                         double svdtol, int size1, int size2)
    : nbox_(nbox), size1_(size1), size2_(size2),
      blocks_(nbox, std::vector<std::vector<block>>(size1, std::vector<block>(size2)))
{
    HNESSI_ASSERT(nrows.size() == nbox && ncols.size() == nbox, "blocks_list::blocks_list", HNESSI_KV(nrows.size()),
                  HNESSI_KV(ncols.size()), HNESSI_KV(nbox));   // BK-3
    // BK-4 (tier 1)
    HNESSI_ASSERT(nbox >= 0 && size1 >= 1 && size2 >= 1, "blocks_list::blocks_list", HNESSI_KV(nbox), HNESSI_KV(size1),
                  HNESSI_KV(size2));
#ifndef NDEBUG
    for (int n = 0; n < nbox; ++n)
        HNESSI_ASSERT(nrows[n] >= 1 && ncols[n] >= 1, "blocks_list::blocks_list", HNESSI_KV(n), HNESSI_KV(nrows[n]),
                      HNESSI_KV(ncols[n]));
#endif

    // Initialize each block
    for (int n = 0; n < nbox_; ++n) {
        for (int i = 0; i < size1_; ++i) {
            for (int j = 0; j < size2_; ++j) {
                blocks_[n][i][j] = block(nrows[n], ncols[n], 0, svdtol); 
            }
        }
    }
}


// t1 and t2 are measured from the edge of the block
void blocks_list::get_compress(int t1,int t2,int b,ZMatrix &M){
  // BK-5: M is not resized
  HNESSI_ASSERT(M.rows() == size1_ && M.cols() == size2_, "blocks_list::get_compress", HNESSI_KV(M.rows()),
                HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  get_compress(t1, t2, b, M.data());
}


void blocks_list::get_compress(int t1,int t2,int b,cplx *M){
    HNESSI_ASSERT(0 <= b && b < nbox_, "blocks_list::get_compress", HNESSI_KV(b), HNESSI_KV(nbox_));   // BK-5
    for(int i=0;i<size1_;i++){
        for(int j=0;j<size2_;j++){
            int epsrank = blocks_[b][i][j].epsrank();
            // BK-5 (tier 1): row t1 built, column t2 inside the block, U/S/V consistent with epsrank
            HNESSI_ASSERT(0 <= t1 && t1 < blocks_[b][i][j].U().rows() && 0 <= t2 && t2 < blocks_[b][i][j].V().rows() &&
                          blocks_[b][i][j].S().size() >= epsrank && blocks_[b][i][j].U().cols() == epsrank &&
                          blocks_[b][i][j].V().cols() == epsrank, "blocks_list::get_compress", HNESSI_KV(t1),
                          HNESSI_KV(t2), HNESSI_KV(b), HNESSI_KV(blocks_[b][i][j].U().rows()),
                          HNESSI_KV(blocks_[b][i][j].V().rows()), HNESSI_KV(epsrank));
            DColVectorMap S = DColVectorMap(blocks_[b][i][j].Sdata(), epsrank);
            ZColVectorMap Ut1 = ZColVectorMap(blocks_[b][i][j].Udata() + t1*epsrank, epsrank);
            ZColVectorMap Vt2 = ZColVectorMap(blocks_[b][i][j].Vdata() + t2*epsrank, epsrank);
            M[i*size2_ + j] = Vt2.dot(Ut1.cwiseProduct(S));
        }
    }
}

/* #######################################################################################
#
#  Specialization for retarded
#
########################################################################################*/


ret_blocks::ret_blocks() {
    data_=blocks_list();
    ndir_ = 0;
    dirtricol_ = std::vector<std::complex<double>>(0);
}


ret_blocks::ret_blocks(int nbox,int ndir,std::vector<int> &nrows,std::vector<int> &ncols,double svdtol,int size1,int size2) :
data_(nbox,nrows,ncols,svdtol,size1,size2)
{
    HNESSI_ASSERT(ndir >= 0, "ret_blocks::ret_blocks", HNESSI_KV(ndir));   // BK-7
//    data_=blocks_list(nbox,nrows,ncols,svdtol,size1,size2);
    ndir_=ndir;
    dirtricol_ = std::vector<std::complex<double>>(ndir_*size1*size2);
}


void ret_blocks::set_direct_col(int i,ZMatrix &M){
    // BK-8, LX-18 (tier 1)
    HNESSI_ASSERT(0 <= i && i < ndir_ && M.rows() == data_.size1() && M.cols() == data_.size2(),
                  "ret_blocks::set_direct_col", HNESSI_KV(i), HNESSI_KV(ndir_), HNESSI_KV(M.rows()), HNESSI_KV(M.cols()));
    ZMatrixMap(dirtricol_.data() + i*data_.size1()*data_.size2(), data_.size1(), data_.size2()) = M;
}

void ret_blocks::get_direct_col(int i,ZMatrix &M){
    HNESSI_ASSERT(0 <= i && i < ndir_, "ret_blocks::get_direct_col", HNESSI_KV(i), HNESSI_KV(ndir_));   // BK-8, LX-18
    M = ZMatrixMap(dirtricol_.data() + i*data_.size1()*data_.size2(), data_.size1(), data_.size2());
}

void ret_blocks::get_direct_col(int i,cplx *M){
    HNESSI_ASSERT(0 <= i && i < ndir_, "ret_blocks::get_direct_col", HNESSI_KV(i), HNESSI_KV(ndir_));   // BK-8, LX-18
    memcpy(M, dirtricol_.data()+i*data_.size1()*data_.size2(), data_.size1()*data_.size2()*sizeof(cplx));
}

// t1 and t2 are measured from the edge of the block
void ret_blocks::get_compress(int t1,int t2,int b,ZMatrix &M){
    data_.get_compress(t1,t2,b,M);   // BK-6: checked in blocks_list
}


// t1 and t2 are measured from the edge of the block
void ret_blocks::get_compress(int t1,int t2,int b,cplx *M){
    data_.get_compress(t1,t2,b,M);
}

/* #######################################################################################
#
#  Specialization for lesser
#
########################################################################################*/

les_blocks::les_blocks() {
    data_=blocks_list();
 }

les_blocks::les_blocks(int nbox,std::vector<int> &nrows,std::vector<int> &ncols,double svdtol,int size1,int size2) :
data_(nbox,nrows,ncols,svdtol,size1,size2)
{
}


void les_blocks::get_compress(int t1,int t2,int b,ZMatrix &M){
    data_.get_compress(t1,t2,b,M);   // BK-6: checked in blocks_list
}

void les_blocks::get_compress(int t1,int t2,int b,cplx *M){
    data_.get_compress(t1,t2,b,M);
}


/* #######################################################################################
#
#  Specialization for mixed
#
########################################################################################*/
tv_blocks::tv_blocks() :
    data_(0,cplx(0.,0.)) {
}

tv_blocks::tv_blocks(int size1,int size2,int nt,int ntau,double svdtol) : 
      data_(nt*ntau*size1*size2,std::complex<double>(0.,0.)) {
    HNESSI_ASSERT(size1 >= 1 && size2 >= 1 && nt >= 1 && ntau >= 1, "tv_blocks::tv_blocks", HNESSI_KV(size1),
                  HNESSI_KV(size2), HNESSI_KV(nt), HNESSI_KV(ntau));   // BK-9
}

} // namespace
