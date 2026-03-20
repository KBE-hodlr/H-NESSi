#include <iostream>
#include "h_nessi/blocks_keldysh.hpp"

namespace hodlr {

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
      blocks_(nbox, std::vector<std::vector<hodlr::block>>(size1, std::vector<hodlr::block>(size2)))
{
    assert(nrows.size() == nbox && ncols.size() == nbox);

    // Initialize each hodlr::block
    for (int n = 0; n < nbox_; ++n) {
        for (int i = 0; i < size1_; ++i) {
            for (int j = 0; j < size2_; ++j) {
                blocks_[n][i][j] = hodlr::block(nrows[n], ncols[n], 0, svdtol); 
            }
        }
    }
}


// t1 and t2 are measured from the edge of the block
void blocks_list::get_compress(int t1,int t2,int b,ZMatrix &M){
  get_compress(t1, t2, b, M.data());
}


void blocks_list::get_compress(int t1,int t2,int b,cplx *M){
    for(int i=0;i<size1_;i++){
        for(int j=0;j<size2_;j++){
            int epsrank = blocks_[b][i][j].epsrank();
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
//    data_=blocks_list(nbox,nrows,ncols,svdtol,size1,size2);
    ndir_=ndir;
    dirtricol_ = std::vector<std::complex<double>>(ndir_*size1*size2);
}


void ret_blocks::set_direct_col(int i,ZMatrix &M){
    assert(i<ndir_);
    ZMatrixMap(dirtricol_.data() + i*data_.size1()*data_.size2(), data_.size1(), data_.size2()) = M;
}

void ret_blocks::get_direct_col(int i,ZMatrix &M){
    assert(i<ndir_);
    M = ZMatrixMap(dirtricol_.data() + i*data_.size1()*data_.size2(), data_.size1(), data_.size2());
}

void ret_blocks::get_direct_col(int i,cplx *M){
    assert(i<ndir_);
    memcpy(M, dirtricol_.data()+i*data_.size1()*data_.size2(), data_.size1()*data_.size2()*sizeof(cplx));
}

// t1 and t2 are measured from the edge of the block
void ret_blocks::get_compress(int t1,int t2,int b,ZMatrix &M){
    data_.get_compress(t1,t2,b,M.data());
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
    data_.get_compress(t1,t2,b,M.data());
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
    data_(0,cplx(0.,0.)),
    data_trans_(0,cplx(0.,0.)) {
}

tv_blocks::tv_blocks(int size1,int size2,int nt,int ntau,double svdtol) : 
      data_(nt*ntau*size1*size2,std::complex<double>(0.,0.)),
      data_trans_(nt*ntau*size1*size2,std::complex<double>(0.,0.)) {
}

} // namespace hodlr
