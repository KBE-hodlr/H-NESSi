#include <iostream>
#include "block.hpp"
#include "utils.hpp"

#include <Eigen/Core>


namespace hodlr {

block::block(){
    rows_=0;
    cols_=0;
    epsrank_=0;
    svdtol_=0;
    S_=DColVector::Zero(epsrank_);
    U_=ZMatrix::Zero(epsrank_,epsrank_);
    V_=ZMatrix::Zero(cols_,epsrank_);
}

block::block(int rows,int cols,int epsrank,double svdtol) {
    rows_=rows;
    cols_=cols;
    epsrank_=epsrank;
    svdtol_=svdtol;
    S_=DColVector::Zero(epsrank_);
    U_=ZMatrix::Zero(epsrank_,epsrank_);
    V_=ZMatrix::Zero(cols_,epsrank_);
}

void block::set_rank(int epsrank){
        epsrank_=epsrank;
// eigen handles resizing already
//        U_.conservativeResize(rows_,epsrank_);
//        V_.conservativeResize(cols_,epsrank_);
//        S_.conservativeResize(epsrank_);   
}

} // namespace hodlr
