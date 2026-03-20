#include <iostream>
#include "h_nessi/block.hpp"
#include "h_nessi/utils.hpp"

#include <Eigen/Core>


namespace h_nessi {

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
}

} // namespace 
