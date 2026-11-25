#include <iostream>
#include "h_nessi/block.hpp"
#include "h_nessi/utils.hpp"
#include "h_nessi/checks.hpp"

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
    // BK-1 (tier 1)
    HNESSI_ASSERT(rows >= 0 && cols >= 0 && epsrank >= 0 && svdtol >= 0, "block::block", HNESSI_KV(rows),
                  HNESSI_KV(cols), HNESSI_KV(epsrank), HNESSI_KV(svdtol));
    rows_=rows;
    cols_=cols;
    epsrank_=epsrank;
    svdtol_=svdtol;
    S_=DColVector::Zero(epsrank_);
    U_=ZMatrix::Zero(epsrank_,epsrank_);
    V_=ZMatrix::Zero(cols_,epsrank_);
}

void block::set_rank(int epsrank){
        HNESSI_ASSERT(epsrank >= 0, "block::set_rank", HNESSI_KV(epsrank));   // BK-2 (tier 1)
        epsrank_=epsrank;
}

} // namespace 
