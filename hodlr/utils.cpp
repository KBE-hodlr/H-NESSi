#include "utils.hpp"
#include <Eigen/SVD>


namespace hodlr {
	const bool RHO_DIAGONAL=true;
	const bool RHO_HORIZONTAL=false;

	void svd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S){
		Eigen::BDCSVD<ZMatrix> svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		V=svd.matrixV();
		U=svd.matrixU();
		S=svd.singularValues();
	}

	int tsvd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S,double svdtol){
		Eigen::BDCSVD<ZMatrix> svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		int epsrank=(svd.singularValues().array() >= svdtol).count(); 	
		V=svd.matrixV().block(0,0,M.cols(),epsrank);
		U=svd.matrixU().block(0,0,M.rows(),epsrank);
		S=svd.singularValues().array().head(epsrank);
		return epsrank;
	}

	void svd( ZMatrix &M,DColVector &S){
		Eigen::BDCSVD<ZMatrix> svd(M);
		S=svd.singularValues();
	}
}
