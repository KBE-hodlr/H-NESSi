#include "h_nessi/utils.hpp"
#include <Eigen/SVD>
#include "h_nessi/checks.hpp"


namespace h_nessi {
	const bool RHO_DIAGONAL=true;
	const bool RHO_HORIZONTAL=false;

	void svd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S){
		HNESSI_ASSERT(M.size() > 0, "svd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()));   // UT-1 (tier 1)
		Eigen::BDCSVD<ZMatrix> svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		V=svd.matrixV();
		U=svd.matrixU();
		S=svd.singularValues();
	}

	int tsvd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S,double svdtol){
		HNESSI_ASSERT(M.size() > 0 && svdtol >= 0, "tsvd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()),
		              HNESSI_KV(svdtol));   // UT-2 (tier 1)
		Eigen::BDCSVD<ZMatrix> svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV);
		int epsrank=(svd.singularValues().array() >= svdtol).count(); 	
		V=svd.matrixV().block(0,0,M.cols(),epsrank);
		U=svd.matrixU().block(0,0,M.rows(),epsrank);
		S=svd.singularValues().array().head(epsrank);
		return epsrank;
	}

	void svd( ZMatrix &M,DColVector &S){
		HNESSI_ASSERT(M.size() > 0, "svd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()));   // UT-1 (tier 1)
		Eigen::BDCSVD<ZMatrix> svd(M);
		S=svd.singularValues();
	}
}
