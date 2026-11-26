#include "h_nessi/utils.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include "h_nessi/checks.hpp"


namespace h_nessi {
	const bool RHO_DIAGONAL=true;
	const bool RHO_HORIZONTAL=false;

namespace {
	// BUGS B38: Eigen 3.4.0's BDCSVD (divide and conquer) returns a wrong factorisation for some matrices of size >= 16,
	// complex AND real, with orthonormal U and V (so nothing looks wrong). Eigen 3.4.1 is correct on every case we
	// found, but the library only requires Eigen >= 3.4. Every BDCSVD result is therefore verified, and JacobiSVD
	// (unconditionally accurate, slower for large matrices) is used when the check fails.
	//
	// Acceptance: residual / ||M||_F and the loss of orthogonality of U and V all <= svd_check_tol(n) = 50 n eps.
	// Correct BDCSVD results stay below 2.5e-14 for n <= 121 (20000 random arrow matrices); the 3.4.0 failures seen
	// range from 5.6e-13 to 6e-3.
	double svd_check_tol(Eigen::Index n) {
		return 50.0 * static_cast<double>(std::max<Eigen::Index>(n, 1)) * std::numeric_limits<double>::epsilon();
	}

	// ||Q^H Q - I||_F of a matrix that should have orthonormal columns
	template <class Mat> double orth_defect(const Mat &Q) {
		Mat G = Q.adjoint() * Q;
		G.diagonal().array() -= 1.0;
		return G.norm();
	}

	// Thin SVD of a general complex matrix: BDCSVD, verified, JacobiSVD as fallback. Returns the path taken.
	svd_path verified_svd(const ZMatrix &M, ZMatrix &U, ZMatrix &V, DColVector &S) {
		const Eigen::Index n = std::max(M.rows(), M.cols());
		if (n < 16) {   // BDCSVD uses JacobiSVD internally below its switch size 16 anyway
			Eigen::JacobiSVD<ZMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
			U = svd.matrixU(); V = svd.matrixV(); S = svd.singularValues();
			return svd_path::jacobi;
		}
		{
			Eigen::BDCSVD<ZMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
			U = svd.matrixU(); V = svd.matrixV(); S = svd.singularValues();
			const double tol = svd_check_tol(n);
			const double res = (M * V - U * S.asDiagonal()).norm();
			if (std::isfinite(res) && res <= tol * M.norm() && orth_defect(U) <= tol && orth_defect(V) <= tol)
				return svd_path::bdc;
		}
		Eigen::JacobiSVD<ZMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
		U = svd.matrixU(); V = svd.matrixV(); S = svd.singularValues();
		return svd_path::bdc_rejected_jacobi;
	}
}  // namespace

	void svd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S){
		HNESSI_ASSERT(M.size() > 0, "svd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()));   // UT-1 (tier 1)
		verified_svd(M, U, V, S);
	}

	int tsvd( ZMatrix &M,ZMatrix &U,ZMatrix &V,DColVector &S,double svdtol){
		HNESSI_ASSERT(M.size() > 0 && svdtol >= 0, "tsvd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()),
		              HNESSI_KV(svdtol));   // UT-2 (tier 1)
		ZMatrix Uf, Vf;
		DColVector Sf;
		verified_svd(M, Uf, Vf, Sf);
		int epsrank=(Sf.array() >= svdtol).count();
		V=Vf.leftCols(epsrank);
		U=Uf.leftCols(epsrank);
		S=Sf.head(epsrank);
		return epsrank;
	}

	void svd( ZMatrix &M,DColVector &S){
		HNESSI_ASSERT(M.size() > 0, "svd", HNESSI_KV(M.rows()), HNESSI_KV(M.cols()));   // UT-1 (tier 1)
		ZMatrix U, V;   // needed to verify the BDCSVD result (B38)
		verified_svd(M, U, V, S);
	}

	svd_path arrow_svd(const DColVector &s, const ZRowVector &w, double beta, ZMatrix &U, ZMatrix &V, DColVector &S){
		const int r = s.size();
		const int n = r + 1;
		HNESSI_ASSERT(w.size() == r && beta >= 0 && (s.array() >= 0).all(), "arrow_svd", HNESSI_KV(r),
		              HNESSI_KV(w.size()), HNESSI_KV(beta));   // tier 1
		// K = [diag(s) 0; w beta] = P A P^H with the real arrow matrix A = [diag(s) 0; |w| beta] and the diagonal unitary
		// P = diag(p_0, ..., p_{r-1}, 1), p_i = conj(w_i)/|w_i| (1 if w_i = 0):
		//   (P A P^H)_{ii} = |p_i|^2 s_i = s_i,  (P A P^H)_{r i} = |w_i| conj(p_i) = w_i,  (P A P^H)_{r r} = beta.
		// With A = U_A S V_A^T:  K = (P U_A) S (P V_A)^H, so U = P U_A and V = P V_A (row i times p_i), and the
		// singular values (order included) are those of A.
		Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);   // column-major: what BDCSVD/JacobiSVD use internally
		ZColVector p = ZColVector::Ones(n);
		for (int i = 0; i < r; i++) {
			A(i, i) = s(i);
			const double a = std::abs(w(i));
			A(r, i) = a;
			if (a > 0) p(i) = std::conj(w(i)) / a;
		}
		A(r, r) = beta;

		Eigen::MatrixXd UA, VA;
		svd_path path;
		if (n < 40) {   // JacobiSVD is as fast as BDCSVD up to n ~ 40 and needs no check
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			UA = svd.matrixU(); VA = svd.matrixV(); S = svd.singularValues();
			path = svd_path::jacobi;
		} else {
			Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
			UA = svd.matrixU(); VA = svd.matrixV(); S = svd.singularValues();
			// verification (B38): A V_A - U_A S in O(n^2) from the arrow structure, and the orthogonality of U_A, V_A
			Eigen::MatrixXd R(n, n);
			R.topRows(r).noalias() = s.asDiagonal() * VA.topRows(r);
			R.row(r).noalias() = A.row(r).head(r) * VA.topRows(r);
			R.row(r) += beta * VA.row(r);
			R.noalias() -= UA * S.asDiagonal();
			const double tol = svd_check_tol(n);
			const double res = R.norm();
			if (std::isfinite(res) && res <= tol * A.norm() && orth_defect(UA) <= tol && orth_defect(VA) <= tol) {
				path = svd_path::bdc;
			} else {
				Eigen::JacobiSVD<Eigen::MatrixXd> jsvd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
				UA = jsvd.matrixU(); VA = jsvd.matrixV(); S = jsvd.singularValues();
				path = svd_path::bdc_rejected_jacobi;
			}
		}
		// tier 1: the returned factorisation reproduces A
		HNESSI_ASSERT((A - UA * S.asDiagonal() * VA.transpose()).norm() <= svd_check_tol(n) * A.norm(), "arrow_svd",
		              HNESSI_KV(n), HNESSI_KV(A.norm()), HNESSI_KV((A - UA * S.asDiagonal() * VA.transpose()).norm()),
		              HNESSI_KV(static_cast<int>(path)));
		U = p.asDiagonal() * UA.cast<cplx>();
		V = p.asDiagonal() * VA.cast<cplx>();
		return path;
	}
}
