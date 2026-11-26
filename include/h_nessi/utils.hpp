#ifndef UTILS_DECL
#define UTILS_DECL

#define PI 3.1415926535897932384626433832795028841971693

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <highfive/H5Easy.hpp>
#include <omp.h>
#include <initializer_list>
#include <string>

namespace h5 = HighFive;
namespace h5e= H5Easy;

namespace h_nessi {
  extern const bool RHO_DIAGONAL;
  extern const bool RHO_HORIZONTAL;

  using cplx = std::complex<double>;
 
  // Matrix Aliases
  // template <typename T>
  // using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // template <typename T>
  // using MatrixMap = Eigen::Map<Matrix<T>>;
  // template <typename T>
  // using MatrixConstMap = Eigen::Map<const Matrix<T>>;
  
  // template <typename T>
  // using RowVector = Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>;
  // template <typename T>
  // using RowVectorMap = Eigen::Map<RowVector<T>>;
  // template <typename T>
  // using RowVectorConstMap = Eigen::Map<const RowVector<T>>;
  
  // template <typename T>
  // using ColVector = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor>;
  // template <typename T>
  // using ColVectorMap = Eigen::Map<ColVector<T>>;
  // template <typename T>
  // using ColVectorConstMap = Eigen::Map<const ColVector<T>>;
  
  using DMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // using ZMatrix = Matrix<cplx>;
  using ZMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IMatrix = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using IColVector = Eigen::Matrix<int, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using DColVector = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>;
  using ZColVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>;
  // using DRowVector = RowVector<double>;
  using ZRowVector = Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic, Eigen::RowMajor>;
  
  // using DMatrixMap = MatrixMap<double>;
  // using ZMatrixMap = MatrixMap<std::complex<double>>;
  using ZMatrixMap = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using DMatrixMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using IMatrixMap = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  using DMatrixConstMap = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  // using ZFMatrixMap = MatrixMap<std::complex<float>>;
  // using ZLMatrixMap = MatrixMap<std::complex<long double>>;
  using DColVectorMap = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor>>;
  using ZColVectorMap = Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1, Eigen::ColMajor>>;
  // using ZFColVectorMap = ColVectorMap<std::complex<float>>;
  // using ZLColVectorMap = ColVectorMap<std::complex<long double>>;
  // using DRowVectorMap = RowVectorMap<double>;
  using ZRowVectorMap = Eigen::Map<Eigen::Matrix<std::complex<double>,1, Eigen::Dynamic,  Eigen::RowMajor>>;
  using IRowVectorMap = Eigen::Map<Eigen::Matrix<int, 1, Eigen::Dynamic,  Eigen::RowMajor>>;
  using IColVectorMap = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 1,  Eigen::ColMajor>>;
  // using ZFRowVectorMap = RowVectorMap<std::complex<float>>;
  // using ZLRowVectorMap = RowVectorMap<std::complex<long double>>;
  
  // using DMatrixConstMap = MatrixConstMap<double>;
  // using FMatrixConstMap = MatrixConstMap<float>;
  // using LMatrixConstMap = MatrixConstMap<long double>;
  // using ZMatrixConstMap = MatrixConstMap<std::complex<double>>;
  // using DColVectorConstMap = ColVectorConstMap<double>;
  // using ZColVectorConstMap = ColVectorConstMap<std::complex<double>>;
  // using DRowVectorConstMap = RowVectorConstMap<double>;
  // using ZRowVectorConstMap = RowVectorConstMap<std::complex<double>>;
  /** Thin SVD M = U diag(S) V^H (S descending). BUGS B38: Eigen 3.4.0's BDCSVD can return a wrong factorisation,
   *  so for max(rows, cols) >= 16 the BDCSVD result is verified (residual and orthogonality <= 50 n eps) and
   *  recomputed with JacobiSVD if the check fails; below 16 JacobiSVD is used directly. */
  void svd(ZMatrix &M, ZMatrix &U, ZMatrix &V, DColVector &S);
  /** Truncated SVD: the singular triplets with S_i >= svdtol (same verified SVD as svd()). Returns the rank. */
  int  tsvd(ZMatrix &M, ZMatrix &U, ZMatrix &V, DColVector &S,double svdtol);
  /** Singular values only (same verified SVD as svd(); U and V are computed for the check). */
  void svd(ZMatrix &M, DColVector &S);

  /** Which SVD produced a result (returned by arrow_svd, for tests and diagnostics). */
  enum class svd_path { jacobi = 0, bdc = 1, bdc_rejected_jacobi = 2 };

  /** SVD of the (r+1)x(r+1) arrow matrix K = [diag(s) 0; w beta] of the incremental TSVD (herm_matrix_hodlr::updatetsvd,
   *  doc 12), s_i >= 0 and beta >= 0 real, w complex (1 x r). K = P A P^H with the real arrow matrix
   *  A = [diag(s) 0; |w| beta] and P = diag(conj(w_i)/|w_i|, 1); A is decomposed by a real SVD (JacobiSVD for
   *  r+1 < 40, else BDCSVD verified in O(r^2) + orthogonality, JacobiSVD fallback: B38), and U = P U_A, V = P V_A,
   *  so K = U diag(S) V^H with S descending. Tier 1 (Debug): asserts ||A - U_A S V_A^T||_F <= 50 n eps ||A||_F. */
  svd_path arrow_svd(const DColVector &s, const ZRowVector &w, double beta, ZMatrix &U, ZMatrix &V, DColVector &S);

  /** HDF5 path join (WP3 side finding 5): joins the non-empty parts with single '/' separators, so callers may pass
   *  labels with or without a trailing '/' ("G" and "G/" both give "G/name", not "G//name"). A leading '/' of the
   *  first part is kept; a trailing '/' is dropped. h5_join("", "x") == "x", which names the same object of a file
   *  as the "/x" that "" + "/x" gave before: the file layout is unchanged. */
  inline std::string h5_join(std::initializer_list<std::string> parts) {
    std::string out;
    bool first = true;
    for(const std::string &p : parts) {
      if(p.empty()) continue;
      if(!first) out += '/';
      out += p;
      first = false;
    }
    std::string res;
    res.reserve(out.size());
    for(char c : out) if(!(c == '/' && !res.empty() && res.back() == '/')) res += c;
    if(res.size() > 1 && res.back() == '/') res.pop_back();
    return res;
  }

} // namespace

#endif // header guard


