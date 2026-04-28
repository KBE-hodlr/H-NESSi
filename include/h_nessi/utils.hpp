#ifndef UTILS_DECL
#define UTILS_DECL

#define PI 3.1415926535897932384626433832795028841971693

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <highfive/H5Easy.hpp>
#include <omp.h>

namespace h5 = HighFive;
namespace h5e= H5Easy;

namespace h_nessi {
  extern const int RHO_DIAGONAL_2;
  extern const int RHO_DIAGONAL;
  extern const int RHO_HORIZONTAL;

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
  void svd(ZMatrix &M, ZMatrix &U, ZMatrix &V, DColVector &S);
  int  tsvd(ZMatrix &M, ZMatrix &U, ZMatrix &V, DColVector &S,double svdtol);
  void svd(ZMatrix &M, DColVector &S);

} // namespace

#endif // header guard


