
/**
 * @file dlr.hpp
 * @brief Declares the dlr_info class for handling Discrete Lehmann Representation (DLR) data and operations in quantum many-body calculations.
 *
 * This file provides the dlr_info class, which manages DLR-related arrays, transformation matrices, and evaluation routines for imaginary time and frequency grids.
 */
#ifndef DLR_DECL
#define DLR_DECL

#include "utils.hpp"
#include <vector>

extern "C"
{
  #include "dlr_c/dlr_c.h"
}


/**
 * @namespace h_nessi
 * @brief Namespace for hierarchical matrix algorithms and quantum many-body data structures.
 */
namespace h_nessi {

/**
 * @class dlr_info
 * @brief Manages DLR (Discrete Lehmann Representation) data, transformations, and evaluation routines.
 *
 * The dlr_info class encapsulates all data and operations required for working with the DLR basis in quantum many-body calculations. It provides access to DLR grids, transformation matrices, convolution/integration tensors, and routines for evaluating functions at arbitrary imaginary time points. It also supports writing DLR data to HDF5 files for persistent storage.
 */
class dlr_info {

private:
  int r_;              /**< DLR rank (number of basis functions). */
  double beta_;        /**< Inverse temperature parameter. */
  double eps_;         /**< DLR accuracy parameter (epsilon). */
  double lambda_;      /**< DLR lambda parameter. */
  int size1_;          /**< Size parameter 1 (number of orbitals). */
  int size2_;          /**< Size parameter 2 (number of orbitals). */
  int xi_;             /**< Particle statistics (+1 for bosons, -1 for fermions). */

  std::vector<double> dlrrf_;      /**< DLR frequency grid array. */
  std::vector<double> dlrit_;      /**< DLR imaginary time grid array. */
  std::vector<double> it0B_;       /**< Imaginary time grid between 0 and beta. */
  std::vector<double> it2cf_;      /**< Transformation matrix from imaginary time to coefficients. */
  std::vector<double> cf2it_;      /**< Transformation matrix from coefficients to imaginary time. */
  std::vector<double> it2itr_;     /**< Transformation matrix from imaginary time to reflected imaginary time. */
  std::vector<int> it2cfp_;        /**< Integer pivots for it2cf transformation. */
  std::vector<double> phi_;        /**< Imaginary-time convolution tensor. */
  std::vector<double> ipmat_;      /**< Imaginary time integration tensor. */

  std::vector<double> Gijc_;       /**< Temporary storage for dlr coefficients. */
  std::vector<double> res_;        /**< Temporary storage for results. */
  Eigen::PartialPivLU<DMatrix> LU_; /**< LU decomposition object for matrix operations. */


public:
  /**
   * @brief Constructs a dlr_info object and initializes DLR data arrays and transformation matrices.
   * @param r Reference to DLR rank parameter.
   * @param lambda DLR lambda parameter.
   * @param eps DLR epsilon parameter (accuracy).
   * @param beta Inverse temperature.
   * @param nao Number of atomic orbitals.
   * @param xi Particle statistics (+1 for bosons, -1 for fermions).
   * @param ntaumax, typically dlr needs much less than 1000 basis functions.  Only change this parameter if more than 1000 basis functions are needed for a given lambda and eps
   */
  dlr_info(int &r, double lambda, double eps, double beta, int nao, int xi, int ntaumax = 1000);

  /**
   * @brief Returns pointer to DLR frequency grid array.
   * @return Pointer to frequency grid.
   */
  double *dlrrf() { return dlrrf_.data(); };
  const double *dlrrf() const { return dlrrf_.data(); };
  /**
   * @brief Returns pointer to DLR imaginary time grid array.
   * @return Pointer to imaginary time grid.
   */
  double *dlrit() { return dlrit_.data(); };
  const double *dlrit() const { return dlrit_.data(); };
  /**
   * @brief Returns pointer to it0B array (imaginary time grid between 0 and beta).
   * @return Pointer to it0B array.
   */
  double *it0B() { return it0B_.data(); };
  const double *it0B() const { return it0B_.data(); };
  /**
   * @brief Returns pointer to cf2it array (transformation matrix from coefficients to imaginary time).
   * @return Pointer to cf2it array.
   */
  double *cf2it() { return cf2it_.data(); };
  const double *cf2it() const { return cf2it_.data(); };
  /**
   * @brief Returns pointer to it2cf array (transformation matrix from imaginary time to coefficients).
   * @return Pointer to it2cf array.
   */
  double *it2cf() { return it2cf_.data(); };
  const double *it2cf() const { return it2cf_.data(); };
  /**
   * @brief Returns pointer to it2itr array (transformation matrix from imaginary time to reflected imaginary time).
   * @return Pointer to it2itr array.
   */
  double *it2itr() { return it2itr_.data(); };
  const double *it2itr() const { return it2itr_.data(); };
  /**
   * @brief Returns pointer to it2cfp array (integer pivots for it2cf transformation).
   * @return Pointer to it2cfp array.
   */
  int *it2cfp() { return it2cfp_.data(); };
  const int *it2cfp() const { return it2cfp_.data(); };
  /**
   * @brief Returns pointer to phi array (imaginary-time convolution tensor).
   * @return Pointer to phi array.
   */
  double *phi() { return phi_.data(); };
  const double *phi() const { return phi_.data(); };
  /**
   * @brief Returns pointer to ipmat array (imaginary time integration tensor).
   * @return Pointer to ipmat array.
   */
  double *ipmat() { return ipmat_.data(); };
  const double *ipmat() const { return ipmat_.data(); };

  /**
   * @brief Returns value from it0B array at index i.
   * @param i Index
   * @return Value at it0B_[i]
   */
  double it0B(int i) const { return it0B_[i]; };
  /**
   * @brief Returns value from dlrit array at index i.
   * @param i Index
   * @return Value at dlrit_[i]
   */
  double dlrit(int i) const { return dlrit_[i]; };
  /**
   * @brief Returns beta parameter (inverse temperature).
   * @return Beta value.
   */
  double beta() const {return beta_; };
  /**
   * @brief Returns epsilon parameter (DLR accuracy).
   * @return Epsilon value.
   */
  double eps() const {return eps_; };
  /**
   * @brief Returns lambda parameter.
   * @return Lambda value.
   */
  double lambda() const {return lambda_; };
  /**
   * @brief Returns size1 parameter.
   * @return Size1 value.
   */
  int size1() const {return size1_; };
  /**
   * @brief Returns size2 parameter.
   * @return Size2 value.
   */
  int size2() const {return size2_; };
  /**
   * @brief Returns DLR rank parameter.
   * @return DLR rank.
   */
  int r() const {return r_;};
  /**
   * @brief Returns xi parameter (particle statistics).
   * @return Xi value.
   */
  int xi() const {return xi_;};

  /**
   * @brief Evaluates the DLR expansion at a single imaginary time point.
   * @param tau Imaginary time value.
   * @param Gtij Input array (double) - values of function at DLR imaginary time points.
   * @param Mij Output array (double).
   */
  void eval_point(double tau, double *Gtij, double *Mij);
  /**
   * @brief Evaluates the DLR expansion at a single imaginary time point (complex output).
   * @param tau Imaginary time value.
   * @param Gtij Input array (double) - values of function at DLR imaginary time points.
   * @param Mij Output array (complex).
   */
  void eval_point(double tau, double *Gtij, cplx *Mij);
  /**
   * @brief Evaluates the DLR expansion at a single imaginary time point (complex input/output).
   * @param tau Imaginary time value.
   * @param Gtij Input array (complex) - values of function at DLR imaginary time points.
   * @param Mij Output array (complex).
   */
  void eval_point(double tau, cplx *Gtij, cplx *Mij);
  /**
   * @brief Evaluates the DLR expansion at multiple imaginary time points.
   * @param tau Imaginary time vector.
   * @param Gtij Input array (double) - values of function at DLR imaginary time points.
   * @param Mtij Output array (double).
   */
  void eval_point(DColVector &tau, double *Gtij, double *Mtij);
  /**
   * @brief Evaluates the DLR expansion at multiple imaginary time points (complex output).
   * @param tau Imaginary time vector.
   * @param Gtij Input array (double) - values of function at DLR imaginary time points.
   * @param Mtij Output array (complex).
   */
  void eval_point(DColVector &tau, double *Gtij, cplx *Mtij);
  /**
   * @brief Evaluates the DLR expansion at multiple imaginary time points (complex input/output).
   * @param tau Imaginary time vector.
   * @param Gtij Input array (complex) - values of function at DLR imaginary time points.
   * @param Mtij Output array (complex).
   */
  void eval_point(DColVector &tau, cplx *Gtij, cplx *Mtij);

  /**
   * @brief Writes DLR data to an HDF5 file.
   * @param out HDF5 file handle.
   * @param label Dataset label.
   */
  void write_to_hdf5(h5e::File &out, std::string label);
};

}
#endif
