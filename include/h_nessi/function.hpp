
/**
 * @file function.hpp
 * @brief Declares the function class for storing and manipulating time-dependent matrix data in the HODLR library.
 *
 * This file provides the function class, which manages arrays of complex matrix data indexed by time slices, supporting value access, assignment, and initialization routines.
 */
#ifndef FUNCTION_DECL
#define FUNCTION_DECL

#include "utils.hpp"
#include <vector>


/**
 * @namespace h_nessi
 * @brief Namespace for hierarchical matrix algorithms and data structures.
 */
namespace h_nessi {

/**
 * @class function
 * @brief Stores and manipulates time-dependent single-time functions on the Keldysh Contour, with initial value on the Matsubara axis.
 *
 * The function class manages a contiguous array of complex matrix data indexed by time slices. It provides methods for value access, assignment, zeroing, and setting constant values across all time slices.
 */
class function {
private:
  std::vector<cplx> data_;        /**< Contiguous complex data array. */
  int nt_;            /**< Number of time slices. */
  int element_size_;  /**< Size of matrix-valued function (size1 * size2). */
  int size2_;         /**< Number of columns in each matrix. */
  int size1_;         /**< Number of rows in each matrix. */

public:
  /**
   * @brief Default constructor. Initializes an empty function object.
   */
  function();

  /**
   * @brief Constructs a function object from a checkpoint file
   * @param file input file from h5e.
   * @param label label of the dataset.
   */
  function(h5e::File &in, std::string label);

  /**
   * @brief Constructs a function object with specified time slices and matrix dimensions.
   * @param nt Number of time slices.
   * @param size1 Number of rows in each matrix.
   * @param size2 Number of columns in each matrix.
   */
  function(int nt, int size1, int size2);

  /**
   * @brief Copy constructor.
   * @param f Function object to copy.
   */
  function(const function &f);

  /**
   * @brief Assignment operator.
   * @param f Function object to assign from.
   * @return Reference to this object.
   */
  function &operator=(const function &f);

  /**
   * @brief Move constructor.
   * @param f Function object to move from.
   */
  function(function &&f) noexcept;

  /**
   * @brief write a checkpoint file which can be used to restart the calculation.
   * @param out file to be written to.
   * @param label name of dataset to be created.
   */
  void write_checkpoint_hdf5(h5e::File &out, std::string label);

  /**
   * @brief write function to output file.
   * @param out file to be written to.
   * @param label name of dataset to be created.
   */
  void write_to_hdf5(h5e::File &out, std::string label);
  /**
   * @brief Returns the size of the matrix function (size1 * size2).
   * @return Element size.
   */
  int element_size(void) { return element_size_; }

  /**
   * @brief Returns the number of rows in each matrix.
   * @return Number of rows.
   */
  int size1(void) { return size1_; }

  /**
   * @brief Returns the number of columns in each matrix.
   * @return Number of columns.
   */
  int size2(void) { return size2_; }

  /**
   * @brief Returns the number of time slices.
   * @return Number of time slices.
   */
  int nt(void) { return nt_; }

  /**
   * @brief Returns a pointer to the matrix data for time slice t.
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @return Pointer to matrix data for time slice t.
   */
  inline cplx *ptr(int t) { return data_.data() + (t+1) * element_size_; }

  /**
   * @brief Returns the value of the function at time t and matrix index i,j
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @param i row index of matrix 
   * @param j col index of matrix
   * @return value of function 
   */
  inline cplx value(int t, int i, int j) { return data_[(t+1) * element_size_ + i*size1_ +j]; }

  /**
   * @brief Parenthesis operator for mutable element access.
   *
   * Returns a reference to the complex matrix element at time slice `t` and
   * matrix indices `(i,j)`. The storage location is equivalent to
   * `ptr(t)[i*size1_ + j]`.
   *
   * @param t Time slice index. A value of -1 corresponds to the initial value on the Matsubara axis.
   * @param i Matrix row index.
   * @param j Matrix column index.
   * @return Reference to the complex value at the specified indices.
   */
  cplx &operator()(int t, int i, int j);

  /**
   * @brief Returns an Eigen Map of the function at time t
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @return Map of matrix valued function at time t
   */
  inline ZMatrixMap get_map(int t) { return ZMatrixMap(data_.data() + (t+1) * element_size_, size1_, size2_); }

  /**
   * @brief Sets the value of the matrix at time slice t.
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @param M Matrix to set.
   */
  void set_value(int t, ZMatrix &M);

  /**
   * @brief Sets the value of the element at time slice t and matrix index ij
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @param i Matrix row index
   * @param j Matrix Col index
   * @param v value to set.
   */
  void set_value(int t, int i, int j, cplx M);


  /**
   * @brief Gets the value of the matrix at time slice t.
   * @param t Time slice index. a value of -1 corresponds to the initial value on the Matsubara axis.
   * @param M Matrix to fill.
   */
  void get_value(int t, ZMatrix &M);

  /**
   * @brief Sets all matrix data to zero.
   */
  void set_zero(void);

  /**
   * @brief Sets all matrix data to a constant value.
   * @param M Matrix value to set.
   */
  void set_constant(ZMatrix &M);

  /**
   * @brief Sets all matrix data to a constant identity value.
   * @param M value to set.
   */
  void set_constant(cplx A);

  /**
   * @brief Sets all matrix data to a constant identity value.
   * @param M value to set.
   */
  void set_constant(double A);
};

}
#endif
