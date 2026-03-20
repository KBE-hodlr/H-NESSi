
/**
 * @file block.hpp
 * @brief Defines the block class for low-rank matrix storage using SVD in the HODLR library.
 */
#pragma once

#include "utils.hpp"



/**
 * @namespace h_nessi
 * @brief Namespace for hierarchical matrix algorithms and data structures.
 */
namespace h_nessi {

/**
 * @class block
 * @brief Stores a single block of a matrix in compressed SVD form for HODLR algorithms.
 *
 * The block class represents a matrix block compressed via singular value decomposition (SVD):
 * \f$ A = U S V \f$.
 * - U: Left singular vectors (rows x epsrank)
 * - S: Singular values (epsrank)
 * - V: Right singular vectors (epsrank x cols)
 *
 * Provides constructors for direct initialization, SVD-based compression, and resizing.
 */
class block {
  public:
    

    /**
     * @brief Default constructor. Initializes a 1x1 block with rank 1 and zero SVD tolerance.
     */
    block();

    /**
     * @brief Constructs a block with specified dimensions, rank, and SVD tolerance.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param epsrank SVD rank (number of singular values/vectors).
     * @param svdtol SVD truncation tolerance.
     */
    block(int rows,int cols,int epsrank,double svdtol);

    /**
     * @brief Resize only the SVD rank of the block.
     * @param epsrank New SVD rank.
     */
    void set_rank(int epsrank);


    /** @brief Get the number of columns in the block. */
    int cols(void) const {return cols_;};
    /** @brief Get the number of rows in the block. */
    int rows(void) const {return rows_;};
    /** @brief Get the SVD rank of the block. */
    int epsrank(void) const {return epsrank_;};
    /** @brief Get the SVD truncation tolerance. */
    double svdtol(void) const {return svdtol_;};
    /** @brief Access the singular values vector. */
    DColVector &S(void) {return S_;};
    /** @brief Access the left singular vector matrix. */
    ZMatrix &U(void) {return U_;};
    /** @brief Access the right singular vector matrix. */
    ZMatrix &V(void) {return V_;};


    /** @brief Get a pointer to the singular values data. */
    double *Sdata(void) {return S_.data();};
    /** @brief Get a pointer to the left singular vector data. */
    cplx *Udata(void) {return U_.data();};
    /** @brief Get a pointer to the right singular vector data. */
    cplx *Vdata(void) {return V_.data();};


    /** @brief Set the singular values vector.
     * @param S New singular values.
     */
    void set_S(const DColVector &S) {S_=S;};
    /** @brief Set the left singular vector matrix.
     * @param U New U matrix.
     */
    void set_U(const ZMatrix &U) {U_=U;};
    /** @brief Set the right singular vector matrix.
     * @param V New V matrix.
     */
    void set_V(const ZMatrix &V) {V_=V;};
    /** @brief Set the SVD truncation tolerance.
     * @param svdtol New tolerance value.
     */
    void set_svdtol(double svdtol) {svdtol_=svdtol;};
    /** @brief Set the SVD rank.
     * @param epsrank New rank value.
     */
    void set_epsrank(double epsrank) {epsrank_=epsrank;};
    /** @brief Set the number of rows in the block.
     * @param rows New number of rows.
     */
    void set_rows(int rows) {rows_=rows;};
    /** @brief Set the number of columns in the block.
     * @param cols New number of columns.
     */
    void set_cols(int cols) {cols_=cols;};


  private:
    ZMatrix U_;      /**< Left singular vector matrix. */
    ZMatrix V_;      /**< Right singular vector matrix. */
    DColVector S_;   /**< Singular values vector. */
    double svdtol_;  /**< SVD truncation tolerance. */
    int epsrank_;    /**< SVD rank. */
    int rows_;       /**< Number of rows in the block. */
    int cols_;       /**< Number of columns in the block. */
};


}  // namespace
