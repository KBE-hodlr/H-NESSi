
/**
 * @file blocks_keldysh.hpp
 * @brief Declares classes for storing and manipulating two-time contour objects using HODLR compression in the Keldysh formalism.
 *
 * This file provides classes for compressed storage and access of two-time Green's functions and related objects on the Keldysh contour, using hierarchical matrix algorithms.
 */
#pragma once

#include "block.hpp"
#include "utils.hpp"


/**
 * @namespace h_nessi
 * @brief Namespace for hierarchical matrix algorithms and quantum many-body data structures.
 */
namespace h_nessi {


/**
 * @class blocks_list
 * @brief Stores arrays of block objects and directly stored entries for two-time Keldysh contour objects.
 *
 * The blocks_list class manages a set of block matrices, each compressed using SVD, for efficient storage and manipulation of two-time Green's functions and related objects.
 */
class blocks_list{
  public: 

    /**
     * @brief Default constructor. Initializes an empty blocks_list object.
     */
    blocks_list();

    /**
     * @brief Constructs a blocks_list object with specified parameters.
     * @param nbox Number of block boxes.
     * @param nrows Vector of row dimensions for each block.
     * @param ncols Vector of column dimensions for each block.
     * @param svdtol SVD truncation tolerance.
     * @param size1 Column dimension in orbital space.
     * @param size2 Row dimension in orbital space.
     */
    blocks_list(int nbox,std::vector<int> &nrows,std::vector<int> &ncols,double svdtol,int size1,int size2);



    /** @brief Returns the number of block boxes. */
    int nbox(void) const {return nbox_;};
    /** @brief Returns the column dimension in orbital space. */
    int size1(void) const {return size1_;};
    /** @brief Returns the row dimension in orbital space. */
    int size2(void) const {return size2_;};
    /** @brief Returns block objects. */
    std::vector<std::vector<std::vector<block>>> &blocks() { return blocks_; }


    /**
     * @brief Retrieves the value of the compressed block and stores it in matrix M.
     * @param i Block row index. - this is a real-time argument but it is measured from the edge of the block
     * @param j Block column index. - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output matrix.
     */
    void get_compress(int i,int j, int b, ZMatrix &M);
    /**
     * @brief Retrieves the value of the compressed block and stores it in array M.
     * @param i Block row index. - this is a real-time argument but it is measured from the edge of the block
     * @param j Block column index. - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output array.
     */
    void get_compress(int i,int j, int b, cplx *M);


  private:
    std::vector<std::vector<std::vector<block>>> blocks_;/**< Set of blocks stored in compressed form. Order: Nbox*size2_*size1_ */
    int nbox_;                /**< Number of block boxes. */
    int size1_;               /**< Column dimension in orbital space. */
    int size2_;               /**< Row dimension in orbital space. */
};

/**
 * @class ret_blocks
 * @brief Stores two-time retarded contour objects compressed with HODLR structure and hermitian symmetry.
 *
 * The ret_blocks class manages compressed storage and direct access for retarded Green's functions and related objects on the Keldysh contour. Stores both compressed blocks and directly stored triangular regions at the diagonal.
 */
class ret_blocks{
  public:

    /** @brief Default constructor. Initializes an empty ret_blocks object. */
    ret_blocks();
    /**
     * @brief Constructs a ret_blocks object with specified parameters.
     * @param nbox Number of block boxes.
     * @param ndir Number of directly stored values.
     * @param nrows Vector of row dimensions for each block.
     * @param ncols Vector of column dimensions for each block.
     * @param svdtol SVD truncation tolerance.
     * @param size1 Column dimension in orbital space.
     * @param size2 Row dimension in orbital space.
     */
    ret_blocks(int nbox,int ndir,std::vector<int> &nrows,std::vector<int> &ncols,double svdtol,int size1,int size2);

    /** @brief Returns the number of directly stored values. */
    int ndir(void) const {return ndir_;};

    /**
     * @brief Retrieves the value of the compressed block and stores it in matrix M.
     * @param i Block row index.  - this is a real-time argument but it is measured from the edge of the block
     * @param j Block column index.  - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output matrix.
     */
    void get_compress(int i,int j, int b,ZMatrix &M);
    /**
     * @brief Retrieves the value of the compressed block and stores it in array M.
     * @param i Block row index.  - this is a real-time argument but it is measured from the edge of the block
     * @param j Block column index.  - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output array.
     */
    void get_compress(int i,int j, int b,cplx *M);
    /** @brief Returns reference to the underlying blocks_list data. */
    blocks_list &data(void) {return data_;};
    /** @brief Returns pointer to directly stored column data. - t is the continuous time index (not t')*/
    std::complex<double> * dirtricol(void) {return dirtricol_.data();};

    /**
     * @brief Sets the directly stored column data for index i using matrix M.
     * @param i Column index.
     * @param M Input matrix data.
     */
    void set_direct_col(int i, ZMatrix &M);
    /**
     * @brief Retrieves the directly stored column data for index i into matrix M.
     * @param i Column index.
     * @param M Output matrix data.
     */
    void get_direct_col(int i, ZMatrix &M);
    /**
     * @brief Retrieves the directly stored column data for index i into array M.
     * @param i Column index.
     * @param M Output array data.
     */
    void get_direct_col(int i, cplx *M);


  private:
    blocks_list data_;              /**< Set of blocks for compressed storage. */
    std::vector<std::complex<double>> dirtricol_;  /**< Directly stored column data. */
    int ndir_;                         /**< Number of directly stored values. */
};

/**
 * @class les_blocks
 * @brief Stores two-time lesser contour objects compressed with HODLR structure and hermitian symmetry.
 *
 * The les_blocks class manages compressed storage and access for lesser Green's functions and related objects on the Keldysh contour.
 */
class les_blocks{
  public:

    /** @brief Default constructor. Initializes an empty les_blocks object. */
    les_blocks();
    /**
     * @brief Constructs a les_blocks object with specified parameters.
     * @param nbox Number of block boxes.
     * @param nrows Vector of row dimensions for each block.
     * @param ncols Vector of column dimensions for each block.
     * @param svdtol SVD truncation tolerance.
     * @param size1 Column dimension in orbital space.
     * @param size2 Row dimension in orbital space.
     */
    les_blocks(int nbox,std::vector<int> &nrows,std::vector<int> &ncols,double svdtol,int size1,int size2);

    /**
     * @brief Retrieves the value of the compressed block and stores it in matrix M.
     * @param t1 First time index.  - this is a real-time argument but it is measured from the edge of the block
     * @param t2 Second time index.  - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output matrix.
     */
    void get_compress(int t1,int t2,int b,ZMatrix &M);
    /**
     * @brief Retrieves the value of the compressed block and stores it in array M.
     * @param t1 First time index.  - this is a real-time argument but it is measured from the edge of the block
     * @param t2 Second time index.  - this is a real-time argument but it is measured from the edge of the block
     * @param b Block box index.
     * @param M Output array.
     */
    void get_compress(int t1,int t2,int b, cplx *M);
    /** @brief Returns reference to the underlying blocks_list data. */
    blocks_list &data(void) {return data_;};


  private:
    blocks_list data_; /**< Set of blocks for compressed storage. */
};

/**
 * @class tv_blocks
 * @brief Stores two-time mixed contour objects compressed with HODLR structure and hermitian symmetry.
 *
 * The tv_blocks class manages compressed storage and access for mixed contour Green's functions and related objects on the Keldysh contour.
 */
class tv_blocks{
  public:

    /** @brief Default constructor. Initializes an empty tv_blocks object. */
    tv_blocks();
    /**
     * @brief Constructs a tv_blocks object with specified parameters.
     * @param size1 Column dimension in orbital space.
     * @param size2 Row dimension in orbital space.
     * @param nt Number of time slices.
     * @param ntau Number of tau slices.
     * @param svdtol SVD truncation tolerance.
     */
    tv_blocks(int size1,int size2,int nt,int ntau,double svdtol);

    /** @brief Returns pointer to the data array. */
    cplx* data(void) { return data_.data(); };
    /** @brief Returns pointer to the transposed data array. */
    cplx* data_trans(void) { return data_trans_.data(); };

    /** @brief Returns reference to the data vector. */
    std::vector<cplx> &data_vector(void) { return data_; }
    /** @brief Returns reference to the transposed data vector. */
    std::vector<cplx> &data_trans_vector(void) { return data_trans_; }

  private:
    std::vector<cplx> data_;       /**< Data array for mixed contour objects. - stored in (t \tau i j) order*/
    std::vector<cplx> data_trans_; /**< Transposed data array - same data as data_ but stored in (t \tau j i) order*/
};


}  // namespace
