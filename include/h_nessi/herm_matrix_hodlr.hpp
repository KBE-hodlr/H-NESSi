#pragma once

#include "block.hpp"
#include "blocks_keldysh.hpp"
#include <hdf5.h>
#include <chrono>
#include <fstream>
#include "integration.hpp"
#include "dlr.hpp"

#ifdef INCLUDE_NESSI
  #include "cntr/cntr_herm_matrix_decl.hpp"
#endif


extern "C"
{
  #include "dlr_c/dlr_c.h"
}

namespace h_nessi {

/**
 * @file herm_matrix_hodlr.hpp
 * @brief HODLR-backed storage for Hermitian two-time contour objects C(t,t').
 *
 * This header declares the class `herm_matrix_hodlr` which stores bosonic
 * (sig = +1) or fermionic (sig = -1) two-time contour functions C(t,t') in a
 * Hierarchical Off-Diagonal Low-Rank (HODLR) representation with additional
 * symmetry and direct-storage optimizations for near-diagonal blocks.
 *
 * The class supports matrix-valued contour functions (square matrices fully
 * supported). It exposes getters and setters for the different contour
 * components (retarded/lesser/tv/matsubara), I/O helpers for HDF5, and
 * utilities used by the time-stepping Dyson solver (SVD updates, direct
 * storage accessors, and convolution tensor initialization).
 */

/**
 * @class herm_matrix_hodlr
 * @brief HODLR container for Hermitian two-time Green's functions C(t,t').
 *
 * Purpose
 * -------
 * The `herm_matrix_hodlr` class encapsulates the geometry and numerical data
 * required to represent two-time contour objects using a HODLR layout. The
 * representation stores:
 * - Hierarchical low-rank blocks for off-diagonal regions (SVD compressed).
 * - Directly stored blocks near the diagonal for accuracy (triangular direct regions).
 * - Special storage for the last `k` timesteps used by the time-stepping
 *   algorithms (current-timestep buffers for retarded and lesser components).
 *
 * Key responsibilities
 * - Provide read/write accessors for the various contour components (ret/les/tv)
 * - Support SVD updates (incremental/thresholded) used during time stepping
 * - Provide geometry bookkeeping utilities (block indexing / direct storage)
 * - Support HDF5 checkpointing and small helper utilities used by Dyson
 *   time-stepping routines.
 */
class herm_matrix_hodlr{
  public:

    /* construction, destruction */
  /** Default constructor. Leaves the object in an empty/uninitialized state. */
  herm_matrix_hodlr();

  /**
   * Construct and initialize a HODLR hermitian matrix container.
   * @param nt Maximum number of time steps
   * @param r Number of DLR grid points on Matsubara axis
   * @param nlvl Number of hierarchical levels in the HODLR partitioning
   * @param svdtol Relative tolerance used for SVD truncation
   * @param size1 Column dimension of the contained orbital matrices
   * @param size2 Row dimension of the contained orbital matrices
   * @param sig +1 for bosonic, -1 for fermionic contour objects
   * @param k Integration order (number of previous timesteps retained)
   */
  herm_matrix_hodlr(int nt,int r,int nlvl,double svdtol,int size1,int size2,int sig,int k);

  /** Copy-assignment (performs a deep copy of internal buffers). */
  herm_matrix_hodlr &operator=(herm_matrix_hodlr &g);

  /**
   * Construct from an HDF5 checkpoint file. Reads metadata and data stored
   * under `label` in the provided `h5e::File`.
   */
  herm_matrix_hodlr(h5e::File &in, std::string label);

  // geometry helpers
  /** Compute and store information about block levels and row/column indices of directly stored regions.
   * @return maxdir - maximum number of directly-stored entries in a row
   */
  int getbkl_indices(void);

  /** Initialize internal geometry arrays for a given number of timesteps and hierarchical levels.
   * @param nt Number of time steps
   * @param nlvl Number of hierarchical levels
   */
  void init_shape(int nt,int nlvl);

  /** Map a time pair (t1,t2) into an index for directly stored column layout.
   * @param t1 first time argument
   * @param t2 second time argument
   */
  int time2direct_col(int t1,int t2);

  /** Map a time pair (t1,t2) into a HODLR block index.
   * @param t1 first time argument
   * @param t2 second time argument
   */
  int time2block(int t1,int t2);

  int nlvl(void) const {return nlvl_;};
  int nt(void) const {return nt_;};
  int ntau(void) const {return ntau_;};
  int r(void) const {return r_;};
  int size(void) const {return size1_;};
  int size1(void) const {return size1_;};
  int size2(void) const {return size2_;};
  int sig(void) const {return sig_;};
  int nbox(void) const {return nbox_;};
  int ndir(void) const {return ndir_;};
  int tstpmk(void) const {return tstpmk_;};
  int tstp(void) const {return tstpmk_ + k_;};
  int maxdir(void) const {return maxdir_;};
  void set_tstpmk(int tstpmk) {tstpmk_=tstpmk;};
  int k(void) const {return k_;};
  bool &can_extrap(void) {return can_extrap_;};


  ret_blocks &ret(void) {return ret_;};
  les_blocks &les(void)  {return les_;};
  tv_blocks &tv(void)  {return tv_;};
  double *mat(void) {return mat_.data();};
  double *GMConvTens(void) {return GMConvTensMatrix_.data();};
  DMatrix &GMConvTensMatrix(void) {return GMConvTensMatrix_;};
    
  /** Modified Gram-Schmidt used by low-rank updates. Returns 
   * pre-normalization norm of vector x
   * @param V Matrix to orthogonalize against (columns are basis vectors)
   * @param b Vector to orthogonalize
   * @param x  output normalized vector
   * @param Vb output vector
   */
  double mgs(const ZMatrix &V,const ZRowVector &b,const ZRowVector &x_,const ZRowVector &Vb_);

  /**
   * Update an existing block `B` by adding the information contained in
   * `row` (complex row vector). `row_cur` is the current row index in the
   * global time/orbital layout. This routine may trigger an SVD update and
   * rank truncation based on the instance's svdtol_. If `print` is true,
   * the routine outputs diagnostic information.
   */
  void updatetsvd(block &B,const ZRowVector &row,int row_cur, bool print = false);

  /** Update all blocks that depend on the current state. The Integrator
   * `I` provides quadrature/weight tables potentially needed during
   * updates (e.g., when building direct-to-lowrank interfaces). */
  void update_blocks(Integration::Integrator &I);

  // Geometry accessors
  /** Number of blocks that have been built/initialized. */
  int built_blocks(void) const {return built_blocks_;};
  /** first row index of block blk. */
  int blkr1(int blk) {if(blk==-1) return 0; else return blkr1_[blk];};
  /** last row index of block blk. */
  int blkr2(int blk) {return blkr2_[blk];};
  /** first column index of block blk. */
  int blkc1(int blk) {return blkc1_[blk];};
  /** last column index of block blk. */
  int blkc2(int blk) {return blkc2_[blk];};
  /** last column index of block blk. */
  int blkdirheight(int blk)  {return blkdirheight_[blk];};
  /** column index of first directly stored element in row tstp */
  int c1_dir(int tstp) {return c1_dir_[tstp];};
  /** row index of last directly stored element in column tstp */
  int r2_dir(int tstp) {return r2_dir_[tstp];};
  /** number of directly stored elements in each row */
  int ntri(int tstp) {return ntri_[tstp];};
  /** the current timestep minus k (integration order) */
  int tstpmk(void) {return tstpmk_;};
  /** number of built rows in each blocks */
  int blklen(int blkindex) {return blklen_[blkindex];};
  /** number of directly stored lesser component elements */
  int len_les_dir_square() { return len_les_dir_square_; }
  /** index of triangluar direct region at row t */
  int t_to_dirlvl(int t) { return t_to_dirlvl_[t]; }
  /** first index in les_dir_square_ for level lvl - data is stored as (i,j,t1,t2), row-major for time indices*/
  int les_dir_square_first_index(int lvl) { return les_dir_square_first_index_[lvl]; }

    
  // Accessors (Get)
  /**
   * Evaluate the Matsubara/mixed component at DLR time `tau` and write the
   * real-valued matrix into `M`.
   */
  void get_mat_tau(double tau,                     dlr_info &dlr, DMatrix &M);
  /** Variant writing into a raw double buffer (column-major) sized
   * size1_*size2_. */
  void get_mat_tau(double tau,                     dlr_info &dlr, double *M);
  /** Variant writing into a complex Eigen matrix. */
  void get_mat_tau(double tau,                     dlr_info &dlr, ZMatrix &M);
  /** Variant writing into a raw complex buffer. */
  void get_mat_tau(double tau,                     dlr_info &dlr, cplx *M);
  /** Evaluate tv component for timestep `tstp` at DLR time `tau`. */
  void get_tv_tau(int tstp, double tau,            dlr_info &dlr, ZMatrix &M);
  /** Variant writing into a complex Eigen matrix. */
  void get_tv_tau(int tstp, double tau,            dlr_info &dlr, cplx *M);
  /** Evaluate Matsubara/mixed component at a vector of DLR times. */
  void get_mat_tau_array(DColVector taus,          dlr_info &dlr, DMatrix &M);
  void get_mat_tau_array(DColVector taus,          dlr_info &dlr, double *M);
  void get_mat_tau_array(DColVector taus,          dlr_info &dlr, ZMatrix &M);
  void get_mat_tau_array(DColVector taus,          dlr_info &dlr, cplx *M);
  void get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, ZMatrix &M);
  void get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, cplx *M);
  /** Retrieve stored matsubara matrix at dlr index `i`. */
  void get_mat(int i,DMatrix &M);
  DMatrixMap map_mat(int i);
  /** Get matsubara component evaluated at points \beta-dlr(i) for i=0...r.  
   * dest is assumed to be a column-major double buffer of size r_*size1_*size2_.
   */
  void get_mat_reversed(dlr_info &dlr, double *dest);
  void get_mat_reversed(dlr_info &dlr, std::complex<double> *dest);
  void get_mat_reversed(dlr_info &dlr, DMatrix &dest);
  /** Retrieve the tv component for tstp t1 and dlr index t2 into `M`. */
  void get_tv(int t1,int t2,ZMatrix &M);
  void get_tv(int t1,int t2,cplx *M);
  ZMatrixMap map_tv(int t1,int t2);
  /** Retrieve the transposed tv component for tstp t1 and dlr index t2 into `M`. */
  void get_tv_trans(int t1,int t2,ZMatrix &M);
  void get_tv_trans(int t1,int t2,cplx *M);
  ZMatrixMap map_tv_trans(int t1,int t2);
  /** Get mixed component evaluated at points \beta-dlr(i) for i=0...r.  
   * dest is assumed to be a column-major double buffer of size r_*size1_*size2_.
   */
  void get_tv_reversed(int tstp, dlr_info &dlr, cplx *dest);
  void get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &dest);
  /** Retrieve vt (mixed component with switched contour arguments) for a given timestep. 
   * dest is assumed to be a column-major complex buffer of size r_*size1_*size2_.
  */
  void get_vt(int tstp, dlr_info &dlr, cplx *dest);
  void get_vt(int tstp, dlr_info &dlr, ZMatrix &dest);
  /** Retrieve the retarded component for times (t1,t2). */
  void get_ret(int t1,int t2,ZMatrix &M);
  void get_ret(int t1,int t2,cplx *M);
  /** Retrieve the lesser component for times (t1,t2). */
  void get_les(int t1,int t2,ZMatrix &M);
  void get_les(int t1,int t2,cplx *M);
  /** Retrieve retarded values from the current timestep circular buffer. */
  void get_ret_curr(int t,int tp,ZMatrix &M);
  void get_ret_curr(int t,int tp,cplx *M);
  ZMatrixMap map_ret_curr(int t,int tp);
  /** Retrieve retarded values from the 'correction' storage region
   * information directly below the direct region. stored in ret_corr_below_tri.
   */
  void get_ret_corr(int t,int tp,ZMatrix &M);
  void get_ret_corr(int t,int tp,cplx *M);
  /** Retrieve lesser values from the current timestep circular buffer. */
  void get_les_curr(int t,int tp,ZMatrix &M);
  void get_les_curr(int t,int tp,cplx *M);
  ZMatrixMap map_les_curr(int t,int tp);
  /** Compute or extract the density matrix for timestep `tstp`.
   * The result is written into `M` or into the raw buffer `res`. */
  void density_matrix(int tstp, dlr_info &dlr, DMatrix &M);
  void density_matrix(int tstp, dlr_info &dlr, ZMatrix &M);

  std::vector<cplx> &les_dir_square() { return les_dir_square_; }
  std::vector<cplx> &les_left_edge() { return les_left_edge_; }
  std::vector<cplx> &ret_left_edge() { return ret_left_edge_; } 

  // Mutators (Set)
  /** Store matsubara matrix at dlr index i from an Eigen matrix. */
  void set_mat(int i, DMatrix &M);
  /** Store matsubara matrix at dlr index i from a raw double buffer. */
  void set_mat(int i, double *M);
  /** Store tv component (t1,t2) from an Eigen complex matrix. */
  void set_tv(int t1, int t2, ZMatrix &M);
  /** Store tv component (t1,t2) from a raw complex buffer. */
  void set_tv(int t1, int t2, cplx* M);
  /** Store retarded current-timestep buffer entry (t1,t2). */
  void set_ret_curr(int t1, int t2, ZMatrix &M);
  void set_ret_curr(int t1, int t2, cplx* M);
  /** Store lesser current-timestep buffer entry (t1,t2). */
  void set_les_curr(int t1, int t2, ZMatrix &M);
  void set_les_curr(int t1, int t2, cplx* M);

  // Get/set for current timesteps
  /** Copy the current timestep `tstp` data into `G`. */
  void get_timestep_curr(int tstp,herm_matrix_hodlr &G);
  /** Copy the current timestep (latest) into `G`. */
  void get_timestep_curr(herm_matrix_hodlr &G);
  /** Replace internal current-timestep buffer for `tstp` with data from `G`. */
  void set_timestep_curr(int tstp,herm_matrix_hodlr &G);
  /** Replace the latest current-timestep buffer with data from `G`. */
  void set_timestep_curr(herm_matrix_hodlr &G);

  /** Raw pointer to matsubara matrix `i` storage (double buffer). */
  double *matptr(int i);

  /** Pointer to tv buffer for timestep `t` and DLR index `tau`. */
  cplx *tvptr(int t, int tau);

  /** Pointer to transposed tv buffer for timestep `t` and DLR index `tau`. */
  cplx *tvptr_trans(int t, int tau);

  /** Pointer to current les timestep buffer */
  cplx *curr_timestep_les_ptr(int t, int tp);

  /** Pointer to current ret timestep buffer */
  cplx *curr_timestep_ret_ptr(int t, int tp);

  /** Compute internal linear offset for tv storage. */
  int tv_offset(int t, int tau) {assert(t <= nt_ && tau < r_); return t*r_*size1_*size2_ + tau*size1_*size2_;}

  /** Pointer to directly stored retarded data in column-major order. */
  std::complex<double>* retptr_col(int t, int t1); // Directly stored data in col order

  /** Pointer to directly stored retarded correlated region (up to k+1 below diagonal). */
  std::complex<double>* retptr_corr(int t, int t1); // Directly stored data up to k+1 below diagonal

  // Algebraic operations on timesteps
  /** Increment the current timestep buffer for `tstp` by `alpha * G`. */
  void incr_timestep_curr(int tstp,herm_matrix_hodlr &G,cplx alpha);

  /** Scale stored current-timestep data for `tstp` by `alpha`. */
  void smul_curr(int tstp,cplx alpha);

  /** Set the circular buffer corresponding to timestep `tstp` to zero. */
  void set_tstp_zero(int tstp);

  /** Zero all stored matsubara matrices. */
  void set_mat_zero(void);

  /** Zero tv entries for timestep `tstp`. */
  void set_tv_tstp_zero(int tstp);

  /** Zero retarded entries for timestep `tstp`. */
  void set_ret_tstp_zero(int tstp);

  /** Zero lesser entries for timestep `tstp`. */
  void set_les_tstp_zero(int tstp);

  // Mixed (Matsubara/real-time) convolution helpers
  /** Initialize the mixed-component convolution tensor using DLR info in `dlr`.
   * This allocates and fills the internal GMConvTensMatrix_. */
  void initGMConvTensor(dlr_info &dlr);

  // I/O and diagnostics
  /** Write a lightweight checkpoint containing the current runtime buffers. */
  void write_checkpoint_hdf5(h5e::File &out, std::string label);

  /** Write the full object to HDF5 under `label`. */
  void write_to_hdf5(h5e::File &out, std::string label, std::string geometry_label = "");

  /** Write the density matrix (rho) to HDF5 using `dlr` for Matsubara transforms. */
  void write_rho_to_hdf5(h5e::File &out, std::string label, dlr_info &dlr);

  /** Write per-block ranks to HDF5 for diagnostics. */
  void write_rank_to_hdf5(h5e::File &out, std::string label);

  /** Write matsubara matrices to HDF5. */
  void write_mat_to_hdf5(h5e::File &out, std::string label);

  /** Write the current circular buffers to HDF5. */
  void write_curr_to_hdf5(h5e::File &out, std::string label);

  /** Write GL0 (left boundary) helper data to HDF5. */
  void write_GL0_to_hdf5(h5e::File &out, std::string label);

  /** Write GR0 (right boundary) helper data to HDF5. */
  void write_GR0_to_hdf5(h5e::File &out, std::string label);

  /** Print a small report on memory usage (compressed and full) to stdout. 
  *   Returns total memory usage in GB.
  */
  double get_memory_usage(bool print_memory_usage=true);


  private:
  DMatrix timing;
// ######################### GEOMETRY ################################
    /// @private
    /** \brief <b> index of directly stored triangle at row tstp </b> */
    std::vector<int> t_to_dirlvl_; 
    /// @private
    /** \brief <b> First rows in blocks. [Jason blki1] </b> */
    std::vector<int> blkr1_; 
    /// @private
    /** \brief <b> Last rows in blocks [Jason blki2] </b> */
    std::vector<int> blkr2_; 
    /// @private
    /** \brief <b> First columns in blocks [Jason blkj1] </b> */
    std::vector<int> blkc1_; 
    /// @private
    /** \brief <b> Last columns in blocks [Jason blkj2] </b> */
    std::vector<int> blkc2_; 
    /// @private
    /** \brief <b> Levels of blocks </b> */
    std::vector<int> blklevel_; 
    /// @private
    /** \brief <b> Height of directly stored values at the of the block </b> */
    std::vector<int> blkdirheight_; 
    /// @private
    /** \brief <b> Levels of blocks </b> */
    std::vector<int> blkndirstart_;
    // /// @private
    // /** \brief <b> Column index of first directly-stored entry in each row. Jason's j1dir  </b> */
    std::vector<int> c1_dir_;
    // /// @private
    // /** \brief <b> Row index of last directly-stored entry in each column   </b> */
    std::vector<int> r2_dir_;


    // /// @private
    // /** \brief <b>  Number of directly stored indices in row  </b> */
    std::vector<int> ntri_;
    // /// @private
    // /** \brief <b>  rows in each block  </b> */
    std::vector<int> blklen_;
    // /// @private
    // /** \brief <b>  rows in each block  </b> */
    std::vector<int> les_dir_square_first_index_;
    int len_les_dir_square_;
    std::vector<int> ret_corr_index_t_;
// ######################### GEOMETRY ################################

// #########################   DATA   ################################
    /// @private
    /** \brief <b> Retarded blocks  </b> */
    ret_blocks ret_;
    /// @private
    /** \brief <b> Lesser blocks   </b> */
    les_blocks les_;
    /// @private
    /** \brief <b> tv blocks   </b> */
    tv_blocks tv_;
    /// @private
    /** \brief <b> mat blocks   </b> */
    std::vector<double> mat_;

    /// @private
    /** \brief <b> Convolution tensor for mixed component </b> */
    DMatrix GMConvTensMatrix_;

    /** \brief <b> ret directly stored slice of the last k timesteps   </b> */
    std::vector<cplx> curr_timestep_ret_;
    /** \brief <b> les directly stored slice of the last k timesteps   </b> */
    std::vector<cplx> curr_timestep_les_;

    std::vector<cplx> ret_corr_below_tri_;
    /// @private
    /** \brief <b> lesser component in squares along diagonal </b> */
    std::vector<cplx> les_dir_square_;
    /// @private
    /** \brief <b> Stores les gf along left edge <\b> */
    std::vector<cplx> les_left_edge_;
    std::vector<cplx> ret_left_edge_;
// #########################   DATA   ################################


//  ################### SCALARS #####################################
    /** \brief <b> integration order </b> */
    int k_;
    /** \brief <b> counts the number of blocks that have been initialized </b> */
    int built_blocks_;
    /** \brief <b> current timestepk</b> */
    // int tstp_;
    /** \brief <b> current timestep minus k</b> */
    int tstpmk_;
    /// @private
    /** \brief <b> Maximum number of directly-stored entries in a row.</b> */
    int maxdir_;
    /// @private
    /** \brief <b> Number of levels in partitioning.</b> */
    int nlvl_; 
    /// @private
    /** \brief <b> Number of boxes.</b> */
    int nbox_; 
    // /// @private
    /** \brief <b> SVD tolerance.</b> */
    double svdtol_; 
    /// @private
    /** \brief <b> Maximum number of time steps.</b> */
    int nt_; 
    /// @private
    /** \brief <b> Number of initially proposed time points on the Matsubara axis.</b> */
    int ntau_;
    /// @private
    /** \brief <b> Number of the DLR time grids on the Matsubara axis.</b> */
    int r_;
    // /// @private
    // /** \brief <b> Number of directly stored enties.</b> */
    int ndir_;
    /// @private
    /** \brief <b> Compressed memory usage ret</b> */
    double cr_;
    /// @private
    /** \brief <b> full memory usage ret</b> */
    double fr_;
    /// @private
    /** \brief <b> Compressed memory usage les</b> */
    double cl_;
    /// @private
    /** \brief <b> full memory usage les</b> */
    double fl_;
    /// @private
    /** \brief <b> Column dimension in the orbital space.</b> */
    int size1_; 
    /// @private
    /** \brief <b> Row dimension in the orbital space.</b> */
    int size2_; 
    /// @private
    /** \brief <b> Bose = +1, Fermi =-1. </b> */
    int sig_; // Bose = +1, Fermi =-1
    /// @private
    /** \brief <b> whether or not the object can be extrapolated.  Extrapolation overwrites the last direct slices, so it can only be called immediately after block update </b> */
    bool can_extrap_;
};

double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_mat(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);

#ifdef INCLUDE_NESSI
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr);
double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2);
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix &g2, dlr_info &dlr);
#endif


}  // namespace
