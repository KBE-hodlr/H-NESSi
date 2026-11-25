#pragma once
#include <algorithm>
#include <cstdint>

#include "block.hpp"
#include "blocks_keldysh.hpp"
#include <hdf5.h>
#include <chrono>
#include <fstream>
#include "integration.hpp"
#include "dlr.hpp"
#include "checks.hpp"

#ifdef INCLUDE_NESSI
// NESSi interoperability (BUGS B8, WP-N). Only the class template is needed for the declarations below; code that
// calls them includes "cntr/cntr.hpp" itself. The INCLUDE_NESSI macro is defined by the CMake option of the same name.
namespace cntr {
template <typename T> class herm_matrix;
}
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

  /**
   * Copy constructor and copy assignment: deep copy of the complete state.
   * Every member is a value type (std::vector, Eigen matrix, ret_/les_/tv_blocks,
   * which hold only std::vector and Eigen members, and scalars); there are no raw
   * pointers or references, so the memberwise copy is an independent deep copy:
   * geometry, compressed blocks and their ranks, all direct stores, the current-
   * timestep window (tstpmk, circular buffers), mat, tv/tv_trans, GMConvTensMatrix,
   * timing and can_extrap. The copy can continue time stepping (update_blocks,
   * dyson_*) exactly like the original. Assignment replaces the target completely,
   * so the geometries of the two objects need not match beforehand.
   * No move operations are declared: an rvalue source is copied as well.
   */
  herm_matrix_hodlr(const herm_matrix_hodlr &g) = default;
  herm_matrix_hodlr &operator=(const herm_matrix_hodlr &g) = default;

  /**
   * Construct from an HDF5 checkpoint file. Reads metadata and data stored
   * under `label` in the provided `h5e::File`.
   */
  herm_matrix_hodlr(h5e::File &in, std::string label);

  /** Map a time pair (t1,t2) into an index for directly stored column layout.
   * Precondition (tier 1): 0 <= t1 < nt and c1_dir(t1) <= t2 <= t1 (the pair lies in the yellow direct triangle).
   * @param t1 first time argument
   * @param t2 second time argument
   */
  int time2direct_col(int t1,int t2);

  /** Map a time pair (t1,t2) into a HODLR block index.
   * Precondition: 0 <= t1 < nt and 0 <= t2 < c1_dir(t1) (tier 1); a pair outside every compressed block throws
   * h_nessi::precondition_error (always on: it is only evaluated on the failure path).
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
  /** Set the window start tstpmk (bootstrap reset to 0, restart). Tier 2: 0 <= tstpmk <= nt (HM-C14). */
  void set_tstpmk(int tstpmk) {
    HNESSI_REQUIRE(0 <= tstpmk && tstpmk <= nt_, "herm_matrix_hodlr::set_tstpmk", HNESSI_KV(tstpmk), HNESSI_KV(nt_));
    tstpmk_=tstpmk;
  };
  int k(void) const {return k_;};
  /** true between update_blocks and the first write into row tstp() (extrapolate, solver calls and the setters of
   * row tstp() clear it; ST-6). */
  bool &can_extrap(void) {return can_extrap_;};


  ret_blocks &ret(void) {return ret_;};
  les_blocks &les(void)  {return les_;};
  tv_blocks &tv(void)  {return tv_;};
  double *mat(void) {return mat_.data();};
  double *GMConvTens(void) {return GMConvTensMatrix_.data();};
  DMatrix &GMConvTensMatrix(void) {return GMConvTensMatrix_;};
    
  /** Update all blocks that depend on the current state. The Integrator
   * `I` provides quadrature/weight tables potentially needed during
   * updates (e.g., when building direct-to-lowrank interfaces).
   * Tier 2: I.k() == k() (HM-U2, precondition_error) and 0 <= tstpmk() < nt() (HM-U1, state_error: at most
   * nt-k-1 time steps plus the final k+1 flush calls). */
  void update_blocks(Integration::Integrator &I);

  // Geometry accessors
  /** Number of blocks that have been built/initialized. */
  int built_blocks(void) const {return built_blocks_;};
  // Index preconditions of the geometry getters are tier 1 (HNESSI_ASSERT, HM-C15..C18).
  /** first row index of block blk (-1 <= blk < nbox; blk == -1 gives 0). */
  int blkr1(int blk) {
    HNESSI_ASSERT(-1 <= blk && blk < nbox_, "herm_matrix_hodlr::blkr1", HNESSI_KV(blk), HNESSI_KV(nbox_));
    if(blk==-1) return 0; else return blkr1_[blk];
  };
  /** last row index of block blk. */
  int blkr2(int blk) {check_blk_("herm_matrix_hodlr::blkr2", blk); return blkr2_[blk];};
  /** first column index of block blk. */
  int blkc1(int blk) {check_blk_("herm_matrix_hodlr::blkc1", blk); return blkc1_[blk];};
  /** last column index of block blk. */
  int blkc2(int blk) {check_blk_("herm_matrix_hodlr::blkc2", blk); return blkc2_[blk];};
  /** height of the direct square at the top of block blk (0 <= blk <= nbox; blk == nbox is the last square). */
  int blkdirheight(int blk)  {
    HNESSI_ASSERT(0 <= blk && blk <= nbox_, "herm_matrix_hodlr::blkdirheight", HNESSI_KV(blk), HNESSI_KV(nbox_));
    return blkdirheight_[blk];
  };
  /** column index of first directly stored element in row tstp */
  int c1_dir(int tstp) {check_t_("herm_matrix_hodlr::c1_dir", tstp); return c1_dir_[tstp];};
  /** row index of last directly stored element in column tstp */
  int r2_dir(int tstp) {check_t_("herm_matrix_hodlr::r2_dir", tstp); return r2_dir_[tstp];};
  /** number of directly stored elements in each row */
  int ntri(int tstp) {check_t_("herm_matrix_hodlr::ntri", tstp); return ntri_[tstp];};
  /** the current timestep minus k (integration order) */
  int tstpmk(void) {return tstpmk_;};
  /** number of built rows in each blocks */
  int blklen(int blkindex) {check_blk_("herm_matrix_hodlr::blklen", blkindex); return blklen_[blkindex];};
  /** number of directly stored lesser component elements */
  int len_les_dir_square() { return len_les_dir_square_; }
  /** index of triangluar direct region at row t */
  int t_to_dirlvl(int t) { check_t_("herm_matrix_hodlr::t_to_dirlvl", t); return t_to_dirlvl_[t]; }
  /** first index in les_dir_square_ for level lvl - data is stored as (i,j,t1,t2), row-major for time indices*/
  int les_dir_square_first_index(int lvl) {
    HNESSI_ASSERT(0 <= lvl && lvl <= nbox_, "herm_matrix_hodlr::les_dir_square_first_index", HNESSI_KV(lvl),
                  HNESSI_KV(nbox_));
    return les_dir_square_first_index_[lvl];
  }

    
  // Accessors (Get)
  // Preconditions (doc 61 §2.2-2.5). Single-element accessors and raw pointers are hot paths: their index checks are
  // tier 1 (HNESSI_ASSERT, Debug/RelWithAsserts only). last = min(tstp(), nt()-1) is the last row that is both
  // computed and allocated.
  //  - get_ret/get_les(t1,t2): 0 <= t1,t2 <= last; *_curr(t1,t2): max(t1,t2) in the window [tstpmk, tstpmk+k],
  //    min(t1,t2) >= 0; map_ret_curr/set_ret_curr(t1,t2): t1 in the window, 0 <= t2 <= max(t1,k) (t2 > t1 only for the
  //    bootstrap continuation entries t2 <= k); map_les_curr/set_les_curr(t1,t2): t2 in the window, 0 <= t1 <= max(t2,k); get_tv/set_tv/tvptr(t,tau): 0 <= t < nt, 0 <= tau < r; matptr/get_mat/set_mat(i): 0 <= i < r;
  //    Eigen-matrix arguments of get_mat/set_* must already have size1 x size2 (they are not resized).
  //  - The DLR evaluators (get_mat_tau*, get_tv_tau*, get_*_reversed, get_vt, density_matrix) check once per call
  //    (tier 2, precondition_error): dlr has r() and the orbital size of this object, 0 <= tau <= beta, the row
  //    tstp is computed (0 <= tstp <= last; density_matrix: -1 <= tstp <= last), Eigen outputs are large enough.
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
  void get_mat_reversed(dlr_info &dlr, DMatrix &dest, int i, int j);
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
  void get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &dest, int i, int j);
  /** Retrieve vt (mixed component with switched contour arguments) for a given timestep. 
   * dest is assumed to be a column-major complex buffer of size r_*size1_*size2_.
  */
  void get_vt(int tstp, dlr_info &dlr, cplx *dest);
  void get_vt(int tstp, dlr_info &dlr, ZMatrix &dest);
  void get_vt(int tstp, dlr_info &dlr, ZMatrix &dest, int i, int j);
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
  void set_mat(int i, const DMatrix &M);
  void set_mat(int i, const double *M);
  template <typename Derived>
  void set_mat(int i, const Eigen::MatrixBase<Derived>& expr) {
    HNESSI_ASSERT(expr.rows() == size1_ && expr.cols() == size2_, "herm_matrix_hodlr::set_mat", HNESSI_KV(expr.rows()),
                  HNESSI_KV(expr.cols()), HNESSI_KV(size1_));
    double* raw_dest_ptr = matptr(i);
    Eigen::Map<DMatrix> dest_map(raw_dest_ptr, size1_, size2_);
    dest_map = expr;
  }
  /** Store tv component (t1,t2) from an Eigen complex matrix. */
  void set_tv(int t1, int t2, const ZMatrix &M);
  void set_tv(int t1, int t2, const cplx* M);
  template <typename Derived>
  void set_tv(int t1, int t2, const Eigen::MatrixBase<Derived>& expr) {
    HNESSI_ASSERT(expr.rows() == size1_ && expr.cols() == size2_, "herm_matrix_hodlr::set_tv", HNESSI_KV(expr.rows()),
                  HNESSI_KV(expr.cols()), HNESSI_KV(size1_));
    if(t1 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    cplx* raw_dest_ptr = tvptr(t1, t2);
    Eigen::Map<ZMatrix> dest_map(raw_dest_ptr, size1_, size2_);
    dest_map = expr;
    cplx* raw_dest_ptr2 = tvptr_trans(t1, t2);
    Eigen::Map<ZMatrix> dest_map2(raw_dest_ptr2, size1_, size2_);
    dest_map2 = dest_map.transpose();
  }
  /** Store retarded current-timestep buffer entry (t1,t2). */
  void set_ret_curr(int t1, int t2, const ZMatrix &M);
  void set_ret_curr(int t1, int t2, const cplx* M);
  template <typename Derived>
  void set_ret_curr(int t1, int t2, const Eigen::MatrixBase<Derived>& expr) {
    HNESSI_ASSERT(expr.rows() == size1_ && expr.cols() == size2_, "herm_matrix_hodlr::set_ret_curr",
                  HNESSI_KV(expr.rows()), HNESSI_KV(expr.cols()), HNESSI_KV(size1_));
    check_ret_curr_("herm_matrix_hodlr::set_ret_curr", t1, t2);
    if(t1 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    cplx* raw_dest_ptr = curr_timestep_ret_ptr(t1, t2);
    Eigen::Map<ZMatrix> dest_map(raw_dest_ptr, size1_, size2_);
    dest_map = expr;
  }
  /** Store lesser current-timestep buffer entry (t1,t2). */
  void set_les_curr(int t1, int t2, const ZMatrix &M);
  void set_les_curr(int t1, int t2, const cplx* M);
  template <typename Derived>
  void set_les_curr(int t1, int t2, const Eigen::MatrixBase<Derived>& expr) {
    HNESSI_ASSERT(expr.rows() == size1_ && expr.cols() == size2_, "herm_matrix_hodlr::set_les_curr",
                  HNESSI_KV(expr.rows()), HNESSI_KV(expr.cols()), HNESSI_KV(size1_));
    check_les_curr_("herm_matrix_hodlr::set_les_curr", t1, t2);
    if(t2 == tstpmk_+k_) can_extrap_ = false;   // ST-6
    cplx* raw_dest_ptr = curr_timestep_les_ptr(t1, t2);
    Eigen::Map<ZMatrix> dest_map(raw_dest_ptr, size1_, size2_);
    dest_map = expr;
  }

  // Get/set for current timesteps
  // get_* copies FROM *this INTO G, set_* copies FROM G INTO *this; neither modifies its source.
  // Both objects must have the same nt, nlvl, r, size1, size2, sig and k (tier 2, h_nessi::precondition_error).
  /** Copy timestep `tstp` of *this into `G` (*this is not modified).
   * tstp == -1: copy the Matsubara component (mat); no window requirement.
   * tstp >= 0: copy row tstp of the retarded buffer, column tstp of the lesser
   * buffer (entries 0..nt) and the tv/tv_trans row; tstp must lie in the current
   * window [tstpmk, tstpmk+k] of both *this and G (tier 2, precondition_error). */
  void get_timestep_curr(int tstp,herm_matrix_hodlr &G);
  /** Copy the whole current window of *this into `G` (*this is not modified):
   * sets G.tstpmk = this->tstpmk and copies the k+1 timesteps tstpmk..tstpmk+k
   * (as above, without mat). Only the window is copied, not the archived
   * history (blocks, direct stores, left edges, mat, GMConvTens): G continues
   * time stepping consistently only if its archive already holds rows
   * 0..tstpmk-1 of the same function. Use operator= for a full copy. */
  void get_timestep_curr(herm_matrix_hodlr &G);
  /** Copy timestep `tstp` of `G` into *this (G is not modified); same pieces
   * and preconditions as get_timestep_curr(tstp,G). */
  void set_timestep_curr(int tstp,herm_matrix_hodlr &G);
  /** Copy the whole current window of `G` into *this (G is not modified):
   * sets this->tstpmk = G.tstpmk and copies timesteps tstpmk..tstpmk+k. */
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
  int tv_offset(int t, int tau) {
    check_tv_("herm_matrix_hodlr::tv_offset", t, tau);
    return t*r_*size1_*size2_ + tau*size1_*size2_;
  }

  /** Pointer to directly stored retarded data in column-major order. */
  std::complex<double>* retptr_col(int t, int t1); // Directly stored data in col order

  /** Pointer to directly stored retarded correlated region (up to k+1 below diagonal). */
  std::complex<double>* retptr_corr(int t, int t1); // Directly stored data up to k+1 below diagonal

  // Algebraic operations on timesteps
  /** Increment the current timestep buffer for `tstp` by `alpha * G`. Tier 2 (HM-T5): G has the geometry of *this;
   * tstp == -1 (Matsubara) or tstp lies in the windows of both *this and G (G's slot is read). */
  void incr_timestep_curr(int tstp,herm_matrix_hodlr &G,cplx alpha);

  /** Scale stored current-timestep data for `tstp` by `alpha` (tstp == -1 or in the window; tier 1). */
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
   * This allocates and fills the internal GMConvTensMatrix_. `dlr` must have the same xi, r and size as this object
   * (tier 2, h_nessi::precondition_error). It must be called after the last change of G^M (after the last
   * dyson_mat): the solvers check (ST-2) that the tensor belongs to the current G^M and throw h_nessi::state_error
   * otherwise. */
  void initGMConvTensor(dlr_info &dlr);
  /** 64-bit hash (FNV-1a over the bit patterns) of the G^M data; O(r*size1*size2). */
  std::uint64_t mat_hash() const;
  /** ST-2: true if the convolution tensor was built by initGMConvTensor from the current G^M. */
  bool gmconv_current() const;
  /** ST-2: throws h_nessi::state_error unless gmconv_current(); `where` names the calling routine. */
  void require_gmconv_current(const char *where, bool check_hash = true) const;

  // I/O and diagnostics
  /** Version written as `label/format_version` by write_checkpoint_hdf5.
   * 2: GMConvTensMatrix carries the xi factor (bosonic sign fix). Missing = 1. */
  static constexpr int checkpoint_format_version = 2;

  /** Write a lightweight checkpoint containing the current runtime buffers. */
  void write_checkpoint_hdf5(h5e::File &out, std::string label);

  /** Write the full object to HDF5 under `label`. Prints a warning (h_nessi::detail::warn) if the k+1 final
   * update_blocks calls have not been made (tstpmk() < nt(): rows >= tstpmk are not in the archive written here).
   * If `geometry/<geometry_label>` already exists it must describe the same geometry (tier 2). */
  void write_to_hdf5(h5e::File &out, std::string label, std::string geometry_label = "");

  /** Write the density matrix (rho) to HDF5 using `dlr` for Matsubara transforms (all rows must be computed:
   * tstp() >= nt()-1, tier 2). */
  void write_rho_to_hdf5(h5e::File &out, std::string label, dlr_info &dlr);

  /** Write per-block ranks to HDF5 for diagnostics. */
  void write_rank_to_hdf5(h5e::File &out, std::string label);

  /** Write matsubara matrices to HDF5. */
  void write_mat_to_hdf5(h5e::File &out, std::string label);

  /** Write the current circular buffers to HDF5. */
  void write_curr_to_hdf5(h5e::File &out, std::string label);

  /** Write GL0 (left boundary) helper data to HDF5 (all rows must be computed: tstp() >= nt()-1, tier 2). */
  void write_GL0_to_hdf5(h5e::File &out, std::string label);

  /** Write GR0 (right boundary) helper data to HDF5 (all rows must be computed: tstp() >= nt()-1, tier 2). */
  void write_GR0_to_hdf5(h5e::File &out, std::string label);

  /** Print a small report on memory usage (compressed and full) to stdout. 
  *   Returns total memory usage in GB.
  */
  double get_memory_usage(bool print_memory_usage=true);


  private:
  // Tests reach the private numerical kernels (mgs, updatetsvd) through this struct (defined in the test sources).
  friend struct herm_matrix_hodlr_test_access;

  // geometry helpers (private, doc 61 Q-12)
  /** Compute and store information about block levels and row/column indices of directly stored regions.
   * @return maxdir - maximum number of directly-stored entries in a row
   */
  int getbkl_indices(void);

  /** Initialize internal geometry arrays for nt_ = nt and nlvl_ = nlvl (k_, size1_, size2_ must be set). */
  void init_shape(int nt,int nlvl);

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

  /// @private last row that is computed and allocated: min(tstp(), nt-1)
  int last_row_() const { return std::min(tstpmk_ + k_, nt_ - 1); }
  /// @private tier-1 index checks shared by the inline accessors
  void check_blk_(const char *where, int b) const {
    HNESSI_ASSERT(0 <= b && b < nbox_, where, HNESSI_KV(b), HNESSI_KV(nbox_));
  }
  void check_t_(const char *where, int t) const {
    HNESSI_ASSERT(0 <= t && t < nt_, where, HNESSI_KV(t), HNESSI_KV(nt_));
  }
  void check_tv_(const char *where, int t, int tau) const {
    HNESSI_ASSERT(0 <= t && t < nt_ && 0 <= tau && tau < r_, where, HNESSI_KV(t), HNESSI_KV(tau), HNESSI_KV(nt_),
                  HNESSI_KV(r_));
  }
  // R: row t1 in the window, 0 <= t2 <= max(t1, k) (the entries t1 < t2 <= k of the bootstrap rows hold the
  // continuation G^R(t1,t2) used by the non-TTI bootstrap and archived into ret_left_edge, doc 22); < likewise.
  void check_ret_curr_(const char *where, int t1, int t2) const {
    HNESSI_ASSERT(tstpmk_ <= t1 && t1 <= tstpmk_ + k_ && 0 <= t2 && t2 <= std::max(t1, k_), where, HNESSI_KV(t1),
                  HNESSI_KV(t2), HNESSI_KV(tstpmk_), HNESSI_KV(k_));
  }
  void check_les_curr_(const char *where, int t1, int t2) const {
    HNESSI_ASSERT(tstpmk_ <= t2 && t2 <= tstpmk_ + k_ && 0 <= t1 && t1 <= std::max(t2, k_), where, HNESSI_KV(t1),
                  HNESSI_KV(t2), HNESSI_KV(tstpmk_), HNESSI_KV(k_));
  }
  /// @private tier-2 checks of the DLR evaluators (getit)
  void check_tau_(const char *where, double tau, const dlr_info &dlr) const;
  void check_row_(const char *where, int tstp) const;
  /// @private HM-T1..T5 (tier 2): G has the geometry of *this
  void check_same_geometry_(const char *where, herm_matrix_hodlr &G) const;

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
    /// @private ST-2: GMConvTensMatrix_ was built by initGMConvTensor (false for a pre-v2 bosonic checkpoint)
    bool gmconv_valid_ = false;
    /// @private ST-2: mat_hash() of the G^M the tensor was built from
    std::uint64_t gmconv_hash_ = 0;
    /// @private dlr pairing check (tier 2)
    void check_dlr_(const char *where, const dlr_info &dlr) const;
    /// @private HM-C1..C7 constructor validation
    static void check_geometry_(const char *where, int nt, int r, int nlvl, double svdtol, int size1, int size2, int sig,
                                int k);
};

/**
 * @name Distances between two herm_matrix_hodlr objects (absolute Frobenius norm of the difference)
 * Preconditions (tier 2, h_nessi::precondition_error): equal size1 and size2; curr_ret/curr_les: 0 <= tstp in the
 * current windows of both objects and tstp < nt of both; mat, tv: equal r; tv: 0 <= tstp < min(g1.nt(), g2.nt());
 * curr: tstp >= -1 (-1 is the Matsubara distance).
 */
///@{
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_mat(herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, herm_matrix_hodlr &g2);
///@}

#ifdef INCLUDE_NESSI
/**
 * @name Distance to a NESSi `cntr::herm_matrix<double>` (compiled only with -DINCLUDE_NESSI=ON)
 *
 * Compare a herm_matrix_hodlr `g1` with a NESSi (uncompressed, equidistant-tau) `cntr::herm_matrix<double>` `g2` of
 * the same contour function. Conventions and the storage mapping are in agent_docs/05_nessi_herm_matrix.md:
 * both store G^R(t,t') for t >= t' and G^<(t,t') for t <= t' with the same orbital order (row-major (i,j)), the
 * same sign conventions (G^R = G^> - G^<, rho = -G^M(beta) = i xi G^<(t,t), G^tv(t,0) = G^<(t,0)), and the same
 * time index t_n = n h. NESSi's `nt()` is the LAST time index (nt+1 points) while H-NESSi's `nt()` is the number of
 * points. NESSi stores G^M and G^tv at tau_m = m beta/ntau, m = 0..ntau (ntau = g2.ntau()); H-NESSi stores them at
 * the DLR nodes, so for mat/tv the DLR expansion of g1 is evaluated at NESSi's grid (get_mat_tau_array /
 * get_tv_tau_array), and these two distances contain the DLR interpolation error (~ eps_dlr * |G|).
 *
 * All five return an absolute, unnormalised Frobenius (l2) norm of the difference, summed over all stored entries
 * listed below (no division by the number of points; mat and tv therefore grow like sqrt(ntau+1)). With
 * ||A||^2 = sum_ij |A_ij|^2 for an N_o x N_o block:
 *  - distance_norm2_curr_ret(tstp,g1,g2)  = sqrt( sum_{t'=0..tstp} ||g1^R(tstp,t') - g2^R(tstp,t')||^2 )
 *  - distance_norm2_curr_les(tstp,g1,g2)  = sqrt( sum_{t=0..tstp}  ||g1^<(t,tstp) - g2^<(t,tstp)||^2 )
 *  - distance_norm2_tv(tstp,g1,g2,dlr)    = sqrt( sum_{m=0..ntau}  ||g1^tv(tstp,tau_m) - g2^tv(tstp,tau_m)||^2 )
 *  - distance_norm2_mat(g1,g2,dlr)        = sqrt( sum_{m=0..ntau}  ||g1^M(tau_m) - g2^M(tau_m)||^2 )
 *    (g1^M is real, H-NESSi limitation D1: an imaginary part of g2^M counts fully in the distance)
 *  - distance_norm2_curr(tstp,g1,g2,dlr)  = distance_norm2_mat(g1,g2,dlr) for tstp == -1, otherwise
 *                                           sqrt( curr_ret^2 + curr_les^2 + tv^2 ) at tstp.
 *
 * Preconditions (checked, h_nessi::precondition_error): equal size1/size2 (square), equal sig, g2.nt() >= tstp; for the
 * real-time rows g1 must hold row tstp in its current window (g1.tstpmk() <= tstp <= g1.tstp()); for mat/tv
 * g2.ntau() >= 1 and dlr.r() == g1.r(), dlr.xi() == g1.sig(), dlr.size1() == g1.size1(); tv needs a computed row,
 * 0 <= tstp <= min(g1.tstp(), g1.nt()-1). The checks throw h_nessi::precondition_error (a std::invalid_argument).
 * The beta of NESSi's tau grid is taken from dlr.beta() (NESSi does not store beta).
 * Not thread-safe for a shared dlr_info (eval_point uses dlr scratch, doc 03).
 */
///@{
double distance_norm2_curr(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr);
double distance_norm2_mat(herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr);
double distance_norm2_curr_ret(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_curr_les(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2);
double distance_norm2_tv(int tstp, herm_matrix_hodlr &g1, cntr::herm_matrix<double> &g2, dlr_info &dlr);
///@}
#endif


}  // namespace
