
/**
 * @file dyson.hpp
 * @brief Declares the dyson class for solving quantum many-body Dyson equations using HODLR and DLR methods.
 *
 * This file provides the dyson class, which manages time-stepping, Green's function construction, bootstrapping, and convolution routines for quantum many-body systems.
 */
#ifndef DYSON_DECL
#define DYSON_DECL

#include <vector>
#include <iostream>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>
#include <chrono>
#include <climits>

#include "function.hpp"
#include "herm_matrix_hodlr.hpp"
#include "integration.hpp"
#include "utils.hpp"
#include "dlr.hpp"
#include "checks.hpp"

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
 * @class dyson
 * @brief Solves quantum many-body Dyson equations using HODLR and DLR representations.
 *
 * The dyson class manages all routines for constructing and evolving Green's functions, performing time-stepping, bootstrapping, and convolutions in quantum many-body calculations. It supports both fermionic and bosonic statistics, and interfaces with hierarchical matrix and DLR basis representations.
 *
 * Checks (checks.hpp): every public entry point validates its arguments and the call order (geometry of G and Sigma,
 * pairing with the dlr_info, I.k(), the time step against G.tstpmk()/Sigma.tstpmk(), the G^M convolution tensor, ...)
 * and throws h_nessi::precondition_error / state_error / numerical_error (inside a user OpenMP parallel region it prints
 * the message and aborts instead). **The h argument is modified:** every slice h(t) a call reads is replaced in place by
 * its Hermitian part (h+h^dagger)/2 (decision D-11; see set_hermiticity_tolerances, set_hermitian_symmetrization).
 *
 * **Returned error (ISSUES H31, decision D-10: kept unnormalised).** The double returned by the dyson_* solvers is
 * the change of the solution made by the call, as an *unnormalised sum* over the time entries written of the
 * Frobenius norm of the nao x nao change, ||G_new(t,t') - G_old(t,t')||_F (sum over t' of one row for a time step,
 * over the k+1 bootstrap rows for dyson_start*; summed over the components ret/tv/les where one call solves several).
 * It is a valid norm of the change, but it grows with the number of entries (~T for a time step) and with nao, so a
 * fixed tolerance becomes stricter per entry as T grows; divide by (T+1)*nao yourself for a per-entry measure.
 * dyson_mat returns the change of the unmixed Matsubara solution. The value is meant for self-consistency loops
 * (iterate until it is below a tolerance), not as an error estimate of the discretisation.
 *
 * **Contract of the time steps (P1): values at past times are final.** The time-step solvers factorise the nao x nao
 * step matrix of each past time t (built from h(t), Sigma^R(t,t), mu, the step h) once and reuse it in all later
 * corrector iterations and time steps. Hence h(t) and Sigma^R(t,t) must not be modified after the time step t is done
 * (the corrector loop of step T may change row T only). The cache is cleared automatically when mu, h, the Sigma object
 * or the h array (pointer) change (so one object may serve several systems, e.g. k-points, at the cost of refactorising
 * on every switch), by every dyson_start* / green_from_H* call (start of a run), and is empty after construction or a
 * checkpoint restart.
 * Debug builds check every reuse (DY-T12, assertion failure on a violation).
 */
class dyson {
private:

  int nao_;           /**< Number of atomic orbitals. */
  int es_;            /**< Element size (nao * nao). */
  int k_;             /**< Time stepping order. */
  int nt_;            /**< Number of time steps. */
  int xi_;            /**< Particle statistics (+1 for bosons, -1 for fermions). */

  // libdlr parameters
  double beta_;       /**< Inverse temperature. */
  int ntau_;          /**< Number of imaginary time points. */
  int r_;             /**< Number of imaginary time points. */
  dlr_info &dlr_;     /**< Reference to DLR information object. */

  bool rho_version_;   /**< Density matrix calculation version. 1 for diagonal, 0 for horizontal. */

  // Temporary storage for linear algebra in time-stepping routines
  std::vector<cplx> NTauTmp_;         /**< imaginary time storage - size ntau*es*/
  std::vector<cplx> NTauTmp2_;        /**< imaginary time storage - size ntau*es*/
  std::vector<double> DNTauTmp_;         /**< imaginary time storage - size ntau*es*/
  std::vector<double> DNTauTmp2_;        /**< imaginary time storage - size ntau*es*/
  std::vector<double> DG0_;              /**< dyson_mat: G0 on the DLR nodes (libdlr layout g0(r,n,n)), kept for fixHam=true - size ntau*es*/
  std::vector<double> DG0ConvMat_;       /**< dyson_mat: matrix of convolution by G0 (libdlr layout (r*n,r*n)), kept for fixHam=true - size ntau*ntau*es*/
  std::vector<cplx> Q_;               /**< storage for timeslice quantities such as history integrals - size (nt+1)*es*/
  std::vector<cplx> M_;               /**< storage for Matrix which is solved when timestepping - size (k+1)*k*es */
  std::vector<cplx> X_;               /**< storage for timeslice quantities such as the G being solved for  - size (nt+1)*es */
  std::vector<cplx> epsnao_tmp_;      /**< storage for intermediate quantities when applying U,S,V sequentially - size (nt+1)/2*nao
                                           (P1: S is applied in place, the second buffer epsnao_tmp_2_ was removed) */
  std::vector<cplx> iden_;            /**< Identity matrix storage. */
  std::vector<cplx> bound_;           /**< storage for matrices when we need to build them from compressed representations - size nao*nao */

  /**< Timing information for profiling. 
   * 0 - Retarded component
   * 1 - Retarded integral
   * 2 - current timestep region of integral
   * 3 - block applications of integral
   * 4 - direct region of integral
   * 5 - direct and correction regions of integral together (4 + 6 incl. loop overhead; was never filled: BUGS B28)
   * 6 - correction region below direct region of integral
   * 7 - Mixed component
   * 8 - real-time mixed integral
   * 9 - imaginary-time mixed convolution
   * 10 - Lesser component
   * 11 - imaginary-time lesser integral
   * 12 - real-time lesser integral from 0 to T
   * 13 - direct region of integral
   * 14 - direct corrections
   * 15 - block applications of integral
   * 16 - transposed block applications of integral
   * 17 - correction region on left edge horizontally
   * 18 - correction region on left edge vertically
   * 19 - current timestep region of integral vertical
   * 20 - current timestep region of integral horizontal
   * 21 - real-time lesser integral from 0 to t
   * 22 - block applications of integral
   * 23 - current timestep region of integral
   * 24 - correction region below direct region of integral - retarded
   * All timers (steady_clock::now() calls) run only if profile_ (P1); with profile_ = false nothing is measured.
   * Per-HODLR-level split of cols 3, 22, 15, 16: timing_lvl_ below.
  */
  bool profile_;
  DMatrix timing;
  /**< P1: per-HODLR-level split of the TSVD block applications, nt x (4*tl_lmax_) if profile_, else 0 x (4*tl_lmax_),
   * microseconds, last call per tstp (like timing). Column g*tl_lmax_ + (L-1) holds group g at code level L of the block
   * (herm_matrix_hodlr::blklevel, 1 = smallest blocks next to the diagonal; doc 11 §2):
   *   g = 0: ret integral blocks (sum over L = timing col 3), g = 1: les int3 blocks (col 22),
   *   g = 2: lesadv blocks (col 15), g = 3: lesadv transposed blocks (col 16).
   * tl_lmax_ = floor(log2(nt)), the largest nlvl a herm_matrix_hodlr with nt points can have (2^nlvl <= nt).
   * Written by write_timing / write_checkpoint_hdf5 as <label>/timing_level. */
  DMatrix timing_lvl_;
  int tl_lmax_ = 1;
  using clock_tp_ = std::chrono::steady_clock::time_point;
  /** steady_clock::now() if profile_, else a zero time point (differences of zero time points are 0) */
  clock_tp_ pnow_() const { return profile_ ? std::chrono::steady_clock::now() : clock_tp_(); }
  /** accumulate the duration b-a (us) into timing_lvl_(tstp, g*tl_lmax_+L-1) (profile_ only) */
  void add_lvl_(int tstp, int g, int L, clock_tp_ a, clock_tp_ b) {
    if(profile_) timing_lvl_(tstp, g*tl_lmax_ + L-1) += std::chrono::duration<double, std::micro>(b-a).count();
  }
  /** zero group g of row tstp of timing_lvl_ (start of a call; profile_ only) */
  void zero_lvl_(int tstp, int g) {
    if(profile_) timing_lvl_.block(tstp, g*tl_lmax_, 1, tl_lmax_).setZero();
  }

  // ---- P1: cached LU factorisations of the nao x nao step matrices ------------------------------------------------
  // dyson_timestep_ret solves M^T(t') X^T = Q^T for each t' = T-n, n > k (t' <= T-k-1, archived), with
  //   M^T(t') = (mu - i a_0/h) 1 - h(t')^T - h w_0 Sigma^R(t',t')^T,
  // the les march (les_march_) solves M(m) X^T = Q for m = k+1..T with M(m) = (mu + i a_0/h) 1 - h(m) - h w_0 Sigma^R(m,m)
  // (RHO_DIAGONAL, m = T: a_0/h 1). a_0 = bd_weights(0), w_0 = omega(0) depend only on k (fixed per object).
  // CONTRACT (maintainer decision): the values h(t) and Sigma^R(t,t) of a time t are final once the time step t is
  // done; they are not modified afterwards. M^T(t') (ret, t' <= T-k-1) and M(m) (les, m < T) are therefore the same in
  // every corrector iteration and every later time step, and each is factorised once, when first needed, and reused
  // without comparing inputs. M(T) of the les march (row T changes every iteration) is never cached. The cache is
  // cleared (lu_valid_ = 0) when mu, the step h, the Sigma object or the h array (pointer) differ from those it was built
  // with (lu_mu_, lu_h_, lu_sig_, lu_H_; checked once per call: one dyson object used for several k-points, as in the
  // mpi_example, refactorises on every switch), by every dyson_start* and green_from_H* call (start of a run), and it
  // starts empty in both constructors
  // (a checkpoint restart rebuilds it lazily; not checkpointed). Debug builds (tier 1) verify every reuse against a fresh
  // factorisation of the newly built M (DY-T12: a violation of the contract). Slots: ret t' -> t', les m -> nt_ + m,
  // scratch (uncached) 2 nt_. Factorisation: LU with partial (row) pivoting, packed L\U with unit L.
  std::vector<cplx> lu_fac_;            /**< (2nt+1)*es: packed L\U of slot s */
  std::vector<int> lu_piv_;             /**< (2nt+1)*nao: row interchanges of slot s */
  std::vector<unsigned char> lu_valid_; /**< 2nt: slot s holds a factorisation */
  double lu_mu_ = 0, lu_h_ = 0;         /**< mu and h the cached factorisations were built with */
  const void *lu_sig_ = nullptr;        /**< Sigma object the cache was built with (identity only, never dereferenced) */
  const cplx *lu_H_ = nullptr;          /**< h array the cache was built with (identity only) */
  // X = (A^{-1} op(B))^T with A = M (nao x nao, row-major), op(B) = B^T if transB else B; X must not alias M or B. With
  // cache = true the factors of slot `slot` are reused if present (else computed and stored), with cache = false M is
  // factorised into the scratch slot. DY-T11 (tier 1) is checked when M is factorised.
  void lu_solve_cached_(int slot, bool cache, const cplx *M, const cplx *B, bool transB, cplx *X, const char *where);
  // clear the cache if (mu, h, Sigma, h array) differ from the cached ones; reset_lu_cache_: clear unconditionally
  void lu_params_(double mu, double h, const herm_matrix_hodlr &Sigma, const cplx *H) {
    if(mu != lu_mu_ || h != lu_h_ || &Sigma != lu_sig_ || H != lu_H_) {
      reset_lu_cache_(); lu_mu_ = mu; lu_h_ = h; lu_sig_ = &Sigma; lu_H_ = H;
    }
  }
  void reset_lu_cache_() { std::fill(lu_valid_.begin(), lu_valid_.end(), (unsigned char)0); }
  bool lu_cache_consistent_(int slot, const cplx *M);
  void alloc_p1_buffers_();
  // P1: tv_ret_int batches tv_chunk_ times t into one GEMM: wide w*Sigma^R row (nao x tv_chunk_*nao) times the stacked
  // transposed G^tv rows (tv_chunk_*nao x nao*r)
  static constexpr int tv_chunk_ = 32;
  std::vector<cplx> tv_wide_;           /**< nao * tv_chunk_ * nao */
  std::vector<cplx> tv_stack_;          /**< tv_chunk_ * r * es */
  // march over rows m = k+1..T of the les components (identical in dyson_timestep_les, _les_nobc, _les_2leg): retles
  // integral (timing cols 21-24), RHS, solve; writes G^<(.,T) from X_ and returns the change
  double les_march_(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                    Integration::Integrator &I, double h, const char *where);

  // ---- WP6 checks (checks.hpp) -------------------------------------------------------------------------------
  int api_depth_ = 0;          /**< nesting depth of public calls: entry checks run only in the outermost call */
  // D-11: Hermiticity of h(t)
  double herm_warn_tol_ = 1e-4;  /**< warn once when the relative anti-Hermitian part of a slice of h exceeds this */
  double herm_err_tol_ = 1e-2;   /**< throw h_nessi::precondition_error when it exceeds this */
  bool herm_symmetrize_ = true;  /**< replace each slice of h read by (h+h^dagger)/2 in place (false: check only) */
  bool herm_warned_ = false;     /**< the warning has been printed (once per dyson object) */
  double herm_last_ = 0;         /**< largest relative anti-Hermitian part found by the last call */
  std::vector<double> herm_diag_; /**< herm_diag_[t+1], -1<=t<nt: largest relative anti-Hermitian part of h(t) found
                                       before symmetrisation (per-step diagnostic, size nt+1) */
  // D-11, WP6a side finding 1 (maintainer decision (a)): the tolerances are enforced on the h(T) used by the corrector
  // solves, not on the h(T) built from the extrapolated predictor. The newest slice T of a time step is measured and
  // symmetrised on its first solve, but its warn/throw tolerances are deferred: a later solve of the same step that
  // finds a rewritten h(T) enforces the new value (the predictor value is dropped); if h(T) is never rewritten, the
  // deferred value is enforced at the next call that reads slice T (the next time step).
  int herm_copy_t_ = INT_MIN;      /**< slice whose symmetrised value is held in herm_copy_ (the newest slice) */
  std::vector<cplx> herm_copy_;    /**< h(herm_copy_t_) as left by the last measurement (size es) */
  bool herm_pend_ = false;         /**< herm_pend_rel_ of slice herm_copy_t_ is measured but not yet enforced */
  double herm_pend_rel_ = 0;       /**< deferred relative anti-Hermitian part of slice herm_copy_t_ */
  // ST-3: dyson_mat(fixHam=true) needs the G0 saved by an earlier fixHam=false call (not checkpointed)
  bool g0_valid_ = false;
  // ST-4: order of the solver components within one time step / the bootstrap (bits of the components solved for
  // (stamp_G_, stamp_tstp_); stamp_tstp_ == boot_stamp_ for the bootstrap)
  const herm_matrix_hodlr *stamp_G_ = nullptr;
  int stamp_tstp_ = INT_MIN;
  unsigned stamp_mask_ = 0;
  enum : unsigned { st4_ret = 1u, st4_tv = 2u, st4_ret_ntti = 4u };
  static constexpr int boot_stamp_ = -2;
  // ST-10 (DY-Z2): one dyson object per thread
  detail::single_thread_owner owner_;
  // ST-11 (D-9): DLR parameters of the dlr_info this object was built with (stored in the checkpoint)
  double dlr_eps_ = 0, dlr_lambda_ = 0;
  // DY-S7: the all-zero G^<(0,0) warning of the _nobc/_2leg bootstraps is printed once
  bool les00_warned_ = false;

  /** RAII guard of every public call: ST-10 ownership (only the outermost call acquires it) and the nesting of public
   * calls (dispatchers call public components; the entry checks run only in the outermost call). */
  struct call_scope {
    dyson &d;
    detail::owner_scope own;
    explicit call_scope(dyson &dd) : d(dd), own(dd.owner_, "dyson") { ++d.api_depth_; }
    ~call_scope() { --d.api_depth_; }
    bool outer() const { return d.api_depth_ == 1; }
  };
  friend struct dyson_test_access;
  void check_solver_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma, Integration::Integrator &I,
                          double h, double mu);
  void check_step_(const char *where, int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void check_boot_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma);
  void check_function_(const char *where, function &H, int tlast);
  double hermitize_(cplx *H0, int t0, int t1, const char *where, int newest = INT_MIN);
  void herm_enforce_(double rel, int t, const char *where);
  // ST-4
  void stamp_(herm_matrix_hodlr &G, int tstp, unsigned bit);
  void require_stamp_(const char *where, herm_matrix_hodlr &G, int tstp, unsigned bits, const char *hint);
  // DY-S7: G^<(0,0) set by the caller for the _nobc/_2leg bootstraps
  void check_les00_(const char *where, herm_matrix_hodlr &G);
  // DY-M10/F10: bosonic spectrum of h - mu (eigenvalues eval0 of mu - h)
  void require_bosonic_spectrum_(const DColVector &eval0, const char *where);
  double hermitize_real_(DMatrix &H, const char *where);
  void require_real_(const cplx *h, const char *where);
  double ensure_finite_(double err, const char *where);
  void enter_timestep_(const char *where, int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma,
                       Integration::Integrator &I, double h, double mu, cplx *H, int hlo, int hhi, bool uses_tv);
  void enter_start_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma, Integration::Integrator &I,
                    double h, double mu, cplx *H, bool reads_h, bool uses_tv);
  void check_dyson_ctor_(const char *where, int nt, int nao, int k, dlr_info &dlr);
  void check_mat_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma, double mu, double alpha,
                       bool fixHam);
  void ensure_mat_finite_(const char *where);
  void check_it_args_(const char *where, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G);
  // X = ipmat^T it2itr^T (i xi conj(G.tv row tstp)) into NTauTmp2_ (r x n x n): the G operand of both les_it_int overloads
  void les_it_int_prelude_(int tstp, herm_matrix_hodlr &G);
  void check_gamma_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta);
  void check_free_args_(const char *where, herm_matrix_hodlr &G, double mu, const ZMatrix &H, double h, int tmax);
  // bootstrap rows m = 1..k of G^< (and the RHO_DIAGONAL diagonal fixed point) shared by dyson_start_les_ntti, _nobc
  // and _2leg (dyson_start.cpp, B34/WP7b); err accumulates the change of G^<(m,n), 1 <= m <= n <= k
  void start_les_rows_(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I,
                       double h, bool mixed, double &err, const char *where);
  // anti-Hermitian part (X - X^dagger)/2 of the boundary value G^<(0,0) (dyson_start_les, _les_ntti; WP7b SF4)
  static ZMatrix boundary_antihermitian_(const ZMatrix &X);

public:
  /**
   * RHO_DIAGONAL bootstrap (dyson_start_ntti, _ntti_nobc, _2leg; B34/WP7b). Rows and diagonal of G^< in t = 0..k
   * depend on each other; each call iterates them to their common fixed point for the given Sigma, until the largest
   * change of a diagonal block, max_i ||Delta G^<(i,i)||_F, is <= start_diag_rel_tol * max(1, max_i ||G^<(i,i)||_F).
   * The contraction factor per pass grows with h (about 0.1 at h = 0.05 and 0.75 at h = 0.4, k = 5, in the test models:
   * 10-20 resp. ~110 passes from a zero diagonal; it can exceed 1 for large h*spread(eps), e.g. h = 0.4 in a closed
   * 5-level model). If the iteration diverges (non-finite change, or after 10 passes a change > 100x the smallest one
   * so far) or start_diag_max_pass passes do not converge, the call throws h_nessi::numerical_error. Static
   * constants: no per-object storage.
   * Large-h limit (WP9a, closed driven 3- and 5-level models, Sigma = 0, k = 3 and 5): the bootstrap fixed point
   * converges up to h*spread ~ 0.9 (k = 3) resp. 0.6 (k = 5) and fails from h*spread ~ 0.9-1.1, where spread is the
   * eigenvalue range of h(t). The RHO_DIAGONAL time step (diagonal fixed point iterated by the user's corrector loop)
   * tolerates about twice that: it converges up to h*spread ~ 1.7 (with ~100-150 corrector iterations near the limit)
   * and diverges or stalls from h*spread ~ 2.2. The bootstrap is therefore the binding limit for RHO_DIAGONAL.
   */
  static constexpr double start_diag_rel_tol = 1e-14;
  static constexpr int start_diag_max_pass = 1000;

  /**
   * @brief Constructs a dyson object with specified parameters.
   * @param nt Number of time steps.
   * @param nao Number of atomic orbitals.
   * @param k Maximum order for time-stepping.
   * @param dlr Reference to DLR information object.
   * @param rho_version Density matrix calculation version (default: 1).  1 = RHO_DIAGONAL: G^<(t,t) from the
   *        density-matrix equation, a fixed point that the bootstrap iterates internally and the time step needs
   *        the user's corrector loop for, even at fixed Sigma; it stops converging for large h (bootstrap from
   *        h*spread(h(t)) ~ 0.6-1, time step from ~2; see start_diag_rel_tol). 0 = RHO_HORIZONTAL (no such limit).
   * @param profile whether or not to include profiling (default: 1).
   */
  dyson(int nt, int nao, int k, dlr_info &dlr, bool rho_version=1, bool profile = 1);

  /**
   * @brief Constructs a dyson object from checkpoint file
   * @param in hdf5 checkpoint file.
   * @param label name of dataset in checkpoint file.
   * @param dlr Reference to DLR information object.
   */
  dyson(h5e::File &in, std::string label, dlr_info &dlr);

  /** @brief Returns number of imaginary time points. */
  int r() {return r_;}
  /** @brief Returns number of imaginary time points. */
  int ntau() {return ntau_;}
  /** @brief Returns particle statistics (+1 for bosons, -1 for fermions). */
  int xi(){return xi_;};

  /**
   * @name Hermiticity of h(t) (decision D-11)
   * Every dyson_* entry point and green_from_H* checks each slice of h it reads. By default it also replaces the slice,
   * **in place in the caller's function object or buffer**, by its Hermitian part (h+h^dagger)/2, so that all code
   * paths (which use H^T and conj(H)) see the same Hermitian h. The relative anti-Hermitian part
   * ||h(t)-h(t)^dagger||_F / ||h(t)||_F of each slice is recorded (before symmetrisation) per slice; a warning is
   * printed once per dyson object above the warning tolerance (default 1e-4) and h_nessi::precondition_error is
   * thrown above the error tolerance (default 1e-2), before the slice is modified.
   * Predictor/corrector (WP6b): in dyson_timestep*(T) the tolerances of the newest slice h(T) apply to the h(T) of the
   * corrector solves. The first solve of step T (h(T) built from the extrapolated predictor) symmetrises h(T) but
   * defers its check; a later solve of the step that finds h(T) rewritten enforces the new value at once; an h(T)
   * never rewritten during the step is enforced by the next call that reads it. Switch drives on smoothly.
   * @{
   */
  void set_hermiticity_tolerances(double warn_tol, double err_tol);
  double hermiticity_warn_tol() const { return herm_warn_tol_; }
  double hermiticity_err_tol() const { return herm_err_tol_; }
  /** true (default): symmetrise h in place; false: check only (h is not modified; for debugging). */
  void set_hermitian_symmetrization(bool on) { herm_symmetrize_ = on; }
  bool hermitian_symmetrization() const { return herm_symmetrize_; }
  /** Largest relative anti-Hermitian part of h found by the last call that read h. */
  double last_antihermiticity() const { return herm_last_; }
  /** Per-slice diagnostic: element t+1 (-1 <= t < nt) is the largest relative anti-Hermitian part of h(t) seen. */
  const std::vector<double> &antihermiticity_per_slice() const { return herm_diag_; }
  /** @} */

  /** ST-3: true if dyson_mat(fixHam=false) has saved a G0 in this object (fixHam=true needs it). */
  bool g0_valid() const { return g0_valid_; }

  /**
   * @brief Provides a small report on memory usage
   */
  void print_memory_usage();

  /**
   * @brief Memory used by the dyson buffers in bytes (actual sizes of all heap buffers), as reported by print_memory_usage
   * @return {DLR buffers (NTauTmp*_, DNTauTmp*_), G0 (DG0_), G0 conv. matrix (DG0ConvMat_), Mat Mul buffer (epsnao_tmp_),
   *          Solver buffers (Q_, X_, M_, iden_, bound_, P1 LU cache lu_*_, P1 tv_ret_int batch tv_wide_/tv_stack_),
   *          Timing/diagnostics (timing, timing_lvl_, herm_diag_, herm_copy_, state), Total}
   */
  std::vector<double> memory_usage_bytes();

  /**
   * @brief Writes timing information to an HDF5 file.
   * @param file HDF5 file handle.
   * @param label name of dataset.
   */
  void write_timing(h5e::File &file, std::string label = "dyson_timing");

  /**
   * @brief Writes checkpoint file.
   * @param out HDF5 file handle.
   * @param label name of dataset.
   */
  void write_checkpoint_hdf5(h5e::File &out, std::string label);

  /**
   * @brief Integrates the dipole field. Eq (11) in Phys. Rev. A 92, 033419 (2015)
   * @param tstp Time step index.
   * @param dfield Dipole field function.
   * @param Gu Green's function (up spin).
   * @param Gd Green's function (down spin).
   * @param dipole Output dipole array.
   * @param l Parameter l.
   * @param n Parameter n.
   * @param h Time step size.
   * @param I Integration object.
   *
   * @warning NOT IMPLEMENTED. The declaration is kept as a placeholder, reserved for a later reimplementation
   *          (decision D-2, ISSUES Q7). There is no definition in the library (the old body in dyson.cpp is
   *          commented out), so any call fails at link time with an undefined reference to dyson::dipole_step.
   */
  void dipole_step(int tstp, function &dfield, herm_matrix_hodlr &Gu, herm_matrix_hodlr &Gd, double *dipole, double l, double n, double h, Integration::Integrator &I);
  /**
   * @brief Constructs free Green's function from Hamiltonian (function version).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (function).
   * @param h Time step size.
   * @param tmax Maximum time step.
   * @param inc_mat Include Matsubara axis (default: false).
   */
  void green_from_H(herm_matrix_hodlr &G, double mu, function &H, double h, int tmax, bool inc_mat = false);
  /**
   * @brief Constructs free Green's function from Hamiltonian (complex version).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param h Time step size.
   * @param tmax Maximum time step.
   * @param inc_mat Include Matsubara axis (default: false).
   */
  void green_from_H(herm_matrix_hodlr &G, double mu, ZMatrix H, double h, int tmax, bool inc_mat = false);
  /**
   * @brief Constructs free Green's function from Hamiltonian and density matrix (complex version). Uses rho to set intial condition for the density matrix, instead of assuming thermal equilibrium.
   * Sets G^R(t,t') = -i U(t) U(t')^\dagger and G^<(t,t') = -i xi U(t) rho U(t')^\dagger with U(t) = exp(-i(H-mu)t).
   * G^\rceil is built from the thermal density matrix of H, not from rho (so it matches G^< only if rho is thermal for H); Matsubara is not written.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param rho initial density matrix rho = -G^M(beta^-) = xi i G^<(0,0) (complex), e.g. G.density_matrix(-1); same convention for fermions and bosons.
   * @param h Time step size.
   * @param tmax Maximum time step.
   */
  void green_from_H_dm(herm_matrix_hodlr &G, double mu, ZMatrix H, ZMatrix &rho, double h, int tmax);
  /**
   * @brief Constructs free Matsubara Green's function from Hamiltonian (complex version, output to array).
   * @param g0 Output array.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   */
  void green_from_H_mat(double *g0, double mu, ZMatrix H);

  /**
   * @brief Extrapolates Green's function data for time-stepping.
   * @param G Green's function object.
   * @param I Integration object.
   */
  void extrapolate(herm_matrix_hodlr &G, Integration::Integrator &I);
  /**
   * @brief Extrapolates Green's function data for two-leg time-stepping.  Does not use initial condition that connects < and \rceil
   * @param G Green's function object.
   * @param I Integration object.
   */
  void extrapolate_2leg(herm_matrix_hodlr &G, Integration::Integrator &I);

  /**
   * @brief Performs a Dyson equation time step (function Hamiltonian).
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step (complex Hamiltonian).
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step - returns separate errors for three Keldysh components.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Changes of the three components (each unnormalised as in the class note, H31).
   */
  std::vector<double> dyson_timestep_errinfo(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step without using </\rceil boundary condition (function Hamiltonian) - only useful for debugging.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step without using </\rceil boundary condition (pointer Hamiltonian) - only useful for debugging.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step for two-leg contour (function Hamiltonian).
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a Dyson equation time step for two-leg contour (complex Hamiltonian).
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  /**
   * @brief Performs a Matsubara axis Dyson equation step (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param fixHam Fix Hamiltonian flag - if true, G_0 is not recomputed (default: false). G_0 and its convolution matrix
   *               are then reused from the last fixHam=false call on this dyson object, so such a call (with the same
   *               mu and H) must have been made first.
   * @param alpha Self-consistent iteration step-size - G^M = alpha*G^M_{new} + (1-alpha)*G^M_{old} (default: 0.1)
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_mat(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, bool fixHam=false, double alpha=0.1);
  /**
   * @brief Performs a Matsubara axis Dyson equation step (Matrix Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param hmf Mean-field Hamiltonian matrix.
   * @param Sigma Self-energy object.
   * @param fixHam Fix Hamiltonian flag - if true, G_0 is not recomputed (default: false). G_0 and its convolution matrix
   *               are then reused from the last fixHam=false call on this dyson object, so such a call (with the same
   *               mu and H) must have been made first.
   * @param alpha Self-consistent iteration step-size - G^M = alpha*G^M_{new} + (1-alpha)*G^M_{old} (default: 0.1)
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_mat(herm_matrix_hodlr &G, double mu, DMatrix &hmf, herm_matrix_hodlr &Sigma, bool fixHam=false, double alpha=0.1);

  /**
   * @brief Performs a retarded axis Dyson equation step.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_ret(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  /**
   * @brief Performs a mixed axis Dyson equation step.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_tv(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs imaginary time convolution appearing in Mixed KBE:
   *        res[a][i][j] += sum_{b,k} C_{ajbk} Sigma^tv_{ik}(tstp,tau_b) with C = G.GMConvTens().
   * Reads row tstp of Sigma.tv directly (one Eigen Tensor contraction; no transposed copy since WP6c).
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array (r x n x n), incremented.
   */
  void tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  /**
   * @brief Performs imaginary time convolution appearing in Mixed KBE - output only single tau point m
   *        (res[i][j] += sum_{b,k} C_{mjbk} Sigma^tv_{ik}(tstp,tau_b)).
   * (The raw-pointer overloads tv_it_conv(cplx*, cplx*, double*) and tv_it_conv(int, cplx*, cplx*, double*), which
   *  took Sigma in the removed orbital-transposed tv_trans layout, were removed in WP6c, ISSUES H27.)
   * @param m Time slice index.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array (n x n), incremented.
   */
  void tv_it_conv(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  /**
   * @brief Performs real-time integral appearing in Mixed KBE.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param I Integration object.
   * @param h Time step size.
   */
  void tv_ret_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, Integration::Integrator &I, double h);

  /**
   * @brief Performs a lesser component Dyson equation step.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_les(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a lesser component Dyson equation step without using Mixed component BC - for debugging.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_les_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs a lesser component Dyson equation step for two-leg contour.
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_timestep_les_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs imaginary time integral appearing in lesser KBE, for t = 0..tstp:
   *        I_{ji}(t) = -i int_0^beta dtau Sigma^rceil_{jk}(t,tau) G^lceil_{ki}(tau,tstp).
   * The result is transposed in ij: res[t][i][j] += I_{ji}(t). Reads Sigma.tv rows 0..tstp and G.tv row tstp
   * directly (one Eigen Tensor contraction over all t; no transposed copy since WP6c).
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array ((tstp+1) x n x n), incremented.
   */
  void les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);
  /**
   * @brief Performs les_it_int for a single time slice t: res[i][j] += I_{ji}(t) (n x n, transposed, incremented).
   * @param t Time slice index.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array.
   */
  void les_it_int(int t, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);
  /**
   * @brief Performs lesser-advanced integral appearing in Lesser KBE.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param h Time step size.
   * @param I Integration object.
   */
  void les_lesadv_int_0_tstp(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double h, Integration::Integrator &I);

  /**
   * @brief Performs bootstrapping for Dyson equation solution - assumes steady-state for first k timesteps (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for Dyson equation solution - assumes steady-state for first k timesteps (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for Dyson equation solution with non-time-translation-invariant Hamiltonian (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for Dyson equation solution with non-time-translation-invariant Hamiltonian (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for Dyson equation solution with non-time-translation-invariant Hamiltonian and no les/mix boundary condition - only useful for debugging (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for Dyson equation solution with non-time-translation-invariant Hamiltonian and no les/mix boundary condition - only useful for debugging (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for Dyson equation solution for two-leg contour (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for Dyson equation solution for two-leg contour (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for retarded component Dyson equation solution - assumes steady-state for first k timesteps (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */   
  double dyson_start_ret(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for retarded component Dyson equation solution - assumes steady-state for first k timesteps (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ret(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for retarded axis Dyson equation solution with non-time-translation-invariant boundary conditions (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for retarded axis Dyson equation solution with non-time-translation-invariant boundary conditions (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @param imp_tp0 Use improved boundary condition at t=0 (default: false).
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0 = false);
  /**
   * @brief Performs bootstrapping for mixed component Dyson equation solution (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_tv(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for mixed component Dyson equation solution (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_tv(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution - assumes steady-state for first k timesteps (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution - assumes steady-state for first k timesteps (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution with non-time-translation-invariant boundary conditions (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution with non-time-translation-invariant boundary conditions (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution with no boundary condition (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution with no boundary condition (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution for two-leg contour (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs bootstrapping for lesser axis Dyson equation solution for two-leg contour (complex Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param Sigma Self-energy object.
   * @param I Integration object.
   * @param h Time step size.
   * @return Change of the solution made by this call (unnormalised sum of Frobenius norms; see the class note, H31).
   */
  double dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);


  /**
   * @brief Performs I^<(t,t).
   * @param tstp Time step index.
   * @param G Green's function object.
   * @param Delta Object being convolved with G.
   * @param h Time step size.
   * @param I Integration object.
   * @param res Output array.
   */
  void gamma_integral(int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res);
  /**
   * @brief Performs Matsubara version of I^<(t,t).
   * @param G Green's function object.
   * @param Delta Object being convolved with G.
   * @param h Time step size.
   * @param I Integration object.
   * @param res Output array.
   */
  void gamma_integral_mat(herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res);
};

} // namespace

#endif // header guard
