
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

#include "function.hpp"
#include "herm_matrix_hodlr.hpp"
#include "integration.hpp"
#include "utils.hpp"
#include "dlr.hpp"

extern "C"
{
  #include "dlr_c/dlr_c.h"
}



/**
 * @namespace hodlr
 * @brief Namespace for hierarchical matrix algorithms and quantum many-body data structures.
 */
namespace hodlr {

/**
 * @class dyson
 * @brief Solves quantum many-body Dyson equations using HODLR and DLR representations.
 *
 * The dyson class manages all routines for constructing and evolving Green's functions, performing time-stepping, bootstrapping, and convolutions in quantum many-body calculations. It supports both fermionic and bosonic statistics, and interfaces with hierarchical matrix and DLR basis representations.
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
  std::vector<cplx> Q_;               /**< storage for timeslice quantities such as history integrals - size (nt+1)*es*/
  std::vector<cplx> M_;               /**< storage for Matrix which is solved when timestepping - size k*k*es */
  std::vector<cplx> X_;               /**< storage for timeslice quantities such as the G being solved for  - size (nt+1)*es */
  std::vector<cplx> epsnao_tmp_;      /**< storage for intermediate quantities when applying U,S,V sequentially - size (nt+1)/2*nao */
  std::vector<cplx> epsnao_tmp_2_;    /**< storage for intermediate quantities when applying U,S,V sequentially - size (nt+1)*/
  std::vector<cplx> iden_;            /**< Identity matrix storage. */
  std::vector<cplx> bound_;           /**< storage for matrices when we need to build them from compressed representations - size nao*nao */

  /**< Timing information for profiling. 
   * 0 - Retarded component
   * 1 - Retarded integral
   * 2 - current timestep region of integral
   * 3 - block applications of integral
   * 4 - direct region of integral
   * 5 - unused
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
  */
  bool profile_;
  DMatrix timing;         

public:

  /**
   * @brief Constructs a dyson object with specified parameters.
   * @param nt Number of time steps.
   * @param nao Number of atomic orbitals.
   * @param k Maximum order for time-stepping.
   * @param dlr Reference to DLR information object.
   * @param rho_version Density matrix calculation version (default: 1).
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
   * @brief Provides a small report on memory usage
   */
  void print_memory_usage();

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
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian (complex).
   * @param rho initial Density matrix (complex).
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Vector of error estimates.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
   */
  double dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);

  /**
   * @brief Performs a Matsubara axis Dyson equation step (function Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param H Hamiltonian function.
   * @param Sigma Self-energy object.
   * @param fixHam Fix Hamiltonian flag - if true, G_0 is not recomputed (default: false).
   * @param alpha Self-consistent iteration step-size - G^M = alpha*G^M_{new} + (1-alpha)*G^M_{old} (default: 0.1)
   * @return Error estimate.
   */
  double dyson_mat(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, bool fixHam=false, double alpha=0.1);
  /**
   * @brief Performs a Matsubara axis Dyson equation step (Matrix Hamiltonian).
   * @param G Green's function object.
   * @param mu Chemical potential.
   * @param hmf Mean-field Hamiltonian matrix.
   * @param Sigma Self-energy object.
   * @param fixHam Fix Hamiltonian flag - if true, G_0 is not recomputed (default: false).
   * @param alpha Self-consistent iteration step-size - G^M = alpha*G^M_{new} + (1-alpha)*G^M_{old} (default: 0.1)
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
   */
  double dyson_timestep_tv(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs imaginary time convolution appearing in Mixed KBE.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array.
   */
  void tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  /**
   * @brief dispatch function for tv_it_conv.
   * @param Sigma Self-energy array - must be transposed version.
   * @param res Output array.
   * @param GMConvTens Convolution tensor.
   */
  void tv_it_conv(cplx *Sigma, cplx *res,double *GMConvTens);
  /**
   * @brief Performs imaginary time convolution appearing in Mixed KBE - output only single tau point.
   * @param m Time slice index.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array.
   */
  void tv_it_conv(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res);
  /**
   * @brief dispatch function for tv_it_conv.
   * @param m Time slice index.
   * @param Sigma Self-energy array - must be transposed version.
   * @param res Output array.
   * @param GMConvTens Convolution tensor.
   */
  void tv_it_conv(int m, cplx *Sigma, cplx *res,double *GMConvTens);
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
   */
  double dyson_timestep_les_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h);
  /**
   * @brief Performs imaginary time integral appearing in lesser KBE.
   * @param tstp Time step index.
   * @param Sigma Self-energy object.
   * @param G Green's function object.
   * @param res Output array.
   */
  void les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res);
  /**
   * @brief Performs les_it_int for a single time slice (block version).
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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
   * @return Error estimate.
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

} // namespace hodlr

#endif // header guard
