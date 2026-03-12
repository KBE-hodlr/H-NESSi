
/**
 * @file selfene.hpp
 * @brief Self-energy calculation utilities and Born approximation class for Green's function computations.
 *
 * Provides functions and a class for computing self-energies from Green's functions using FFTW and MPI-based parallelism.
 */
#pragma once

#include <iostream>
#include <fftw3.h>
#include "hodlr/mpi_comm.hpp"


/**
 * @brief Computes the Lesser and Retarded self-energy in r space from the Lesser and Greater Green's functions in r space.
 * @param U Interaction strength
 * @param GLGr Input Green's function matrix
 * @return Computed self-energy matrix
 */
std::vector<std::vector<std::complex<double>>> GLGr_to_SigmaLRr(double U, std::vector<std::vector<std::complex<double>>> &GLGr);

/**
 * @brief Computes the Matsubara self-energy in r space from the Matsubara Green's function in r space.
 * @param U Interaction strength
 * @param GMatr Input Green's function matrix
 * @return Computed self-energy matrix
 */
std::vector<std::vector<std::complex<double>>> GMatr_to_SigmaMatr(double U, std::vector<std::vector<std::complex<double>>> &GMatr);

/**
 * @brief Computes the tv self-energy in r space from the tv and vt Green's functions in r space.
 * @param U Interaction strength
 * @param Gtvr Input Green's function matrix
 * @return Computed self-energy matrix
 */
std::vector<std::vector<std::complex<double>>> Gtvr_to_Sigmatvr(double U, std::vector<std::vector<std::complex<double>>> &Gtvr);

/**
 * @class born_approx_se
 * @brief Implements the Born approximation for self-energy calculations in Green's function methods.
 *
 * Provides methods for FFT-based transformations and self-energy calculations. For each self-energy calculation, two variants are provided:
 *   - Methods with the _spawn suffix: spawn OpenMP threads at each function call (intended for use in single-threaded regions).
 *   - Methods with the _nospawn suffix: do not spawn threads at each call (intended for use when the caller already manages threading).
 * Uses user-provided transformation lambdas for flexible self-energy evaluation.
 */
class born_approx_se{
  public:

  /**
   * @brief Default constructor.
   */
  born_approx_se();

  /**
   * @brief Destructor.
   */
  ~born_approx_se();

  /**
   * @brief Parameterized constructor for Born approximation setup.
   * @param U Interaction strength
   * @param L Orbital dimension
   * @param Nk Number of k-points
   * @param nao number of atomic orbitals
   * @param nthreads Number of OpenMP threads to use
   */
  born_approx_se( double U, int L, int Nk, int nao, int nthreads);

  /**
   * @brief Cleans up FFTW resources and internal buffers.
   */
  void cleanup();

  /**
   * @brief Estimate memory used by this instance (heap allocations) in MiB.
   *
   * Estimates the memory held by internal std::vector buffers and FFTW-allocated arrays
   * using container capacities. Returns a human-readable value in mebibytes (MiB).
   * Note: this is an approximation and does not include allocator overhead or
   * transient allocations performed inside methods.
   * @return Memory usage in MiB as a double
   */
  double memory_usage() const;

  /**
   * @brief Performs FFT from k-space to r-space for Green's function.
   * @param Gk Input Green's function in k-space
   * @param Gk_to_Gr FFTW plan for transformation
   * @return Green's function in r-space
   */
  std::vector<std::complex<double>> fft_Green(std::vector<std::complex<double>> &Gk, const fftw_plan &Gk_to_Gr);

  /**
   * @brief Performs FFT from r-space to k-space for self-energy.
   * @param Sigmar Input self-energy in r-space
   * @param Sigmar_to_Sigmak FFTW plan for transformation
   * @return Self-energy in k-space
   */
  std::vector<std::complex<double>> fft_Sigma(std::vector<std::complex<double>> &Sigmar,const fftw_plan &Sigmar_to_Sigmak);

  /**
   * @brief Calculates self-energy in imaginary time domain (spawning threads at each function call).
   * @param my_Ntau Number of tau points
   * @param comm MPI communicator
   * @param getsSize Size of get functions
   * @param setsSize Size of set functions
   * @param Gr_to_Sigmar Lambda for transformation from Gr to Sigmar
   */
  void Sigma_tau_spawn
  ( int my_Ntau,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  /**
   * @brief Calculates self-energy in imaginary time domain (without spawning threads at each call).
   * @param my_Ntau Number of tau points
   * @param comm MPI communicator
   * @param getsSize Size of get functions
   * @param setsSize Size of set functions
   * @param Gr_to_Sigmar Lambda for transformation from Gr to Sigmar
   */
  void Sigma_tau_nospawn
  ( int my_Ntau,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  /**
   * @brief Calculates self-energy in real time domain (spawning threads at each function call).
   * @param my_Ntp Number of t points
   * @param comm MPI communicator
   * @param getsSize Size of get functions
   * @param setsSize Size of set functions
   * @param Gr_to_Sigmar Lambda for transformation from Gr to Sigmar
   */
  void Sigma_t_spawn
  ( int my_Ntp,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  /**
   * @brief Calculates self-energy in real time domain (without spawning threads at each call).
   * @param my_Ntp Number of t points
   * @param comm MPI communicator
   * @param getsSize Size of get functions
   * @param setsSize Size of set functions
   * @param Gr_to_Sigmar Lambda for transformation from Gr to Sigmar
   */
  void Sigma_t_nospawn
  ( int my_Ntp,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  /**
   * @brief Calculates self-energy for a given time step (spawning threads at each function call).
   * @param tstp Time step
   * @param comm MPI communicator
   * @param getsMat get functions for Matsubara components
   * @param getstv get functions for tv and vt components
   * @param getsLG get functions for Lesser and Greater components
   * @param setsMat set functions for Matsubara components
   * @param setstv set functions for tv and vt components
   * @param setsLG set functions for Lesser and Greater components
   */
  void Sigma_spawn
  ( int tstp,
    mpi_comm &comm,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsMat,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getstv,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsMat,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLG
  );

  /**
   * @brief Calculates self-energy for a given time step (without spawning threads at each call).
   * @param tstp Time step
   * @param comm MPI communicator
   * @param getsMat get functions for Matsubara components
   * @param getstv get functions for tv and vt components
   * @param getsLG get functions for Lesser and Greater components
   * @param setsMat set functions for Matsubara components
   * @param setstv set functions for tv and vt components
   * @param setsLG set functions for Lesser and Greater components
   */
  void Sigma_nospawn
  ( int tstp,
    mpi_comm &comm,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsMat,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getstv,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsMat,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLG
  );



  int L_;
  int Nk_;
  double U_;
  int max_component_size_;
  std::complex<double> tmp;
  std::vector<std::vector<std::complex<double>>> Gk;
  std::vector<std::vector<std::complex<double>>> Gr;
  std::vector<std::vector<std::complex<double>>> Sigmar;
  std::vector<std::vector<std::complex<double>>> Sigmak;

  fftw_complex *in_dummy, *out_dummy;
  fftw_plan Gk_to_Gr, Sigmar_to_Sigmak;

  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> lambda_GLGr_to_SigmaLRr;
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> lambda_Gtvr_to_Sigmatvr;
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> lambda_GMatr_to_SigmaMatr;

  //omp thread safety
  std::vector<fftw_plan> G_plans_vec, Sigma_plans_vec;
  std::vector<std::vector<std::vector<std::complex<double>>>> Gk_thread_vec;
  std::vector<std::vector<std::vector<std::complex<double>>>> Gr_thread_vec;
  std::vector<std::vector<std::vector<std::complex<double>>>> Sigmar_thread_vec;
  std::vector<std::vector<std::vector<std::complex<double>>>> Sigmak_thread_vec;

};
