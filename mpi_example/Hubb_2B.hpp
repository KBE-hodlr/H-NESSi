#pragma once

#include <vector>
#include <functional>
#include <complex>
#include <fftw3.h>
#include <omp.h>

#include "hodlr/mpi_comm.hpp"
#include "hodlr/dlr.hpp"
#include "hodlr/herm_matrix_hodlr.hpp"

/**
 * @class Hubb_2B
 * @brief Implements the Second Born (2B) self-energy for the Hubbard model
 * using optimized FFTW executions and thread-local buffering.
 */
class Hubb_2B {
public:
    /**
     * @brief Constructor
     * @param U Interaction strength
     * @param L Linear dimension of the square lattice
     * @param Nk Number of irreducible k-points
     * @param nthreads Number of OpenMP threads for local execution
     */
    Hubb_2B(double U, int L, int nthreads);

    /**
     * @brief Destructor - handles FFTW plan destruction and memory cleanup
     */
    ~Hubb_2B();

    /**
     * @brief High-level dispatcher for self-energy calculation.
     * Manages MPI communication and routes to specific time-domain branches.
     */
    void Sigma_spawn(int tstp, 
                     mpi_comm &comm, 
                     std::vector<std::reference_wrapper<hodlr::herm_matrix_hodlr>> &Grefs, 
                     std::vector<std::reference_wrapper<hodlr::herm_matrix_hodlr>> &Srefs, 
                     hodlr::dlr_info &dlr);

    /**
     * @brief High-level dispatcher for self-energy calculation.
     * Manages MPI communication and routes to specific time-domain branches.
     */
    void Sigma_nospawn(int tstp, 
                     mpi_comm &comm, 
                     std::vector<std::reference_wrapper<hodlr::herm_matrix_hodlr>> &Grefs, 
                     std::vector<std::reference_wrapper<hodlr::herm_matrix_hodlr>> &Srefs, 
                     hodlr::dlr_info &dlr);
private:
    // Constants
    double U_; 
    int L_;
    int Nk_;
    int nthreads_;

    // Shared FFTW plans (Thread-safe for execution)
    fftw_plan Gk_to_Gr;
    fftw_plan Sigmar_to_Sigmak;

    // Pre-allocated thread-local buffers (fftw_malloc'd for SIMD alignment)
    std::vector<fftw_complex*> in_thread_vec;
    std::vector<fftw_complex*> out_thread_vec;

    // Internal Evaluation Branches
    /**
     * @brief Matsubara evaluation branch.
     */
    void Sigma_Mat_spawn(mpi_comm &comm); 

    /**
     * @brief Real-time evaluation branch.
     */
    void Sigma_Real_spawn(int tstp, 
                          mpi_comm &comm);
    void Sigma_Real_tv_spawn(int tstp, 
                          mpi_comm &comm);
    void Sigma_Real_lesret_spawn(int tstp, 
                          mpi_comm &comm);

    // Internal Evaluation Branches
    /**
     * @brief Matsubara evaluation branch.
     */
    void Sigma_Mat_nospawn(mpi_comm &comm); 

    /**
     * @brief Real-time evaluation branch.
     */
    void Sigma_Real_nospawn(int tstp, 
                          mpi_comm &comm);
    void Sigma_Real_tv_nospawn(int tstp, 
                          mpi_comm &comm);
    void Sigma_Real_lesret_nospawn(int tstp, 
                          mpi_comm &comm);
};
