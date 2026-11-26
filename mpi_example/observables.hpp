/**
 * @file observables.hpp
 * @brief Provides functions for calculating local observables in a 2D lattice system.
 */

#pragma once

#include "kpoint.hpp"
#include "h_nessi/mpi_comm_utils.hpp"

namespace observables {

    /**
     * @brief Computes local observables [jx(t),jy(t),Ekin(t),Navg(t)] in parallel for a given time step.
     * @param tstp Time step index
     * @param Nk_rank Number of k-points for the current MPI rank
     * @param lattice Reference to the lattice object
     * @param kindex_rank Mapping of local k-point indices to global indices
     * @param corrK_rank Vector of unique pointers to kpoint objects for the current rank
     * @return Array of four doubles representing the computed observables
     */
    std::array<double,4> get_obs_local
    (int tstp,
     int Nk_rank,
     lattice_2d_ysymm &lattice,
     std::vector<int> &kindex_rank,
     std::vector<std::unique_ptr<kpoint>> &corrK_rank
    );

    /**
     * @brief Weight of a stored k-point in a Brillouin-zone average (1/N_k) sum_k, N_k = L^2: 2/L^2 for 0 < ky_i < L/2
     *        (the point stands for itself and its y-reflected copy, Eq. 31), 1/L^2 otherwise.
     */
    double bz_weight(int global_k, lattice_2d_ysymm &lattice);

    /**
     * @brief Rank-local part of the band energy (paper Eq. 27 with the driven dispersion, spin summed):
     *        E_band(t) = -(2i/N_k) sum_k eps_{k-A(t)} G^<_k(t,t) = (2/N_k) sum_k eps_{k-A(t)} rho_k(t),
     *        eps from lattice.epsk_band (no chemical potential). Summed over the ranks it is a per-site energy.
     */
    double get_energy_band_local
    (int tstp,
     int Nk_rank,
     lattice_2d_ysymm &lattice,
     std::vector<int> &kindex_rank,
     std::vector<std::unique_ptr<kpoint>> &corrK_rank
    );

    /**
     * @brief Computes local observables [jx(t),jy(t),Ekin(t),Navg(t)] without parallelization for a given time step.
     * @param tstp Time step index
     * @param lattice Reference to the lattice object
     * @param corrK_rank Vector of unique pointers to kpoint objects for the current rank
     * @return Array of four doubles representing the computed observables
     */
    std::array<double,4> get_obs_local_no_parallel
    (int tstp,
     lattice_2d_ysymm &lattice,
     std::vector<std::unique_ptr<kpoint>> &corrK_rank
    );
}


