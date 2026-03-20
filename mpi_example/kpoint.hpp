#pragma once
/**
 * @file kpoint.hpp
 * @brief Defines the kpoint class for managing k-space points and solving Dyson equations.
 */

#include "lattice.hpp"
#include "h_nessi/dyson.hpp"
/**
 * @class kpoint
 * @brief Represents a k-space point and provides methods for solving Dyson equations and managing Green's functions and Self-energy.
 *
 * The kpoint class handles:
 * - Initialization of Green's functions and self-energies for a given k-point.
 * - Construction of the single-particle Hamiltonian for a given k-point.
 * - Computation of the density matrix.
 * - Solving Dyson equations for time evolution.
 */
class kpoint{
public:
    /**
     * @brief Default constructor.
     */
    kpoint();

    /**
     * @brief Destructor.
     */
    ~kpoint();

    /**
     * @brief Constructs a kpoint with specified parameters.
     * @param nt Number of time steps
     * @param r Number of representative tau points
     * @param nlvl Number of hierarchical levels
     * @param svdtol SVD tolerance
     * @param size Matrix size
     * @param beta Inverse temperature
     * @param dt Time step size
     * @param SolverOrder Solver order for Dyson equation
     * @param kk k-point vector
     * @param latt Reference to the lattice object
     * @param mu Chemical potential
     * @param Ainitx Initial vector potential in x direction
     */
    kpoint(int nt,
        int r,
        int nlvl,
        double svdtol,
        int size, 
        double beta, 
        double dt, 
        int SolverOrder,
        Vector2D& kk, 
        lattice_2d_ysymm &latt, 
        double mu,
        double Ainitx);

    /**
     * @brief Constructs a kpoint with checkpoint data.
     * @param nt Number of time steps
     * @param r Number of representative tau points
     * @param nlvl Number of hierarchical levels
     * @param svdtol SVD tolerance
     * @param size Matrix size
     * @param beta Inverse temperature
     * @param dt Time step size
     * @param SolverOrder Solver order for Dyson equation
     * @param kk k-point vector
     * @param latt Reference to the lattice object
     * @param mu Chemical potential
     * @param Ainitx Initial vector potential in x direction
     * @param checkpoint_file HDF5 file for checkpoint data
     */
    kpoint(int nt,
        int r,
        int nlvl,
        double svdtol,
        int size, 
        double beta, 
        double dt, 
        int SolverOrder,
        Vector2D& kk, 
        lattice_2d_ysymm &latt, 
        double mu,
        double Ainitx,
        h5e::File &checkpoint_file);

    /**
     * @brief Sets the single-particle Hamiltonian for a given time step.
     * @param tstp Time step index
     * @param latt Reference to the lattice object
     * @param Ainitx Initial vector potential in x direction
     */
    void set_hk(int tstp, lattice_2d_ysymm &latt, double Ainitx);

    /**
     * @brief Computes the density matrix for a given time step.
     * @param tstp Time step index
     * @param dlr Reference to the DLR information object
     */
    void get_Density_matrix(int tstp, h_nessi::dlr_info &dlr);

    /**
     * @brief Solves the Dyson equation for a given time step.
     * @param tstp Time step index
     * @param SolverOrder Solver order for Dyson equation
     * @param latt Reference to the lattice object
     * @param I Integration object for Dyson equation
     * @param dyson_sol Dyson solver object
     * @param dlr Reference to the DLR information object
     * @return Squared error of the Dyson solution
     */
    double step_dyson(int tstp, int SolverOrder, lattice_2d_ysymm &latt, Integration::Integrator &I, h_nessi::dyson &dyson_sol, h_nessi::dlr_info &dlr);

    double beta_; ///< Inverse temperature
    double dt_; ///< Time step size
    double svdtol_; ///< SVD tolerance
    int nt_; ///< Number of time steps
    int r_; ///< Number of representative tau points
    int nlvl_; ///< Number of hierarchical levels
    int size_; ///< Matrix size

    h_nessi::function hk_; ///< Single-particle Hamiltonian as a function of time
    std::vector<h_nessi::DMatrix> rho_; ///< Density matrices for each time step
    h_nessi::herm_matrix_hodlr G_, Sigma_; ///< Green's function and self-energy matrices

    double mu_; ///< Chemical potential
    Vector2D kk_; ///< k-point vector
};
