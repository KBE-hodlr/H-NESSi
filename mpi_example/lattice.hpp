
/**
 * @file lattice.hpp
 * @brief Defines the lattice_2d_ysymm class for 2D lattice models with y-symmetry, including k-space and Hamiltonian utilities.
 */
#pragma once

#include <complex>

#include "hodlr/herm_matrix_hodlr.hpp"

using Vector2D = Eigen::Vector2d;


/**
 * @class lattice_2d_ysymm
 * @brief 2D lattice with y-symmetry: handles k-space grid, Hamiltonian, and velocity matrix construction.
 *
 * Provides methods for initializing k-points, converting electric field to vector potential, constructing the single-particle Hamiltonian,
 * and computing velocity matrices. Supports mapping between flat and 2D k-indices.
 */
class lattice_2d_ysymm{

public:
  int Nt_;
  double dt_;
  int L_;
  int Nk_;

  std::vector<Vector2D> kpoints_;

  int size_;
  double mu_;
  double U_;
  double J_;
  std::vector<Vector2D> A_;


  /**
   * @brief Default constructor.
   */
  lattice_2d_ysymm(void);

  /**
   * @brief Constructs and initializes the lattice with given parameters.
   * @param L Lattice size (number of sites per dimension)
   * @param Nt Number of time steps
   * @param dt Time step size
   * @param J Hopping amplitude
   * @param Amax Electric field intensity
   * @param mu Chemical potential
   * @param size Matrix size for Hamiltonian/velocity
   */
  lattice_2d_ysymm(int L, int Nt, double dt, double J, double Amax, double mu, int size);

  /**
   * @brief Returns the flat index for k-space coordinates (kxi, kyi).
   * @param kxi kx index
   * @param kyi ky index
   * @return Flat index in k-space array
   */
  int kflatindex(int kxi, int kyi);

  /**
   * @brief Returns the (kxi, kyi) indices for a given flat k-space index.
   * @param index Flat index
   * @return Array with [kxi, kyi]
   */
  std::array<int,2> kxikyi(int index);

  /**
   * @brief Initializes the k-space grid (k-points) for the lattice.
   * @param L Lattice size
   */
  void init_kk(int L);

  /**
   * @brief Constructs the single-particle Hamiltonian matrix for a given k-point and time step.
   * @param hkmatrix Output Hamiltonian matrix
   * @param tstp Time step index
   * @param kk k-point vector
   * @param Ainitx Initial vector potential in x direction
   */
  void hk(hodlr::ZMatrix &hkmatrix,int tstp, Vector2D& kk, double Ainitx);

  /**
   * @brief Converts a constant electric field to a vector potential for all time steps.
   * @param Nt Number of time steps
   * @param dt Time step size
   * @param Amax Electric field intensity
   */
  void efield_to_afield(int Nt, double dt, double Amax);

  /**
   * @brief Constructs the velocity matrices for a given k-point and time step.
   * @param vxkmatrix Output velocity matrix in x direction
   * @param vykmatrix Output velocity matrix in y direction
   * @param tstp Time step index
   * @param kk k-point vector
   */
  void vk(hodlr::ZMatrix &vxkmatrix, hodlr::ZMatrix &vykmatrix, int tstp, Vector2D& kk);

};

