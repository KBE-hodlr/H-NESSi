
/**
 * @file lattice.hpp
 * @brief Defines the lattice_2d_ysymm class for 2D lattice models with y-symmetry, including k-space and Hamiltonian utilities.
 */
#pragma once

#include <complex>

#include "h_nessi/herm_matrix_hodlr.hpp"

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
  // Constant initial vector potential (x) added to A(t) in hk and vk (input __Ainitx=; set by Main after construction).
  double Ainitx_ = 0.0;


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
  void hk(h_nessi::ZMatrix &hkmatrix,int tstp, Vector2D& kk, double Ainitx);

  /**
   * @brief Converts a constant electric field to a vector potential for all time steps.
   * @param Nt Number of time steps
   * @param dt Time step size
   * @param Amax Electric field intensity
   */
  void efield_to_afield(int Nt, double dt, double Amax);

  /**
   * @brief Replaces A(t) by a smooth pulse that is fully contained in (t_start, t_start + T_pulse) (input __pulse= smooth):
   *        A_x(t) = Amax sin^npow(pi (t - t_start)/T_pulse) sin(omega (t - t_start)) inside, 0 outside; A_y = 0.
   *        A vanishes at both ends like x^npow (x^(npow+1) if omega*T_pulse is a multiple of pi), so A is C^(npow-1)
   *        (C^npow) there; E(t) = -dA/dt is continuous for npow >= 2. Used for the energy-balance test.
   *        The smoothness at the ends limits the observable convergence order of the balance (WP8b).
   * @param Nt Number of time steps
   * @param dt Time step size
   * @param Amax Amplitude
   * @param t_start Start of the pulse
   * @param T_pulse Duration of the pulse
   * @param omega Carrier frequency (0: A_x = Amax sin^npow, a unipolar A, i.e. a single-cycle field)
   * @param npow Power of the sin envelope (>= 2; default 2)
   */
  void smooth_pulse_afield(int Nt, double dt, double Amax, double t_start, double T_pulse, double omega, int npow = 2);

  /**
   * @brief Constructs the velocity matrices for a given k-point and time step.
   * @param vxkmatrix Output velocity matrix in x direction
   * @param vykmatrix Output velocity matrix in y direction
   * @param tstp Time step index
   * @param kk k-point vector
   */
  void vk(h_nessi::ZMatrix &vxkmatrix, h_nessi::ZMatrix &vykmatrix, int tstp, Vector2D& kk);

  /**
   * @brief Band energy eps_{k - A(t) - Ainit} WITHOUT the chemical potential (the dispersion of paper Eq. 25 / 27).
   * @param tstp Time step index (-1: equilibrium, A = 0)
   * @param kk k-point vector
   */
  double epsk_band(int tstp, Vector2D& kk);

};

