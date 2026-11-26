
/**
 * @file bubble.hpp
 * @brief Declares functions for taking products of Keldysh components
 *
 */
#ifndef BUBBLE_DECL
#define BUBBLE_DECL

#include "utils.hpp"
#include "herm_matrix_hodlr.hpp"

/**
 * @namespace h_nessi
 * @brief Namespace for hierarchical matrix algorithms and data structures.
 */
namespace h_nessi {

  /*
   * Preconditions of Bubble1/Bubble2 (tier 2, h_nessi::precondition_error; doc 61 BU-*): C is a different object from
   * A and B (aliasing is forbidden, audit Q-10); every orbital index is inside its object; A, B and C have the same
   * DLR rank r; tstp == -1 or tstp lies in the current windows [tstpmk, tstpmk+k] of A, B and C (Bubble1 also needs
   * dlr to match B, checked by B's evaluators). Statistics (not checked, doc 61 N9): the products are Langreth
   * products of two-time functions, so C is expected to have the statistics of the product, xi_C = xi_A * xi_B
   * (e.g. a polarisation from two fermionic G is bosonic). Bubble1 takes the Matsubara sign from B (-xi_B).
   * Both write C's tv row tstp; every later reader (get_vt, Bubble1, les_it_int, ...) sees it (no tv_trans cache
   * since WP6c, ISSUES H27). Writing row C.tstp() clears C.can_extrap() (ST-6): a following extrapolate(C) is a no-op.
   */

  /**
   * @brief C_{c1,c2}(t1,t2) = ii * A_{a1,a2}(t1,t2) * B_{b2,b1}(t2,t1)
   */
  void Bubble1(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, int a1, int a2, herm_matrix_hodlr &B, int b1, int b2, dlr_info &dlr); 

  /**
   * @brief C_{c1,c2}(t1,t2) = ii * A_{a1,a2}(t1,t2) * B_{b1,b2}(t1,t2)
   */
  void Bubble2(int tstp, herm_matrix_hodlr &C, int c1, int c2, herm_matrix_hodlr &A, int a1, int a2, herm_matrix_hodlr &B, int b1, int b2); 

}
#endif
