// This class is a very general template for performing the HF and GF2 contractions for general Hamiltonians
// There are functions that allow for the evaluation of self-energies for spatially-invariant systems, as well as for non-invariant systems
// Spatial-invariant functions allow for general number of orbitals in each Unit cell
// These contractions scale as N_s^5 N_b^5, where N_s is number of real-space sites, and N_b is number of orbitals at each site
// Specialized functions for spatial-invariant systems scale as N_s^4 N_b^5
// Much more efficient methods can be implemented, especially for Hubbard interactions, or density-density interactions, but this should serve as a baseline reference to test these specialized implementations against
// Notes describing the functions implemented here can be found in .../notes/nsite_nband_contraction_notes.pdf

#ifndef SRSU_GF2_DECL
#define SRSU_GF2_DECL

#include "utils.hpp"
#include "herm_matrix_hodlr.hpp"
#include "distributed_single_timeslice.hpp"

namespace hodlr {

class nband_nsite_SRSU_gf2 {
private:
  int Nb_;
  int Nb2_;
  int Nb3_;
  int Nb4_;
  int Ns_;
  int Ns2_;
  int Ns3_;
  int N_;
  int Ntau_;
  
  ZMatrix Git_rev_Tnn_;
  ZMatrix tmp_nn_;
  ZMatrix tmp2_nn_;
  ZMatrix tmp_nnn_;
  ZMatrix tmp2_nnn_;

  // When doing full matrix calculation (real or k does not matter) the arguments must be interwoven, (b1,s1,b2,s2,b3,s3,b3,s3) or (s1,b1,s2,b2,s3,b3,s4,b4)
  // When doing block-diagonal calculation the arguments must be (s1,s2,s3,s4,b1,b2,b3,b4)
  ZMatrix &U_tot_nnnn_;
  ZMatrix &U_SU_nnnn_;
  ZMatrix &U_SUEX_nnnn_;

  // General GF2 contractions
  void gf2_contract_deploy(cplx *res, cplx *U1, cplx *U2, cplx *G1, cplx *G2, cplx *G3);
  void gf2_contract_deploy(double *res, cplx *U1, cplx *U2, double *G1, double *G2, double *G3);

  // General HF contractions
  void hf_contract_deploy(cplx *res, cplx *U, cplx *rho_trans);


  // GF2 for Green's functions
  void Mat_gf2_contract(herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset, double *it2itr);
  void Ret_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset);
  void Les_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset);
  void TV_gf2_contract (int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset, double *it2itr);

  // MPI GF2 for Green's functions
  void Mat_gf2_contract(herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset, double *it2itr);
  void Ret_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset);
  void Les_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset);
  void TV_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset, double *it2itr);


public:

  // Constructor
  nband_nsite_SRSU_gf2(int Ns, int Nb, int Ntau, ZMatrix &U_tot, ZMatrix &U_SU, ZMatrix &U_SUEX);


  // GF2 for Green's functions of full system
  void gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr);
  // HF for Green's functions of full system
  void hf_contract(int tstp, cplx *hmf, herm_matrix_hodlr &G, double *it2cf, int *it2cfp, double *dlrrf, int U_offset = 0);
  void hf_contract(int tstp, cplx *hmf, herm_matrix_hodlr &G, int U_offset = 0);


  // GF2 for vector of Block-diagonal Green's functions - non parallel
  void gf2_contract(int tstp, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G, double *it2itr);
  // HF for vector of Block-diagonal Green's functions - non parallel
  void hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G_vec, double *it2cf, int *it2cfp, double *dlrrf);
  void hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G_vec);


  // GF2 for vector of Block-diagonal Green's functions - parallel
  void gf2_contract(int tstp, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points, double *it2itr);

  // HF for vector of Block-diagonal Green's functions - parallel
  void hf_contract(int tstp, cplx *hmf, distributed_single_timeslice &G, int b, double *it2cf, int *it2cfp, double *dlrrf, int U_offset);
  void hf_contract(int tstp, cplx *hmf, distributed_single_timeslice &G, int b, int U_offset);
  void hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points, double *it2cf, int *it2cfp, double *dlrrf);
  void hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points);
};
}

#endif
