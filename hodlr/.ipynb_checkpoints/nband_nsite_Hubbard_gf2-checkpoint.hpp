#ifndef MPI_HUBB_GF2_DECL
#define MPI_HUBB_GF2_DECL

#include "utils.hpp"
#include "herm_matrix_hodlr.hpp"
#include "distributed_single_timeslice_FT.hpp"

namespace hodlr {

class nband_nsite_Hubbard_gf2 {
private:
  int Nb_;
  int Nb2_;
  int Ns_;
  int Ntau_;

  ZMatrix Git_rev_TnnS_;
  ZMatrix tmp_rnn_;
  ZMatrix tmp2_rnn_;

  double U_;

  Eigen::PermutationMatrix<Eigen::Dynamic> Perm_;

  fftw_plan p_r2k;
  fftw_plan p_k2r;

public:
  nband_nsite_Hubbard_gf2(int Ns, int Nb, int Ntau, double U, std::vector<int>& permutation, int ndim);

  void hf_gf2_contract(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2cf, int *it2cfp, double *dlrrf, double *it2itr, bool eval_hmf = true, bool eval_gf2 = true);
  void hf_gf2_contract_boot(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool eval_hmf = true, bool eval_gf2 = true);
  std::vector<double> hf_gf2_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool eval_hmf = true, bool eval_gf2 = true);
};

}

#endif
