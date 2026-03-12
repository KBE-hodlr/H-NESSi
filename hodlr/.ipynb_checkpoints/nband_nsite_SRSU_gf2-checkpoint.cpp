#ifndef SRSU_GF2_IMPL
#define SRSU_GF2_IMPL

#include "nband_nsite_SRSU_gf2.hpp"

namespace hodlr {

nband_nsite_SRSU_gf2::nband_nsite_SRSU_gf2(int Ns, int Nb, int Ntau, ZMatrix &U_tot, ZMatrix &U_SU, ZMatrix &U_SUEX) :
Nb_(Nb),
Nb2_(Nb*Nb),
Nb3_(Nb*Nb*Nb),
Nb4_(Nb*Nb*Nb*Nb),
Ns_(Ns),
Ns2_(Ns*Ns),
Ns3_(Ns*Ns*Ns),
N_(Ns*Nb),
Ntau_(Ntau),
Git_rev_Tnn_(Ntau, Nb*Nb),
tmp_nn_(Nb,Nb),
tmp2_nn_(Nb,Nb),
tmp_nnn_(Nb,Nb*Nb),
tmp2_nnn_(Nb,Nb*Nb),
U_tot_nnnn_(U_tot),
U_SU_nnnn_(U_SU),
U_SUEX_nnnn_(U_SUEX) {}


// #############################################################################
// #############################################################################
// ###              Deploy functions - actually do contractions              ###
// #############################################################################
// #############################################################################

void nband_nsite_SRSU_gf2::gf2_contract_deploy(cplx *res, cplx *U1, cplx *U2, cplx *G1, cplx *G2, cplx *G3) {
  ZMatrixMap(tmp_nnn_.data(), Nb_, Nb2_) = ZMatrixMap(G3, Nb_, Nb_) * ZMatrixMap(U1, Nb_, Nb2_);
  ZMatrixMap(tmp2_nnn_.data(), Nb_, Nb2_) = (ZMatrixMap(tmp_nnn_.data(), Nb2_, Nb_) * ZMatrixMap(G1, Nb_, Nb_)).transpose();
  ZMatrixMap(tmp_nnn_.data(), Nb2_, Nb_) = ZMatrixMap(tmp2_nnn_.data(), Nb2_, Nb_) * ZMatrixMap(G2, Nb_, Nb_);
  for(int i = 0; i < Nb_; i++) {
    ZMatrixMap(tmp2_nnn_.data() + i*Nb2_,  Nb_, Nb_) = ZMatrixMap(tmp_nnn_.data() + i*Nb2_,  Nb_, Nb_).transpose();
  }
  ZMatrixMap(res, 1, Nb_) += ZMatrixMap(tmp2_nnn_.data(), 1, Nb3_) * ZMatrixMap(U2, Nb3_, Nb_);
}

void nband_nsite_SRSU_gf2::gf2_contract_deploy(double *res, cplx *U1, cplx *U2, double *G1, double *G2, double *G3) {
  ZMatrixMap(tmp_nnn_.data(), Nb_, Nb2_) = DMatrixMap(G3, Nb_, Nb_) * ZMatrixMap(U1, Nb_, Nb2_);
  ZMatrixMap(tmp2_nnn_.data(), Nb_, Nb2_) = (ZMatrixMap(tmp_nnn_.data(), Nb2_, Nb_) * DMatrixMap(G1, Nb_, Nb_)).transpose();
  ZMatrixMap(tmp_nnn_.data(), Nb2_, Nb_) = ZMatrixMap(tmp2_nnn_.data(), Nb2_, Nb_) * DMatrixMap(G2, Nb_, Nb_);
  for(int i = 0; i < Nb_; i++) {
    ZMatrixMap(tmp2_nnn_.data() + i*Nb2_,  Nb_, Nb_) = ZMatrixMap(tmp_nnn_.data() + i*Nb2_,  Nb_, Nb_).transpose();
  }
  DMatrixMap(res, 1, Nb_) += (ZMatrixMap(tmp2_nnn_.data(), 1, Nb3_) * ZMatrixMap(U2, Nb3_, Nb_)).real();
}

void nband_nsite_SRSU_gf2::hf_contract_deploy(cplx *res, cplx *U, cplx *rho_trans) {
  ZMatrixMap(res, 1, Nb_) += ZMatrixMap(rho_trans, 1, Nb2_) * ZMatrixMap(U, Nb2_, Nb_);
}

// #############################################################################
// #############################################################################
// ###             GF2 specialization for each Keldysh component             ###
// #############################################################################
// #############################################################################

void nband_nsite_SRSU_gf2::Mat_gf2_contract(herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset, double *it2itr) {
  assert(Sigma.size1() == Nb_);
  assert(G1.size1() == Nb_);
  assert(G2.size1() == Nb_);
  assert(G3.size1() == Nb_);
  assert(Sigma.ntau() == Ntau_);
  assert(G1.ntau() == Ntau_);
  assert(G2.ntau() == Ntau_);
  assert(G3.ntau() == Ntau_);

  double *double_data_ptr_Git_rev = reinterpret_cast<double*>(Git_rev_Tnn_.data());

  G3.get_mat_reversed(double_data_ptr_Git_rev, it2itr);

  for(int t = 0; t < Ntau_; t++) {
    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(Sigma.matptr(t) + i*Nb_, U_tot_nnnn_.data()+U1_offset+i*Nb3_, U_tot_nnnn_.data()+U2_offset, G1.matptr(t), G2.matptr(t), double_data_ptr_Git_rev + t*Nb2_);
      gf2_contract_deploy(Sigma.matptr(t) + i*Nb_, U_SU_nnnn_.data()+U1_offset+i*Nb3_, U_SUEX_nnnn_.data()+U2_offset, G1.matptr(t), G2.matptr(t), double_data_ptr_Git_rev + t*Nb2_);
    }
  }
}

void nband_nsite_SRSU_gf2::Les_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset) {
  assert(Sigma.size1() == Nb_);
  assert(G1.size1() == Nb_);
  assert(G2.size1() == Nb_);
  assert(G3.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k());
  assert(G1.tstpmk() + G1.k() == tstp or tstp <= Sigma.k());
  assert(G2.tstpmk() + G2.k() == tstp or tstp <= Sigma.k());
  assert(G3.tstpmk() + G3.k() == tstp or tstp <= Sigma.k());

  cplx *G1_les_ptr = G1.curr_timestep_les_ptr(0,tstp);
  cplx *G2_les_ptr = G2.curr_timestep_les_ptr(0,tstp);
  cplx *G3_les_ptr = G3.curr_timestep_les_ptr(0,tstp);
  cplx *G3_ret_ptr = G3.curr_timestep_ret_ptr(tstp,0);
  cplx *S_les_ptr = Sigma.curr_timestep_les_ptr(0,tstp);

  cplx *tmp_data = tmp_nn_.data();

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  for(int t = 0; t <= tstp; t++) {
    // G^>(T,t) = G^R(T,t) + G^<(T,t) = G^R(T,t) - G^<(T,t)^\dagger
    ZMatrixMap(tmp_data, Nb_, Nb_) = ZMatrixMap(G3_ret_ptr, Nb_, Nb_) - ZMatrixMap(G3_les_ptr, Nb_, Nb_).adjoint();

    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_les_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data+U2_offset, G1_les_ptr, G2_les_ptr, tmp_data);
      gf2_contract_deploy(S_les_ptr + i*Nb_, U_SU_data+U1_offset+i*Nb3_, U_SUEX_data+U2_offset, G1_les_ptr, G2_les_ptr, tmp_data);
    }

    G1_les_ptr += Nb2_;
    G2_les_ptr += Nb2_;
    G3_les_ptr += Nb2_;
    G3_ret_ptr += Nb2_;
    S_les_ptr += Nb2_;
  }
}

void nband_nsite_SRSU_gf2::Ret_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset) {
  assert(Sigma.size1() == Nb_);
  assert(G1.size1() == Nb_);
  assert(G2.size1() == Nb_);
  assert(G3.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k() );
  assert(G1.tstpmk() + G1.k() == tstp or tstp <= Sigma.k());
  assert(G2.tstpmk() + G2.k() == tstp or tstp <= Sigma.k());
  assert(G3.tstpmk() + G3.k() == tstp or tstp <= Sigma.k());

  cplx *G1_les_ptr = G1.curr_timestep_les_ptr(0,tstp);
  cplx *G1_ret_ptr = G1.curr_timestep_ret_ptr(tstp,0);
  cplx *G2_les_ptr = G2.curr_timestep_les_ptr(0,tstp);
  cplx *G2_ret_ptr = G2.curr_timestep_ret_ptr(tstp,0);
  cplx *G3_les_ptr = G3.curr_timestep_les_ptr(0,tstp);
  cplx *S_les_ptr = Sigma.curr_timestep_les_ptr(0,tstp);
  cplx *S_ret_ptr = Sigma.curr_timestep_ret_ptr(tstp,0);

  cplx *tmp_data1 = tmp_nn_.data();
  cplx *tmp_data2 = tmp2_nn_.data();

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  for(int t = 0; t <= tstp; t++) {
    // S^R(T,t) = S^>(T,t) - S^<(T,t) = S^>(T,t) + S^<(t,T)^\dagger
//    ZMatrixMap(S_ret_ptr, Nb_, Nb_) = ZMatrixMap(S_les_ptr, Nb_, Nb_).adjoint();

    // G^>(T,t) = G^R(T,t) + G^<(T,t) = G^R(T,t) - G^<(T,t)^\dagger
    ZMatrixMap(tmp_data1, Nb_, Nb_) = ZMatrixMap(G1_ret_ptr, Nb_, Nb_) - ZMatrixMap(G1_les_ptr, Nb_, Nb_).adjoint();
    ZMatrixMap(tmp_data2, Nb_, Nb_) = ZMatrixMap(G2_ret_ptr, Nb_, Nb_) - ZMatrixMap(G2_les_ptr, Nb_, Nb_).adjoint();

    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_ret_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data +U2_offset, tmp_data1, tmp_data2, G3_les_ptr);
      gf2_contract_deploy(S_ret_ptr + i*Nb_, U_SU_data +U1_offset+i*Nb3_, U_SUEX_data+U2_offset, tmp_data1, tmp_data2, G3_les_ptr);
    }

    G1_les_ptr += Nb2_;
    G1_ret_ptr += Nb2_;
    G2_les_ptr += Nb2_;
    G2_ret_ptr += Nb2_;
    G3_les_ptr += Nb2_;
    S_les_ptr += Nb2_;
    S_ret_ptr += Nb2_;
  }
}

void nband_nsite_SRSU_gf2::TV_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G1, herm_matrix_hodlr &G2, herm_matrix_hodlr &G3, int U1_offset, int U2_offset, double *it2itr) {
  assert(Sigma.size1() == Nb_);
  assert(G1.size1() == Nb_);
  assert(G2.size1() == Nb_);
  assert(G3.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k() );
  assert(G1.tstpmk() + G1.k() == tstp or tstp <= G1.k());
  assert(G2.tstpmk() + G2.k() == tstp or tstp <= G2.k());
  assert(G3.tstpmk() + G3.k() == tstp or tstp <= G3.k());
  assert(Sigma.ntau() == Ntau_);
  assert(G1.ntau() == Ntau_);
  assert(G2.ntau() == Ntau_);
  assert(G3.ntau() == Ntau_);

  cplx *G1_tv_ptr = G1.tvptr(tstp,0);
  cplx *G2_tv_ptr = G2.tvptr(tstp,0);
  cplx *G3_vt_ptr = Git_rev_Tnn_.data();
  cplx *S_tv_ptr = Sigma.tvptr(tstp,0);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  G3.get_vt(tstp, G3_vt_ptr, it2itr);

  for(int t = 0; t < Ntau_; t++) {
    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_tv_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data+U2_offset, G1_tv_ptr, G2_tv_ptr, G3_vt_ptr);
      gf2_contract_deploy(S_tv_ptr + i*Nb_, U_SU_data+U1_offset+i*Nb3_, U_SUEX_data+U2_offset, G1_tv_ptr, G2_tv_ptr, G3_vt_ptr);
    }

    G1_tv_ptr += Nb2_;
    G2_tv_ptr += Nb2_;
    G3_vt_ptr += Nb2_;
    S_tv_ptr += Nb2_;
  }
}

// #############################################################################
// #############################################################################
// ###           MPI GF2 specialization for each Keldysh component           ###
// #############################################################################
// #############################################################################

void nband_nsite_SRSU_gf2::Mat_gf2_contract(herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset, double *it2itr) {
  assert(Sigma.size1() == Nb_);
  assert(Sigma.ntau() == Ntau_);

  double *double_data_ptr_Git_rev = reinterpret_cast<double*>(Git_rev_Tnn_.data());

  G.get_mat_reversed(double_data_ptr_Git_rev, it2itr, indices[2]);

  for(int t = 0; t < Ntau_; t++) {
    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(Sigma.matptr(t) + i*Nb_, U_tot_nnnn_.data()+U1_offset+i*Nb3_, U_tot_nnnn_.data()+U2_offset, G.matptr(indices[0],t), G.matptr(indices[1],t), double_data_ptr_Git_rev + t*Nb2_);
      gf2_contract_deploy(Sigma.matptr(t) + i*Nb_, U_SU_nnnn_.data()+U1_offset+i*Nb3_, U_SUEX_nnnn_.data()+U2_offset, G.matptr(indices[0],t), G.matptr(indices[1],t), double_data_ptr_Git_rev + t*Nb2_);
    }
  }
}


void nband_nsite_SRSU_gf2::Les_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset) {
  assert(Sigma.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k());

  cplx *G1_les_ptr = G.lesptr(indices[0],0,tstp);
  cplx *G2_les_ptr = G.lesptr(indices[1],0,tstp);
  cplx *G3_les_ptr = G.lesptr(indices[2],0,tstp);
  cplx *G3_ret_ptr = G.retptr(indices[2],tstp,0);
  cplx *S_les_ptr = Sigma.curr_timestep_les_ptr(0,tstp);

  cplx *tmp_data = tmp_nn_.data();

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  for(int t = 0; t <= tstp; t++) {
    // G^>(T,t) = G^R(T,t) + G^<(T,t) = G^R(T,t) - G^<(T,t)^\dagger
    ZMatrixMap(tmp_data, Nb_, Nb_) = ZMatrixMap(G3_ret_ptr, Nb_, Nb_) - ZMatrixMap(G3_les_ptr, Nb_, Nb_).adjoint();

    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_les_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data+U2_offset, G1_les_ptr, G2_les_ptr, tmp_data);
      gf2_contract_deploy(S_les_ptr + i*Nb_, U_SU_data+U1_offset+i*Nb3_, U_SUEX_data+U2_offset, G1_les_ptr, G2_les_ptr, tmp_data);
    }

    G1_les_ptr += Nb2_;
    G2_les_ptr += Nb2_;
    G3_les_ptr += Nb2_;
    G3_ret_ptr += Nb2_;
    S_les_ptr += Nb2_;
  }
}

void nband_nsite_SRSU_gf2::Ret_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset) {
  assert(Sigma.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k());

  cplx *G1_les_ptr = G.lesptr(indices[0],0,tstp);
  cplx *G1_ret_ptr = G.retptr(indices[0],tstp,0);
  cplx *G2_les_ptr = G.lesptr(indices[1],0,tstp);
  cplx *G2_ret_ptr = G.retptr(indices[1],tstp,0);
  cplx *G3_les_ptr = G.lesptr(indices[2],0,tstp);
  cplx *S_les_ptr = Sigma.curr_timestep_les_ptr(0,tstp);
  cplx *S_ret_ptr = Sigma.curr_timestep_ret_ptr(tstp,0);

  cplx *tmp_data1 = tmp_nn_.data();
  cplx *tmp_data2 = tmp2_nn_.data();

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  for(int t = 0; t <= tstp; t++) {
    // S^R(T,t) = S^>(T,t) - S^<(T,t) = S^>(T,t) + S^<(t,T)^\dagger
//    ZMatrixMap(S_ret_ptr, Nb_, Nb_) = ZMatrixMap(S_les_ptr, Nb_, Nb_).adjoint();

    // G^>(T,t) = G^R(T,t) + G^<(T,t) = G^R(T,t) - G^<(T,t)^\dagger
    ZMatrixMap(tmp_data1, Nb_, Nb_) = ZMatrixMap(G1_ret_ptr, Nb_, Nb_) - ZMatrixMap(G1_les_ptr, Nb_, Nb_).adjoint();
    ZMatrixMap(tmp_data2, Nb_, Nb_) = ZMatrixMap(G2_ret_ptr, Nb_, Nb_) - ZMatrixMap(G2_les_ptr, Nb_, Nb_).adjoint();

    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_ret_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data +U2_offset, tmp_data1, tmp_data2, G3_les_ptr);
      gf2_contract_deploy(S_ret_ptr + i*Nb_, U_SU_data +U1_offset+i*Nb3_, U_SUEX_data+U2_offset, tmp_data1, tmp_data2, G3_les_ptr);
    }

    G1_les_ptr += Nb2_;
    G1_ret_ptr += Nb2_;
    G2_les_ptr += Nb2_;
    G2_ret_ptr += Nb2_;
    G3_les_ptr += Nb2_;
    S_les_ptr += Nb2_;
    S_ret_ptr += Nb2_;
  }
}

void nband_nsite_SRSU_gf2::TV_gf2_contract(int tstp, herm_matrix_hodlr &Sigma, distributed_single_timeslice &G, std::vector<int> &indices, int U1_offset, int U2_offset, double *it2itr) {
  assert(Sigma.size1() == Nb_);
  assert(Sigma.tstpmk() + Sigma.k() == tstp or tstp <= Sigma.k());
  assert(Sigma.ntau() == Ntau_);

  cplx *G1_tv_ptr = G.tvptr(indices[0],tstp,0);
  cplx *G2_tv_ptr = G.tvptr(indices[1],tstp,0);
  cplx *G3_vt_ptr = Git_rev_Tnn_.data();
  cplx *S_tv_ptr = Sigma.tvptr(tstp,0);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SU_data =   U_SU_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();

  G.get_vt_transpose(tstp, G3_vt_ptr, it2itr, indices[2]);
  for(int t = 0; t < Ntau_; t++) {
    tmp_nn_ = ZMatrixMap(G3_vt_ptr + t*Nb_*Nb_, Nb_, Nb_).transpose();
    ZMatrixMap(G3_vt_ptr + t*Nb_*Nb_, Nb_, Nb_) = tmp_nn_;
  }

  for(int t = 0; t < Ntau_; t++) {
    for(int i = 0; i < Nb_; i++) {
      gf2_contract_deploy(S_tv_ptr + i*Nb_, U_tot_data+U1_offset+i*Nb3_, U_tot_data+U2_offset, G1_tv_ptr, G2_tv_ptr, G3_vt_ptr);
      gf2_contract_deploy(S_tv_ptr + i*Nb_, U_SU_data+U1_offset+i*Nb3_, U_SUEX_data+U2_offset, G1_tv_ptr, G2_tv_ptr, G3_vt_ptr);
    }

    G1_tv_ptr += Nb2_;
    G2_tv_ptr += Nb2_;
    G3_vt_ptr += Nb2_;
    S_tv_ptr += Nb2_;
  }
}


// #############################################################################
// #############################################################################
// ###                              Full system functions                    ###
// #############################################################################
// #############################################################################


void nband_nsite_SRSU_gf2::hf_contract(int tstp, cplx *hmf, herm_matrix_hodlr &G, double *it2cf, int *it2cfp, double *dlrrf, int U_offset) {
  assert(tstp == -1);
  assert(G.size1() == Nb_);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();
  cplx *tmp_data = tmp_nn_.data();
  cplx *rho_trans = tmp_nnn_.data();
  
  G.density_matrix(tstp, it2cf, it2cfp, dlrrf, tmp_data);
  ZMatrixMap(rho_trans, Nb_, Nb_) = ZMatrixMap(tmp_data, Nb_, Nb_).transpose();

  for(int i = 0; i < Nb_; i++) {
    hf_contract_deploy(hmf + i*Nb_, U_tot_data+U_offset + i*Nb3_, rho_trans);
    hf_contract_deploy(hmf + i*Nb_, U_SUEX_data+U_offset + i*Nb3_, rho_trans);
  }
}

void nband_nsite_SRSU_gf2::hf_contract(int tstp, cplx *hmf, herm_matrix_hodlr &G, int U_offset) {
  assert(tstp >= 0);
  assert(tstp <= G.tstpmk() + G.k());
  assert(G.size1() == Nb_);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();
  cplx *tmp_data = tmp_nn_.data();
  cplx *rho_trans = tmp_nnn_.data();
  
  G.get_les(tstp, tstp, tmp_data);
  ZMatrixMap(rho_trans, Nb_, Nb_) = cplx(0.,(double)G.sig()) * ZMatrixMap(tmp_data, Nb_, Nb_).transpose();
  
  for(int i = 0; i < Nb_; i++) {
    hf_contract_deploy(hmf + tstp * Nb2_ + i*Nb_, U_tot_data+U_offset + i*Nb3_, rho_trans);
    hf_contract_deploy(hmf + tstp * Nb2_ + i*Nb_, U_SUEX_data+U_offset + i*Nb3_, rho_trans);
  }
}

void nband_nsite_SRSU_gf2::hf_contract(int tstp, cplx *hmf, distributed_single_timeslice &G, int b, double *it2cf, int *it2cfp, double *dlrrf, int U_offset) {
  assert(tstp == -1);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();
  cplx *tmp_data = tmp_nn_.data();
  cplx *rho_trans = tmp_nnn_.data();
  
  G.density_matrix(tstp, it2cf, it2cfp, dlrrf, tmp_data, b);
  ZMatrixMap(rho_trans, Nb_, Nb_) = ZMatrixMap(tmp_data, Nb_, Nb_).transpose();
  
  for(int i = 0; i < Nb_; i++) {
    hf_contract_deploy(hmf + i*Nb_, U_tot_data+U_offset + i*Nb3_, rho_trans);
    hf_contract_deploy(hmf + i*Nb_, U_SUEX_data+U_offset + i*Nb3_, rho_trans);
  }
}

void nband_nsite_SRSU_gf2::hf_contract(int tstp, cplx *hmf, distributed_single_timeslice &G, int b, int U_offset) {
  assert(tstp >= 0);

  cplx *U_tot_data =  U_tot_nnnn_.data();
  cplx *U_SUEX_data = U_SUEX_nnnn_.data();
  cplx *tmp_data = tmp_nn_.data();
  cplx *rho_trans = tmp_nnn_.data();
  
  G.get_les(b, tstp, tstp, tmp_data);
  ZMatrixMap(rho_trans, Nb_, Nb_) = cplx(0.,(double)G.sig()) * ZMatrixMap(tmp_data, Nb_, Nb_).transpose();
  
  for(int i = 0; i < Nb_; i++) {
    hf_contract_deploy(hmf + tstp * Nb2_ + i*Nb_, U_tot_data+U_offset + i*Nb3_, rho_trans);
    hf_contract_deploy(hmf + tstp * Nb2_ + i*Nb_, U_SUEX_data+U_offset + i*Nb3_, rho_trans);
  }
}


void nband_nsite_SRSU_gf2::gf2_contract(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  assert(G.size1() == Nb_);
  assert(Sigma.size1() == Nb_);
  assert(G.ntau() == Ntau_);
  assert(Sigma.ntau() == Ntau_);
  assert(G.tstpmk() == Sigma.tstpmk());
  assert(tstp == -1 or (tstp >= G.tstpmk() and tstp <= G.tstpmk() + G.k()));

  Sigma.set_tstp_zero(tstp);

  if(tstp == -1) {
    Mat_gf2_contract(Sigma, G, G, G, 0, 0, it2itr);
  }
  else {
    Les_gf2_contract(tstp, Sigma, G, G, G, 0, 0);
    Ret_gf2_contract(tstp, Sigma, G, G, G, 0, 0);
    TV_gf2_contract(tstp, Sigma, G, G, G, 0, 0, it2itr);

    for(int t = 0; t <= tstp; t++) {
      ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, t), Nb_, Nb_) += ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), Nb_, Nb_).adjoint();
    }
  }
}



// #############################################################################
// #############################################################################
// ###                       Public Callable Functions                       ###
// #############################################################################
// #############################################################################


void nband_nsite_SRSU_gf2::hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G_vec, double *it2cf, int *it2cfp, double *dlrrf) {
  assert(tstp == -1);
  assert(Ns_ == hmf_vec.size());
  assert(Ns_ == G_vec.size());
  assert(Nb_ == G_vec[0]->size1());

  for(int a = 0; a < Ns_; a++) {
    for(int b = 0; b < Ns_; b++) {
      int U_offset = (a*Ns3_+b*Ns2_+b*Ns_+a)*Nb4_;
      hf_contract(tstp, hmf_vec[a].ptr(-1), *G_vec[b], it2cf, it2cfp, dlrrf, U_offset);
    }
  }
}

void nband_nsite_SRSU_gf2::hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G_vec) {
  assert(tstp >= 0);
  assert(Ns_ == hmf_vec.size());
  assert(Ns_ == G_vec.size());
  assert(Nb_ == G_vec[0]->size1());

  for(int a = 0; a < Ns_; a++) {
    for(int b = 0; b < Ns_; b++) {
      int U_offset = (a*Ns3_+b*Ns2_+b*Ns_+a)*Nb4_;
      hf_contract(tstp, hmf_vec[a].ptr(0), *G_vec[b], U_offset);
    }
  }
}


void nband_nsite_SRSU_gf2::gf2_contract(int tstp, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &G_vec, double *it2itr) {
  assert(Ns_ == G_vec.size());

  for(int k = 0; k < Ns_; k++) {
    Sigma_vec[k]->set_tstp_zero(tstp);
  }

  for(int a = 0; a < Ns_; a++) {
    for(int b = 0; b < Ns_; b++) {
      for(int c = 0; c < Ns_; c++) {
        for(int d = 0; d < Ns_; d++) {
          int U1_offset = (a*Ns3_+d*Ns2_+c*Ns_+b)*Nb4_;
          int U2_offset = (b*Ns3_+c*Ns2_+d*Ns_+a)*Nb4_;
          if(tstp == -1) {
            Mat_gf2_contract(*Sigma_vec[a], *G_vec[b], *G_vec[c], *G_vec[d], U1_offset, U2_offset, it2itr);
          }
          else {
            Les_gf2_contract(tstp, *Sigma_vec[a], *G_vec[b], *G_vec[c], *G_vec[d], U1_offset, U2_offset);
            Ret_gf2_contract(tstp, *Sigma_vec[a], *G_vec[b], *G_vec[c], *G_vec[d], U1_offset, U2_offset);
            TV_gf2_contract (tstp, *Sigma_vec[a], *G_vec[b], *G_vec[c], *G_vec[d], U1_offset, U2_offset, it2itr);
          }
        }
      }
    }
  }

  for(int k = 0; k < Ns_; k++) {
    for(int t = 0; t <= tstp; t++) {
      ZMatrixMap(Sigma_vec[k]->curr_timestep_ret_ptr(tstp, t), Nb_, Nb_) += ZMatrixMap(Sigma_vec[k]->curr_timestep_les_ptr(t,tstp), Nb_, Nb_).adjoint();
    }
  }
}


// #############################################################################
// #############################################################################
// ###                       MPI Public Callable Functions                   ###
// #############################################################################
// #############################################################################

void nband_nsite_SRSU_gf2::hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points, double *it2cf, int *it2cfp, double *dlrrf) {
  assert(tstp == -1);
  assert(Ns_ == hmf_vec.size());
  int loc_Ns = local_k_points.size();

  for(int a = 0; a < loc_Ns; a++) {
    for(int b = 0; b < Ns_; b++) {
      int a_loc = local_k_points[a];
      int U_offset = (a_loc*Ns3_+b*Ns2_+b*Ns_+a_loc)*Nb4_;
      hf_contract(tstp, hmf_vec[a].ptr(-1), G, b, it2cf, it2cfp, dlrrf, U_offset);
    }
  }
}

void nband_nsite_SRSU_gf2::hf_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points) {
  assert(tstp == -1);
  assert(Ns_ == hmf_vec.size());
  int loc_Ns = local_k_points.size();

  for(int a = 0; a < loc_Ns; a++) {
    for(int b = 0; b < Ns_; b++) {
      int a_loc = local_k_points[a];
      int U_offset = (a_loc*Ns3_+b*Ns2_+b*Ns_+a_loc)*Nb4_;
      hf_contract(tstp, hmf_vec[a].ptr(0), G, b, U_offset);
    }
  }
}

void nband_nsite_SRSU_gf2::gf2_contract(int tstp, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma_vec, distributed_single_timeslice &G, std::vector<int> &local_k_points, double *it2itr) {

  int loc_Ns = local_k_points.size();
  for(int k = 0; k < loc_Ns; k++) {
    Sigma_vec[k]->set_tstp_zero(tstp);
  }
  
  std::vector<int> indices(3);

  for(int a = 0; a < loc_Ns; a++) {
    for(int b = 0; b < Ns_; b++) {
      for(int c = 0; c < Ns_; c++) {
        for(int d = 0; d < Ns_; d++) {
          int a_loc = local_k_points[a];
          int U1_offset = (a_loc*Ns3_+d*Ns2_+c*Ns_+b)*Nb4_;
          int U2_offset = (b*Ns3_+c*Ns2_+d*Ns_+a_loc)*Nb4_;
          indices = {b,c,d};
          if(tstp == -1) {
            Mat_gf2_contract(*Sigma_vec[a], G, indices, U1_offset, U2_offset, it2itr);
          }
          else {
            Les_gf2_contract(tstp, *Sigma_vec[a], G, indices, U1_offset, U2_offset);
            Ret_gf2_contract(tstp, *Sigma_vec[a], G, indices, U1_offset, U2_offset);
            TV_gf2_contract (tstp, *Sigma_vec[a], G, indices, U1_offset, U2_offset, it2itr);
          }
        }
      }
    }
  }

  for(int k = 0; k < loc_Ns; k++) {
    for(int t = 0; t <= tstp; t++) {
      ZMatrixMap(Sigma_vec[k]->curr_timestep_ret_ptr(tstp, t), Nb_, Nb_) += ZMatrixMap(Sigma_vec[k]->curr_timestep_les_ptr(t,tstp), Nb_, Nb_).adjoint();
    }
  }
}



}

#endif

