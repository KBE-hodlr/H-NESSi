#include "nband_nsite_ExtHubbard_gf2.hpp"

#define ISMAP Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >
#define IS Eigen::InnerStride<>

namespace hodlr {
nband_nsite_ExtHubbard_gf2::nband_nsite_ExtHubbard_gf2(int Ns, int Nb, int Nt, int Ntau, double U, DMatrix &W, std::vector<int>& permutation, int nmpi, int order_k) :
  Nt_(Nt),
  Nb_(Nb),
  Nb2_(Nb*Nb),
  Ns_(Ns),
  Ns2_(Ns*Ns),
  Ntau_(Ntau),
  Nz_(std::max(2*(order_k+1)*(order_k+2)/2+(order_k+1)*Ntau, 2*(Nt/npmi+1)+(Ntau/nmpi+1)));
  order_k_(order_k),
  Git_rev_Tarbs_(Ntau, Ns*Ns*Nb*Nb),
  D_zarbs_(Nz_, Nb*Nb*Ns*Ns),
  D2_zarbs_(Nz_, Nb*Nb*Ns*Ns),
  tmp_arbs_(Nb*Ns, Nb*Ns),
  tmp2_arbs_(Nb*Ns, Nb*Ns),
  W_(W),
  U_(U),
  Perm_(Ns)
{
  Perm_.indices() = Eigen::Map<const Eigen::VectorXi>(permutation.data(), Ns);

  int rank = 1;
  int n[] = {Ns_};
  int howmany = Nb2_;
  int istride = Nb2_;
  int ostride = Nb2_;
  int idist = 1;
  int odist = 1;
  p_r2k = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(D2_zarbs.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(D_zarbs_.data()), NULL, ostride, odist, FFTW_FORWARD,  FFTW_ESTIMATE);
  p_k2r = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(D_zarbs.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(D2_zarbs_.data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
}

void nband_nsite_ExtHubbard_gf2::rho_abi_rho_aibj() {
  // first fill rho_abi0 = rho_abi
  ISMAP(D2_zarbs_.data(), Nb2_, Ns_, IS(Ns_)) = ZMatrixMap(D_zarbs_.data(), Nb2_, Ns_);

  // rho_abij = rho_abi(j-1)*P
  for(int j = 1; j < Ns_; j++) {
    ISMAP(D2_zarbs_.data()+j, Nb2_, Ns_, IS(Ns_)).noalias() = ISMAP(D2_zarbs_.data()+j, Nb2_, Ns_, IS(Ns_)) * Perm_;
  }

  // TRANSPOSE
  // rho_aibj = rho_abij
  for(int a = 0, a < Nb_; a++) {
    for(int j = 0; j < Ns_; j++) {
      ISMAP(D_zarbs_.data() + a*Ns2_*Nb_ + j, Ns_, Nb_, IS(Ns_)) = ISMAP(D2_zarbs_.data() + a*Ns2_*Nb_ + j, Nb_, Ns_, IS(Ns_)).transpose();
    }
  }
}

void nband_nsite_ExtHubbard_gf2:zabi_zaibj(distributed_single_timeslice_FT &G) {
  int num_z = G.blocksize();

  // zabi0 = zabi
  ISMAP(D2_zarbs_.data(), num_z * Nb2_, Ns_, IS(Ns_)) = ZMatrixMap(G.retptr_zab_S(0,0,0,0), num_z * Nb2_, Ns_);

  // zabij = zabi(j-1)*P
  for(int j = 1; j < Ns_; j++) {
    ISMAP(D2_zarbs_.data()+j, num_z * Nb2_, Ns_, IS(Ns_)).noalias() = ISMAP(D2_zarbs_.data()+j, num_z * Nb2_, Ns_, IS(Ns_)) * Perm_;
  }

  //TRANSPOSE
  // zaibj = zabij
  for(int z = 0; z < num_z; z++) {
    for(int a = 0, a < Nb_; a++) {
      for(int j = 0; j < Ns_; j++) {
        ISMAP(D_zarbs_.data() + z*Ns2_*Nb2_ + a*Ns2_*Nb_ + j, Ns_, Nb_, IS(Ns_)) = ISMAP(D2_zarbs_.data() + z*Ns2_*Nb2_ + a*Ns2_*Nb_ + j, Nb_, Ns_, IS(Ns_)).transpose();
      }
    }
  }
}

void nband_nsite_ExtHubbard_gf2::hf_gf2_contract(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2cf, int *it2cfp, double *dlrrf, double *it2itr, bool print) {
  
  // fill tmp_rnn_ with Rho(k)_nn data
  for(int k = 0; k < Ns_; k++) {
    G.density_matrix(-1, it2cf, it2cfp, dlrrf, D_zarbs_.data() + k*Nb2_, k);
  }

  // fill tmp2_rnn_ with Rho(i)_nn data
  fftw_execute(p_k2r);
  ZMatrixMap(D2_zarbs_.data(), Ns_*Nb2_, 1) /= Ns_;

  // rho_iab -> rho_abi
  ZMatrixMap(D_zarbs_.data(), Nb2_, Ns_) = ZMatrixMap(D2_zarbs_.data(), Ns_, Nb2_).transpose();

  // rho_abi -> rho_aibj
  rho_abi_rho_aibj(); 

  // Hubbard Hartree
  ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_) = U * ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_).diagonal().asDiagonal();
  
  // Ext Hartree
  ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_).noalias() += 2 * W_ * ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_).diagonal();

  // Ext Fock
  ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_).noalias() -= W_.cwiseProduct(ZMatrixMap(D2_zarbs_.data(), Nb_*Ns_, Nb_*Ns_));

  // aibj -> abi
  for(int a = 0, a < Nb_; a++) {
    ZMatrixMap(D_zarbs_.data() + a*Ns_*Nb_, Nb_, Ns_) = ISMAP(D2_zarbs_.data(), Ns_, Nb_, IS(Ns_)).transpose();
  }
  
  // abi -> iab
  ZMatrixMap(D2_zarbs_.data(), Ns_, Nb2_) = ZMatrixMap(D_zarbs_.data(), Nb2_, Ns_).transpose();

  // FFT
  fftw_execute(p_r2k);
  
  // put results into hmf_vec
  for(int k = 0; k < local_k_points.size(); k++) {
    ZMatrixMap(hmf_vec[k].ptr(-1), Nb_, Nb_) += ZMatrixMap(D_zarbs_.data() + local_k_points[k] * Nb2_, Nb_, Nb_);
  }

  // G_kzab -> G_rzab
  G.FT_k2r();

  // G_rzab -> G_zabr
  G.Trans_Sin();

  // zabi -> zaibj
  zabi_zaibj(G);

  // fill Git_rev_TnnR
  ZMatrixMap(Git_rev_Tarbs_.data(), Ntau_, Nb2_*Ns2_).noalias() = DMatrixMap(it2itr, Ntau_, Ntau_).transpose() * ZMatrixMap(D_zarbs_.data(), Ntau_, Nb2_*Ns2_);

  // S_(Taibj) = G(Taibj) G(Taibj) G(-Taibj)
  for(int t = 0; t < Ntau_; t++) {
    // Hubbard-Hubbard
    ZMatrixMap(D2_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_) = U_*U_* ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(
                                                                          ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(
                                                                          ZMatrixMap(Git_rev_Tarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_)));
    // Hubbard-Ext
    tmp_arbs = ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(
               ZMatrixMap(Git_rev_Tarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).transpose());
    tmp2_arbs.noalias() = tmp_arbs * W_.transpose();
    ZMatrixMap(D2_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_) += ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(tmp2_arbs);
  
    // Ext-Hubbard
    tmp2_arbs.noalias() = W_ * tmp_arbs;
    ZMatrixMap(D2_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_) += ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(tmp2_arbs);

    // Ext-Ext Bubb
    tmp_arbs.noalias() = tmp2_arbs * W_.transpose();
    ZMatrixMap(D2_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_) += 2 * ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).cwiseProduct(tmp_arbs);

    // Ext-Ext Exch
    for(int i = 0; i < Ns_; i++) {
      tmp_arbs.noalias() = W_.row(i).transpose() * ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).row(i);
      tmp2_arbs.noalias() = tmp_arbs.cwiseProduct(ZMatrixMap(Git_rev_Tarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).transpose());
      tmp_arbs.noalias() = ZMatrixMap(D_zarbs_.data() + t*Nb2_*Ns2_, Ns_*Nb_, Ns_*Nb_).transpose() * tmp2_arbs;
      ZMatrixMap(D2_zarbs_.data() + t*Nb2_*Ns2_ + i*Ns_*Nb_, Ns_*Nb_, 1) -= (W_ * tmp_arbs.transpose()).diagonal();
    }
  }

  // TRANSPOSE
  // zarbs -> zabr 
  for(int t = 0; t < Ntau_; t++) {
    for(int a = 0; a < Nb_; a++) {
      ZMatrixMap(G.matptr2_zab_S(t,a,0), Nb_, Ns_) = ISMAP(D2_zarbs_.data() + t*Nb2_*Ns2_ + a*Nb_*Ns2_, Ns_, Nb_, IS(Ns_)).transpose();
    }
  }

  // zabr -> rzab
  G.Trans_Sout();

  // S_(KTnn)
  G.FT_r2k();

  // results into sigma
  for(int k = 0; k < local_k_points.size(); k++) {
    DMatrixMap(Sigma[k]->matptr(0), Ntau_ * Nb_, Nb_) = ZMatrixMap(G.matptr2(local_k_points[k], 0), Ntau_ * Nb_, Nb_).real();
  }
}

/*

void nband_nsite_Hubbard_gf2::hf_gf2_contract_boot(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool eval_hmf) {
  int order_k = G.order_k();

  if(eval_hmf) {
    for(int t = 0; t <= order_k; t++) {
      // fill tmp_rnn_ with Rho(k)_nn data
      for(int k = 0; k < Ns_; k++) {
        ZMatrixMap(tmp_rnn_.data() + k*Nb2_, Nb_, Nb_) = cplx(0.,G.sig()) * ZMatrixMap(G.lesptr(k, t, t), Nb_, Nb_);
      }
  
      // fill tmp2_rnn_ with Rho(i)_nn data
      fftw_execute(p_k2r);
      ZMatrixMap(tmp2_rnn_.data(), Ns_*Nb2_, 1) /= Ns_;
  
      // Sigma_{0aa} = U \rho_{0aa}
      tmp_rnn_.setZero();
      for(int a = 0; a < Nb_; a++) {
        tmp_rnn_.data()[a*Nb_ + a] = U_ * tmp2_rnn_.data()[a*Nb_ + a];
      }
  
      // fill tmp2_rnn_ with Sigma(k)_nn data
      fftw_execute(p_r2k);
  
      // put results into hmf_vec
      for(int k = 0; k < local_k_points.size(); k++) {
        ZMatrixMap(hmf_vec[k].ptr(t), Nb_, Nb_) += ZMatrixMap(tmp2_rnn_.data() + local_k_points[k] * Nb2_, Nb_, Nb_);
      }
    }
  }
  
  // G_kzab -> G_rzab
  G.FT_k2r();

  // G_rzab -> G_zabr
  G.Trans_Sin();

  for(int t = 0; t <= order_k; t++) {
    // fill Git_rev_TnnR
    ZMatrixMap(Git_rev_TnnS_.data(), Ntau_, Nb2_*Ns_).noalias() = -G.sig() * DMatrixMap(it2itr, Ntau_, Ntau_).transpose() * ZMatrixMap(G.tvptr_zab_S(t,0,0,0), Ntau_, Nb2_*Ns_).conjugate();

    // S_(TnnR) = G(TnnR) G(TnnR) G(-TnnP(R))
    for(int tau = 0; tau < Ntau_; tau++) {
      for(int a = 0; a < Nb_; a++) {
        ZMatrixMap(G.tvptr2_zab_S(t,tau,a,a), Ns_, 1) = U_*U_* ZMatrixMap(G.tvptr_zab_S(t,tau,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(G.tvptr_zab_S(t,tau,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(Git_rev_TnnS_.data() + tau*Nb2_*Ns_ + a*Nb_*Ns_ + a*Ns_, Ns_, 1)));
      }
    }
    for(int tp = 0; tp <= t; tp++) {
      for(int a = 0; a < Nb_; a++) {
        // G^>(t,tp)_{i,0}
        ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = ZMatrixMap(G.retptr_zab_S(t,tp,a,a), Ns_, 1) - Perm_ * ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1).conjugate();
        ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1) = U_*U_* ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1).cwiseProduct(
                                                         Perm_*ZMatrixMap(tmp_rnn_.data(), Ns_, 1)));
        ZMatrixMap(G.retptr2_zab_S(t,tp,a,a), Ns_, 1) = U_*U_* ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                         Perm_*ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1)));
        ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = Perm_*ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1).conjugate();
        ZMatrixMap(G.retptr2_zab_S(t,tp,a,a), Ns_, 1) += ZMatrixMap(tmp_rnn_.data(), Ns_, 1);
      }
    }
  }

  // S_(RTnn) = S_(TnnR)
  G.Trans_Sout();

  // S_(KTnn)
  G.FT_r2k();

  // results into sigma
  for(int k = 0; k < local_k_points.size(); k++) {
    for(int t = 0; t <= order_k; t++) {
      ZMatrixMap(Sigma[k]->curr_timestep_ret_ptr(t,0), (t+1) * Nb_, Nb_) = ZMatrixMap(G.retptr2(local_k_points[k], t, 0), (t+1) * Nb_, Nb_);
      ZMatrixMap(Sigma[k]->curr_timestep_les_ptr(0,t), (t+1) * Nb_, Nb_) = ZMatrixMap(G.lesptr2(local_k_points[k], 0, t), (t+1) * Nb_, Nb_);
      ZMatrixMap(Sigma[k]->tvptr(t,0), Ntau_ * Nb_, Nb_) = ZMatrixMap(G.tvptr2(local_k_points[k], t, 0), Ntau_ * Nb_, Nb_);
    }
  }
}

void nband_nsite_Hubbard_gf2::hf_gf2_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool print) {
  int order_k = G.order_k();

  // fill tmp_rnn_ with Rho(k)_nn data
  for(int k = 0; k < Ns_; k++) {
    ZMatrixMap(tmp_rnn_.data() + k*Nb2_, Nb_, Nb_) = cplx(0.,G.sig()) * ZMatrixMap(G.lesptr(k, tstp, tstp), Nb_, Nb_);
  }

  // fill tmp2_rnn_ with Rho(i)_nn data
  fftw_execute(p_k2r);
  ZMatrixMap(tmp2_rnn_.data(), Ns_*Nb2_, 1) /= Ns_;

  // Sigma_{0aa} = U \rho_{0aa}
  tmp_rnn_.setZero();
  for(int a = 0; a < Nb_; a++) {
    tmp_rnn_.data()[a*Nb_ + a] = U_ * tmp2_rnn_.data()[a*Nb_ + a];
  }

  // fill tmp2_rnn_ with Sigma(k)_nn data
  fftw_execute(p_r2k);

  // put results into hmf_vec
  for(int k = 0; k < local_k_points.size(); k++) {
    ZMatrixMap(hmf_vec[k].ptr(tstp), Nb_, Nb_) += ZMatrixMap(tmp2_rnn_.data() + local_k_points[k] * Nb2_, Nb_, Nb_);
  }
  
  // G_kzab -> G_rzab
  G.FT_k2r();

  // G_rzab -> G_zabr
  G.Trans_Sin();

  // fill Git_rev_TnnR
  ZMatrixMap(Git_rev_TnnS_.data(), Ntau_, Nb2_*Ns_).noalias() = -G.sig() * DMatrixMap(it2itr, Ntau_, Ntau_).transpose() * ZMatrixMap(G.tvptr_zab_S(tstp,0,0,0), Ntau_, Nb2_*Ns_).conjugate();

  // S_(TnnR) = G(TnnR) G(TnnR) G(-TnnP(R))
  for(int tau = 0; tau < Ntau_; tau++) {
    for(int a = 0; a < Nb_; a++) {
      ZMatrixMap(G.tvptr2_zab_S(tstp,tau,a,a), Ns_, 1) = U_*U_* ZMatrixMap(G.tvptr_zab_S(tstp,tau,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(G.tvptr_zab_S(tstp,tau,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(Git_rev_TnnS_.data() + tau*Nb2_*Ns_ + a*Nb_*Ns_ + a*Ns_, Ns_, 1)));
    }
  }

  for(int tp = G.tstart(); tp <= G.tend(); tp++) {
    for(int a = 0; a < Nb_; a++) {
      // G^>(t,tp)_{i,0}
      ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = ZMatrixMap(G.retptr_zab_S(tstp,tp,a,a), Ns_, 1) - Perm_ * ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1).conjugate();
      ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1) = U_*U_* ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1).cwiseProduct(
                                                          Perm_*ZMatrixMap(tmp_rnn_.data(), Ns_, 1)));
      ZMatrixMap(G.retptr2_zab_S(tstp,tp,a,a), Ns_, 1) = U_*U_* ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                          Perm_*ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1)));
      ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = Perm_*ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1).conjugate();
      ZMatrixMap(G.retptr2_zab_S(tstp,tp,a,a), Ns_, 1) += ZMatrixMap(tmp_rnn_.data(), Ns_, 1);
    }
  }

  // S_(RTnn) = S_(TnnR)
  G.Trans_Sout();

  // Need elements zero for mpi_sum
  memset(G.retptr2_zab_S(tstp,0,0,0), 0, 2*(tstp+1+Ntau_)*Nb_*Nb_*Ns_*sizeof(cplx));

  // S_(KTnn)
  G.FT_r2k();

  // results into sigma^TV
  for(int k = 0; k < local_k_points.size(); k++) {
    ZMatrixMap(Sigma[k]->tvptr(tstp,0), Ntau_ * Nb_, Nb_) = ZMatrixMap(G.tvptr2(local_k_points[k], tstp, 0), Ntau_ * Nb_, Nb_);
  }

  // collect data for all T
  G.mpi_sum();

  // results into sigma
  for(int k = 0; k < local_k_points.size(); k++) {
    ZMatrixMap(Sigma[k]->curr_timestep_ret_ptr(tstp,0), (tstp+1) * Nb_, Nb_) = ZMatrixMap(G.retptr2(local_k_points[k], tstp, 0), (tstp+1) * Nb_, Nb_);
    ZMatrixMap(Sigma[k]->curr_timestep_les_ptr(0,tstp), (tstp+1) * Nb_, Nb_) = ZMatrixMap(G.lesptr2(local_k_points[k], 0, tstp), (tstp+1) * Nb_, Nb_);
  }
}
*/
}
