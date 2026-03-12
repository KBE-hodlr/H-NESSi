#include "nband_nsite_Hubbard_gf2.hpp"

#define ISMAP Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >
#define IS Eigen::InnerStride<>

namespace hodlr {
nband_nsite_Hubbard_gf2::nband_nsite_Hubbard_gf2(int Ns, int Nb, int Ntau, double U, std::vector<int>& permutation, int ndim) :
  Nb_(Nb),
  Nb2_(Nb*Nb),
  Ns_(Ns),
  Ntau_(Ntau),
  Git_rev_TnnS_(Ntau*Nb*Nb, Ns),
  tmp2_rnn_(Ns, Nb*Nb),
  U_(U),
  Perm_(Ns)
{
  Perm_.indices() = Eigen::Map<const Eigen::VectorXi>(permutation.data(), Ns);

  int nomp;
  #pragma omp parallel
  {
    nomp = omp_get_num_threads();
  }
  tmp_rnn_ = ZMatrix::Zero(nomp * Ns, Nb*Nb);

  if(ndim == 2) {
    int rank = 2;
    int L = (int) std::sqrt(Ns);
    int n[] = {L, L};
    int howmany = Nb2_;
    int istride = Nb2_;
    int ostride = Nb2_;
    int idist = 1;
    int odist = 1;
    p_r2k = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(tmp_rnn_.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(tmp2_rnn_.data()), NULL, ostride, odist, FFTW_FORWARD,  FFTW_ESTIMATE);
    p_k2r = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(tmp_rnn_.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(tmp2_rnn_.data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else if(ndim == 1) {
    int rank = 1;
    int L = Ns;
    int n[] = {L};
    int howmany = Nb2_;
    int istride = Nb2_;
    int ostride = Nb2_;
    int idist = 1;
    int odist = 1;
    p_r2k = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(tmp_rnn_.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(tmp2_rnn_.data()), NULL, ostride, odist, FFTW_FORWARD,  FFTW_ESTIMATE);
    p_k2r = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(tmp_rnn_.data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(tmp2_rnn_.data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }

}

void nband_nsite_Hubbard_gf2::hf_gf2_contract(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2cf, int *it2cfp, double *dlrrf, double *it2itr, bool eval_hmf, bool eval_gf2) {
  
  if(eval_hmf) {
    // fill tmp_rnn_ with Rho(k)_nn data
    for(int k = 0; k < Ns_; k++) {
      G.density_matrix(-1, it2cf, it2cfp, dlrrf, tmp_rnn_.data() + k*Nb2_, k);
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
      ZMatrixMap(hmf_vec[k].ptr(-1), Nb_, Nb_) += ZMatrixMap(tmp2_rnn_.data() + local_k_points[k] * Nb2_, Nb_, Nb_);
    }
  }

  if(eval_gf2) {
    // G_kzab -> G_zabr
    G.FT_k2r();
  
    // fill Git_rev_TnnR
    ZMatrixMap(Git_rev_TnnS_.data(), Ntau_, Nb2_*Ns_).noalias() = DMatrixMap(it2itr, Ntau_, Ntau_).transpose() * ZMatrixMap(G.matptr2_zab_S(0,0,0), Ntau_, Nb2_*Ns_);
  
    // S_(TnnR) = G(TnnR) G(TnnR) G(-TnnP(R))
    double U2divnk3 = U_*U_/Ns_/Ns_/Ns_;
    for(int t = 0; t < Ntau_; t++) {
      for(int a = 0; a < Nb_; a++) {
        ZMatrixMap(G.matptr_zab_S(t,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(G.matptr2_zab_S(t,a,a), Ns_, 1).cwiseProduct(
                                                            ZMatrixMap(G.matptr2_zab_S(t,a,a), Ns_, 1).cwiseProduct(
                                                      Perm_*ZMatrixMap(Git_rev_TnnS_.data() + t*Nb2_*Ns_ + a*Nb_*Ns_ + a*Ns_, Ns_, 1)));
      }
    }
  
    // S_(KTnn)
    G.FT_r2k();
  
    // results into sigma
    for(int k = 0; k < local_k_points.size(); k++) {
      DMatrixMap(Sigma[k]->matptr(0), Ntau_ * Nb_, Nb_) = ISMAP(G.matptr2_zab_S(0, 0, 0)+local_k_points[k], Ntau_ * Nb_, Nb_, IS(Ns_)).real();
    }
  }
}



void nband_nsite_Hubbard_gf2::hf_gf2_contract_boot(std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool eval_hmf, bool eval_gf2) {
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

  if(eval_gf2){
  // G_kzab -> G_zabr
  G.FT_k2r();

  double U2divnk3 = U_*U_/Ns_/Ns_/Ns_;
  for(int t = 0; t <= order_k; t++) {
    // fill Git_rev_TnnR
    ZMatrixMap(Git_rev_TnnS_.data(), Ntau_, Nb2_*Ns_).noalias() = -G.sig() * DMatrixMap(it2itr, Ntau_, Ntau_).transpose() * ZMatrixMap(G.tvptr2_zab_S(t,0,0,0), Ntau_, Nb2_*Ns_).conjugate();

    // S_(TnnR) = G(TnnR) G(TnnR) G(-TnnP(R))
    for(int tau = 0; tau < Ntau_; tau++) {
      for(int a = 0; a < Nb_; a++) {
        ZMatrixMap(G.tvptr_zab_S(t,tau,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(G.tvptr2_zab_S(t,tau,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(G.tvptr2_zab_S(t,tau,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(Git_rev_TnnS_.data() + tau*Nb2_*Ns_ + a*Nb_*Ns_ + a*Ns_, Ns_, 1)));
      }
    }
    for(int tp = 0; tp <= t; tp++) {
      for(int a = 0; a < Nb_; a++) {
        // G^>(t,tp)_{i,0}
        ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = ZMatrixMap(G.retptr2_zab_S(t,tp,a,a), Ns_, 1) - Perm_ * ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1).conjugate();
        ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1).cwiseProduct(
                                                         Perm_*ZMatrixMap(tmp_rnn_.data(), Ns_, 1)));
        ZMatrixMap(G.retptr_zab_S(t,tp,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                               ZMatrixMap(tmp_rnn_.data(), Ns_, 1).cwiseProduct(
                                                         Perm_*ZMatrixMap(G.lesptr2_zab_S(tp,t,a,a), Ns_, 1)));
        ZMatrixMap(tmp_rnn_.data(), Ns_, 1) = Perm_*ZMatrixMap(G.lesptr_zab_S(tp,t,a,a), Ns_, 1).conjugate();
        ZMatrixMap(G.retptr_zab_S(t,tp,a,a), Ns_, 1) += ZMatrixMap(tmp_rnn_.data(), Ns_, 1);
      }
    }
  }

  // S_(KTnn)
  G.FT_r2k();

  // results into sigma
  for(int k = 0; k < local_k_points.size(); k++) {
    for(int t = 0; t <= order_k; t++) {
      ZMatrixMap(Sigma[k]->curr_timestep_ret_ptr(t,0), (t+1) * Nb_, Nb_) = ISMAP(G.retptr2_zab_S(t, 0, 0, 0)+local_k_points[k], (t+1) * Nb_, Nb_, IS(Ns_));
      ZMatrixMap(Sigma[k]->curr_timestep_les_ptr(0,t), (t+1) * Nb_, Nb_) = ISMAP(G.lesptr2_zab_S(0, t, 0, 0)+local_k_points[k], (t+1) * Nb_, Nb_, IS(Ns_));
      ZMatrixMap(Sigma[k]->tvptr(t,0), Ntau_ * Nb_, Nb_) = ISMAP(G.tvptr2_zab_S(t, 0, 0, 0)+local_k_points[k], Ntau_ * Nb_, Nb_, IS(Ns_));
    }
  }
  } //eval_gf2
}

std::vector<double> nband_nsite_Hubbard_gf2::hf_gf2_contract(int tstp, std::vector<cntr::function<double>> &hmf_vec, std::vector<std::unique_ptr<herm_matrix_hodlr>> &Sigma, distributed_single_timeslice_FT &G, std::vector<int> &local_k_points, double *it2itr, bool eval_hmf, bool eval_gf2) {
  int order_k = G.order_k();

  if(eval_hmf) {
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
  }

  std::vector<double> times(10);

  if(eval_gf2) {
  // Need elements zero for mpi_sum_it
  memset(G.retptr2_zab_S(tstp,0,0,0), 0, (2*(tstp+1)+Ntau_)*Nb_*Nb_*Ns_*sizeof(cplx));
  
  // G_kzab -> G_zabr
  auto start = std::chrono::high_resolution_clock::now();
  G.FT_k2r();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[0] = duration.count();

  // Get all IT points for eval of Gvt
  start = std::chrono::high_resolution_clock::now();
  G.mpi_sum_it();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[2] = duration.count();

  // fill Git_rev_TnnR
  start = std::chrono::high_resolution_clock::now();
  ZMatrixMap(Git_rev_TnnS_.data(), G.tauend()-G.taustart()+1, Nb2_*Ns_).noalias() = -G.sig() * DMatrixMap(it2itr, Ntau_, Ntau_).block(0, G.taustart(), Ntau_, G.tauend()-G.taustart()+1).transpose() * ZMatrixMap(G.tvptr2_zab_S(tstp,0,0,0), Ntau_, Nb2_*Ns_).conjugate();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[3] = duration.count();

  // S_(TnnR) = G(TnnR) G(TnnR) G(-TnnP(R))
  double U2divnk3 = U_*U_/Ns_/Ns_/Ns_;
  start = std::chrono::high_resolution_clock::now();
  for(int tau = G.taustart(); tau <= G.tauend(); tau++) {
    for(int a = 0; a < Nb_; a++) {
      ZMatrixMap(G.tvptr_zab_S(tstp,tau,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(G.tvptr2_zab_S(tstp,tau,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(G.tvptr2_zab_S(tstp,tau,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(Git_rev_TnnS_.data() + (tau-G.taustart())*Nb2_*Ns_ + a*Nb_*Ns_ + a*Ns_, Ns_, 1)));
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[4] = duration.count();

  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int tp = G.tstart(); tp <= G.tend(); tp++) {
    int omp_id = omp_get_thread_num();
    for(int a = 0; a < Nb_; a++) {
      // G^>(t,tp)_{i,0}
      ZMatrixMap(tmp_rnn_.data() + omp_id * Ns_, Ns_, 1) = ZMatrixMap(G.retptr2_zab_S(tstp,tp,a,a), Ns_, 1) - Perm_ * ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1).conjugate();
      ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1).cwiseProduct(
                                                          Perm_*ZMatrixMap(tmp_rnn_.data()+omp_id*Ns_, Ns_, 1)));
      ZMatrixMap(G.retptr_zab_S(tstp,tp,a,a), Ns_, 1) = U2divnk3 * ZMatrixMap(tmp_rnn_.data()+omp_id*Ns_, Ns_, 1).cwiseProduct(
                                                                ZMatrixMap(tmp_rnn_.data()+omp_id*Ns_, Ns_, 1).cwiseProduct(
                                                          Perm_*ZMatrixMap(G.lesptr2_zab_S(tp,tstp,a,a), Ns_, 1)));
      ZMatrixMap(tmp_rnn_.data() + omp_id * Ns_, Ns_, 1) = Perm_*ZMatrixMap(G.lesptr_zab_S(tp,tstp,a,a), Ns_, 1).conjugate();
      ZMatrixMap(G.retptr_zab_S(tstp,tp,a,a), Ns_, 1) += ZMatrixMap(tmp_rnn_.data() + omp_id*Ns_, Ns_, 1);
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[5] = duration.count();

  // Need elements zero for mpi_sum
  memset(G.retptr2_zab_S(tstp,0,0,0), 0, (2*(tstp+1)+Ntau_)*Nb_*Nb_*Ns_*sizeof(cplx));

  // S_(Tnnk)
  start = std::chrono::high_resolution_clock::now();
  G.FT_r2k();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[7] = duration.count();

  // collect data for all z
  start = std::chrono::high_resolution_clock::now();
  G.mpi_sum();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[8] = duration.count();

  // results into sigma
  start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int k = 0; k < local_k_points.size(); k++) {
    ZMatrixMap(Sigma[k]->curr_timestep_ret_ptr(tstp,0), (tstp+1) * Nb_, Nb_) = ISMAP(G.retptr2_zab_S(tstp, 0, 0, 0)+local_k_points[k], (tstp+1) * Nb_, Nb_, IS(Ns_));
    ZMatrixMap(Sigma[k]->curr_timestep_les_ptr(0,tstp), (tstp+1) * Nb_, Nb_) = ISMAP(G.lesptr2_zab_S(0, tstp, 0, 0)+local_k_points[k], (tstp+1) * Nb_, Nb_, IS(Ns_));
    ZMatrixMap(Sigma[k]->tvptr(tstp,0), Ntau_ * Nb_, Nb_) = ISMAP(G.tvptr2_zab_S(tstp, 0, 0, 0)+local_k_points[k], Ntau_ * Nb_, Nb_, IS(Ns_));
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  times[9] = duration.count();
  }//evalgf2

  return times;
}

}
