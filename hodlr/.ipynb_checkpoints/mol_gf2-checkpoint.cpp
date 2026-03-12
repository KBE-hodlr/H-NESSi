#include "mol_gf2.hpp"

using namespace hodlr;

mol_gf2::mol_gf2(int ntau, int nao, const double *U_int, const double *U_exch, bool use_dlr) {
  nao_ = nao;
  ntau_ = ntau;
  A1_aaa = new cplx[nao*nao*nao];
  B1_aaa = new cplx[nao*nao*nao];
  B2_aaa = new cplx[nao*nao*nao];
  rho_T = ZMatrix(nao_, nao_);
  rho = ZMatrix(nao_, nao_);

  use_dlr_ = use_dlr;

  if(use_dlr) {
    Git_rev = new cplx[ntau*nao*nao];
    DGit_rev = reinterpret_cast<double *>(Git_rev);
  }
  else {
    Git_rev = 0;
    DGit_rev = 0;
  }

  Uijkl_ = U_int;
  Uijkl_exch_ = U_exch;
}

mol_gf2::~mol_gf2() {
  delete [] A1_aaa; 
  delete [] B1_aaa; 
  delete [] B2_aaa; 

  delete [] Git_rev;
}

void mol_gf2::solve_mat(herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();

  if(use_dlr_) {
    DMatrixMap(DGit_rev, ntau_, nao2).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * DMatrixMap(G.mat(), ntau_, nao2);
  }
  
  auto A_aA = ZMatrixMap(A1_aaa, nao_, nao2);
  auto A_Aa = ZMatrixMap(A1_aaa, nao2, nao_);
  auto A_Q  = ZMatrixMap(A1_aaa, nao3, 1);

  auto B_aA = ZMatrixMap(B1_aaa, nao_, nao2);
  auto B_Aa = ZMatrixMap(B1_aaa, nao2, nao_);
  
  auto Uexch_i_jkl = DMatrixConstMap(Uijkl_exch_, nao_, nao3);

  for(int tau = 0; tau < ntau_; tau++) {
    auto g_ij = DMatrixMap(G.matptr(tau), nao_, nao_);
    auto gminus_ij = use_dlr_ ? DMatrixMap(DGit_rev + tau*nao2, nao_, nao_) : DMatrixMap(G.matptr(ntau_-tau-1), nao_, nao_);

    for(int i = 0; i < nao_; i++) {
      A_aA.noalias() = g_ij.transpose() * DMatrixConstMap(Uijkl_ + i*nao3, nao_, nao2);
      B_aA.noalias() = (A_Aa * g_ij).transpose();
      A_Aa.noalias() = B_Aa * gminus_ij.transpose();
      DMatrixMap(Sigma.matptr(tau) + i*nao_, nao_, 1).noalias() = (Uexch_i_jkl * A_Q).real();
    }
  }
}

void mol_gf2::solve_tv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();
  
  auto A_aA = ZMatrixMap(A1_aaa, nao_, nao2);
  auto A_Aa = ZMatrixMap(A1_aaa, nao2, nao_);
  auto A_Q  = ZMatrixMap(A1_aaa, nao3, 1);

  auto B_aA = ZMatrixMap(B1_aaa, nao_, nao2);
  auto B_Aa = ZMatrixMap(B1_aaa, nao2, nao_);
  
  auto Uexch_i_jkl = DMatrixConstMap(Uijkl_exch_, nao_, nao3);

  if(use_dlr_) {
    ZMatrixMap(Git_rev, ntau_, nao2).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * ZMatrixMap(G.tvptr(tstp,0), ntau_, nao2);
  }

  for(int tau = 0; tau < ntau_; tau++) {
    auto gtv = ZMatrixMap(G.tvptr(tstp,tau), nao_, nao_);
    auto gvt = use_dlr_ ? ZMatrixMap(Git_rev + tau*nao2, nao_, nao_) : ZMatrixMap(G.tvptr(tstp, ntau_-tau-1), nao_, nao_);

    for(int i = 0; i < nao_; i++) {
      // A^\rceil(t,t')knp = (G^\rceilT(t,t'))_kl Ui_lnp
      A_aA.noalias() = gtv.transpose() * DMatrixConstMap(Uijkl_ + i*nao3, nao_, nao2);

      // B^\rceil(t,t')_qkn = [A^\rceil(t,t')_knp * G^\rceil(t,t')_pq]^T
      B_aA.noalias() = (A_Aa * gtv).transpose();

      // A^\rceil(t,t')_qkm = B^\rceil(t,t')_qkn * (G^lceil(t',t)^T)_nm
      //               =                         * -sig (G^rceil(t,ntau-t')^* 
      A_Aa.noalias() = -sig * B_Aa * gvt.conjugate();

      // Sigma^<(t,t')i_j = Uexch_jqkm A^<(t,t')_qkm
      ZMatrixMap(Sigma.tvptr(tstp,tau) + i*nao_, nao_, 1).noalias() = Uexch_i_jkl * A_Q;
    }
    
    ZMatrixMap(Sigma.tvptr_trans(tstp,tau), nao_, nao_).noalias() = ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).transpose();
  }
}


void mol_gf2::solve_les(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();
  
  auto A_aA = ZMatrixMap(A1_aaa, nao_, nao2);
  auto A_Aa = ZMatrixMap(A1_aaa, nao2, nao_);
  auto A_Q  = ZMatrixMap(A1_aaa, nao3, 1);

  auto B_aA = ZMatrixMap(B1_aaa, nao_, nao2);
  auto B_Aa = ZMatrixMap(B1_aaa, nao2, nao_);
  
  auto Uexch_i_jkl = DMatrixConstMap(Uijkl_exch_, nao_, nao3);

  for(int t = 0; t <= tstp; t++) {
    auto gles = ZMatrixMap(G.curr_timestep_les_ptr(t,tstp), nao_, nao_);
    auto gret = ZMatrixMap(G.curr_timestep_ret_ptr(tstp,t), nao_, nao_);

    for(int i = 0; i < nao_; i++) {
      // A^<(t,t')knp = (G^<T(t,t'))_kl Ui_lnp
      A_aA.noalias() = gles.transpose() * DMatrixConstMap(Uijkl_ + i*nao3, nao_, nao2);

      // B^<(t,t')_qkn = [A^<(t,t')_knp * G^<(t,t')_pq]^T
      B_aA.noalias() = (A_Aa * gles).transpose();

      // A^<(t,t')_qkm = B^<(t,t')_qkn * (G^>(t',t)^T)_nm
      A_Aa.noalias() = B_Aa * (gret.transpose() - gles.conjugate());

      // Sigma^<(t,t')i_j = Uexch_jqkm A^<(t,t')_qkm
      ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp) + i*nao_, nao_, 1).noalias() = Uexch_i_jkl * A_Q;
    }
  }
}


void mol_gf2::solve_ret(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();
  
  auto A1_aA = ZMatrixMap(A1_aaa, nao_, nao2);
  auto A1_Aa = ZMatrixMap(A1_aaa, nao2, nao_);
  auto A1_Q  = ZMatrixMap(A1_aaa, nao3, 1);

  auto B1_aA = ZMatrixMap(B1_aaa, nao_, nao2);
  auto B1_Aa = ZMatrixMap(B1_aaa, nao2, nao_);
  
  auto B2_aA = ZMatrixMap(B2_aaa, nao_, nao2);
  auto B2_Aa = ZMatrixMap(B2_aaa, nao2, nao_);

  auto Uexch_i_jkl = DMatrixConstMap(Uijkl_exch_, nao_, nao3);

  for(int t = 0; t <= tstp; t++) {
    auto gles = ZMatrixMap(G.curr_timestep_les_ptr(t,tstp), nao_, nao_);
    auto gret = ZMatrixMap(G.curr_timestep_ret_ptr(tstp,t), nao_, nao_);

    for(int i = 0; i < nao_; i++) {
      // A^<(t,t')knp = (G^<T(t,t'))_kl Ui_lnp
      //              =-(G^<*(t',t))_kl Ui_lnp
      A1_aA.noalias() = -gles.conjugate() * DMatrixConstMap(Uijkl_ + i*nao3, nao_, nao2);

      // B^<(t,t')_qkn = [A^<(t,t')_knp * G^< (t,t')_pq]^T
      //               = [A^<(t,t')_knp *-G^<\dagger(t',t)_pq]^T
      B1_aA.noalias() = -(A1_Aa * gles.adjoint()).transpose();

      // B^R(t,t')_qkn = [A^<(t,t')_knp * G^R (t,t')_pq]^T
      B2_aA.noalias() = (A1_Aa * gret).transpose();

      // A^R(t,t')_knp = (G^RT(t,t'))_kl Ui_lnp
      A1_aA.noalias() = gret.transpose() * DMatrixConstMap(Uijkl_ + i*nao3, nao_, nao2);

      // B^R(t,t')_qkn += (A^R(t,t'))_knp G^>(t,t')_pq
      //               += (A^R(t,t'))_knp [ G^< (t,t')_pq + G^R(t,t')_pq ]
      //               += (A^R(t,t'))_knp [-G^<\dagger(t',t)_pq + G^R(t,t')_pq ]
      B2_aA.noalias() += (A1_Aa * (gret-gles.adjoint())).transpose();

      // C^R(t,t')_qkm = B^R(t,t')_qkn * G^<(t',t)_mn  + B^<(t,t')_qkn * G^A(t',t)_mn
      //               = B^R(t,t')_qkn * G^<T(t',t)_nm + B^<(t,t')_qkn * G^R*(t,t')_nm
      A1_Aa.noalias() = B2_Aa * gles.transpose() + B1_Aa * gret.conjugate();

      // Sigma^R(t,t')i_j = Uexch_jqkm A^R(t,t')_qkm
      ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t) + i*nao_, nao_, 1).noalias() = Uexch_i_jkl * A1_Q;
    }
  }
}

void mol_gf2::solve(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  if(tstp == -1) {
    solve_mat(Sigma, G, it2itr);
  }
  else {
    solve_ret(tstp, Sigma, G);
    solve_tv(tstp, Sigma, G, it2itr);
    solve_les(tstp, Sigma, G);
  }
}

void mol_gf2::solve_hf(int tstp, cplx *hmf, herm_matrix_hodlr &G) {
  int nao2 = nao_*nao_;
  int nao3 = nao2*nao_;

  G.get_les(tstp, tstp, rho.data());
  rho *= cplx(0.,G.sig());
  rho_T = rho.transpose();

  ZRowVectorMap rho_T_flat(rho_T.data(), nao2);
  for(int i = 0; i < nao_; ++i) {
    DMatrixConstMap U_i_lk_j = DMatrixConstMap(Uijkl_ + i*nao3, nao2, nao_);
    ZColVectorMap(hmf + tstp*nao2 + i*nao_, nao_).noalias() -= rho_T_flat * U_i_lk_j;
  }

  ZColVectorMap(hmf + tstp*nao2, nao2).noalias() += 2*DMatrixConstMap(Uijkl_, nao2, nao2) * ZColVectorMap(rho.data(), nao2);
}

void mol_gf2::solve_hf_mat(double *hmf, herm_matrix_hodlr &G, double *it2cf, int *it2cfp, double *dlrrf) {
  int nao2 = nao_*nao_;
  int nao3 = nao2*nao_;

  if(use_dlr_) {
    G.get_mat_tau(1., 1., it2cf, it2cfp, dlrrf, rho.data());
    rho *= -1.;
  }
  else {
    rho = -DMatrixMap(G.matptr(ntau_-1), nao_, nao_);
  }
  rho_T = rho.transpose();

  ZRowVectorMap rho_T_flat(rho_T.data(), nao2);
  for(int i = 0; i < nao_; ++i) {
    DMatrixConstMap U_i_lk_j = DMatrixConstMap(Uijkl_ + i*nao3, nao2, nao_);
    DColVectorMap(hmf + i*nao_, nao_).noalias() -= (rho_T_flat * U_i_lk_j).real();
  }

  DColVectorMap(hmf, nao2).noalias() += (2*DMatrixConstMap(Uijkl_, nao2, nao2) * ZColVectorMap(rho.data(), nao2)).real();
}
