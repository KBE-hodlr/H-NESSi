#include "hubb_gf2_ext.hpp"

using namespace hodlr;

hubb_gf2_ext::hubb_gf2_ext(int ntau, int nao, double U, double gamma, bool use_dlr) {
  nao_ = nao;
  ntau_ = ntau;
  rho = ZMatrix(nao_, nao_);
  tmp = ZMatrix(nao_, nao_);
  tmp2 = ZMatrix(nao_, nao_);

  W = DMatrix::Zero(nao_, nao_);
  for(int i = 0; i < nao_; i++) {
    for(int j = 0; j < nao_; j++) {
      if(i != j) W(i,j) = 1./(std::abs(i-j));
    }
  }

  U_ = U;
  gamma_ = gamma;

  use_dlr_ = use_dlr;

  if(use_dlr) {
    Git_rev = new cplx[ntau*nao*nao];
    DGit_rev = reinterpret_cast<double *>(Git_rev);
  }
  else {
    Git_rev = 0;
    DGit_rev = 0;
  }
}

hubb_gf2_ext::~hubb_gf2_ext() {
  delete [] Git_rev;
}

void hubb_gf2_ext::solve_mat(herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();

  if(use_dlr_) {
    DMatrixMap(DGit_rev, ntau_, nao2).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * DMatrixMap(G.mat(), ntau_, nao2);
  }
  

  for(int tau = 0; tau < ntau_; tau++) {
    auto g_ij = DMatrixMap(G.matptr(tau), nao_, nao_);
    auto gminus_ij = use_dlr_ ? DMatrixMap(DGit_rev + tau*nao2, nao_, nao_) : DMatrixMap(G.matptr(ntau_-tau-1), nao_, nao_);

    // Hubbard Bubble    
    DMatrixMap(Sigma.matptr(tau), nao_, nao_) = U_ * U_ * (g_ij.cwiseProduct(g_ij)).cwiseProduct(gminus_ij.transpose());

    // Extended Bubble
    tmp = g_ij.cwiseProduct(gminus_ij.transpose());
    tmp2 = W * tmp;
    tmp = tmp2 * W.transpose();
    DMatrixMap(Sigma.matptr(tau), nao_, nao_).noalias() += 2 * U_ * U_ * gamma_ * gamma_ * g_ij.cwiseProduct(tmp).real();

    // Extended Exchange
    for(int i = 0; i < nao_; i++) {
      tmp = W.row(i).transpose() * g_ij.row(i);
      tmp2 = tmp.cwiseProduct(gminus_ij.transpose());
      tmp = tmp2 * W;
      tmp2 = g_ij.transpose() * tmp;
      for(int j = 0; j < nao_; j++) {
        Sigma.matptr(tau)[i*nao_ + j] += -U_ * U_ * gamma_ * gamma_ * tmp2(j,j).real();
      }
    }

    // Mixed Bubble with Ext on left
    tmp = g_ij.cwiseProduct(gminus_ij.transpose());
    tmp2 = W * tmp;
    tmp = g_ij.cwiseProduct(tmp2);
    DMatrixMap(Sigma.matptr(tau), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp.real();

    // Mixed Bubble with Ext on right
    tmp = g_ij.cwiseProduct(gminus_ij.transpose());
    tmp2 = tmp * W.transpose();
    tmp = g_ij.cwiseProduct(tmp2);
    DMatrixMap(Sigma.matptr(tau), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp.real();
  }
}



void hubb_gf2_ext::solve_tv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();

  
  if(use_dlr_) {
    ZMatrixMap(Git_rev, ntau_, nao2).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * ZMatrixMap(G.tvptr(tstp,0), ntau_, nao2);
  }

  for(int tau = 0; tau < ntau_; tau++) {
    auto gtv = ZMatrixMap(G.tvptr(tstp,tau), nao_, nao_);
    auto gvt = use_dlr_ ? ZMatrixMap(Git_rev + tau*nao2, nao_, nao_) : ZMatrixMap(G.tvptr(tstp, ntau_-tau-1), nao_, nao_);

    // Hubbard Bubble    
    ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).noalias() = -sig * U_ * U_ * (gtv.cwiseProduct(gtv)).cwiseProduct(gvt.conjugate());

    // Extended Bubble
    tmp = -sig * gtv.cwiseProduct(gvt.conjugate());
    tmp2 = W * tmp;
    tmp = tmp2 * W.transpose();
    ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).noalias() += 2 * U_ * U_ * gamma_ * gamma_ * gtv.cwiseProduct(tmp);

    // Extended Exchange
    for(int i = 0; i < nao_; i++) {
      tmp = W.row(i).transpose() * gtv.row(i);
      tmp2 = -sig * tmp.cwiseProduct(gvt.conjugate());
      tmp = tmp2 * W;
      tmp2 = gtv.transpose() * tmp;
      for(int j = 0; j < nao_; j++) {
        Sigma.tvptr(tstp,tau)[i*nao_ + j] += -U_ * U_ * gamma_ * gamma_ * tmp2(j,j);
      }
    }

    // Mixed Bubble with Ext on left
    tmp = -sig * gtv.cwiseProduct(gvt.conjugate());
    tmp2 = W * tmp;
    tmp = gtv.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;

    // Mixed Bubble with Ext on right
    tmp = -sig * gtv.cwiseProduct(gvt.conjugate());
    tmp2 = tmp * W.transpose();
    tmp = gtv.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;


    // additional transpose
    ZMatrixMap(Sigma.tvptr_trans(tstp,tau), nao_, nao_).noalias() = ZMatrixMap(Sigma.tvptr(tstp,tau), nao_, nao_).transpose();
  }
}


void hubb_gf2_ext::solve_les(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig = G.sig();
  auto ggrt = ZMatrixMap(rho.data(), nao_, nao_);

  for(int t = 0; t <= tstp; t++) {
    auto gles = ZMatrixMap(G.curr_timestep_les_ptr(t,tstp), nao_, nao_);
    auto gret = ZMatrixMap(G.curr_timestep_ret_ptr(tstp,t), nao_, nao_);
    ggrt = gret - gles.adjoint();

    // Hubbard Bubble
    ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), nao_, nao_).noalias() = U_ * U_ * (gles.cwiseProduct(gles)).cwiseProduct(ggrt.transpose());

    // Extended Bubble
    tmp = gles.cwiseProduct(ggrt.transpose());
    tmp2 = W * tmp;
    tmp = tmp2 * W.transpose();
    ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), nao_, nao_).noalias() += 2 * U_ * U_ * gamma_ * gamma_ * gles.cwiseProduct(tmp);

    // Extended Exchange
    for(int i = 0; i < nao_; i++) {
      tmp = W.row(i).transpose() * gles.row(i);
//      std::cout << "This product should be a 12x12 matrix" << std::endl;
//      std::cout << W.row(i).transpose() * gles.row(i) << std::endl;
      tmp2 = tmp.cwiseProduct(ggrt.transpose());
      tmp = tmp2 * W;
      tmp2 = gles.transpose() * tmp;
      for(int j = 0; j < nao_; j++) {
        Sigma.curr_timestep_les_ptr(t,tstp)[i*nao_ + j] += -U_ * U_ * gamma_ * gamma_ * tmp2(j,j);
      }
    }

    // Mixed Bubble with Ext on left
    tmp = gles.cwiseProduct(ggrt.transpose());
    tmp2 = W * tmp;
    tmp = gles.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;

    // Mixed Bubble with Ext on right
    tmp = gles.cwiseProduct(ggrt.transpose());
    tmp2 = tmp * W.transpose();
    tmp = gles.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;

  }
}


void hubb_gf2_ext::solve_ret(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G) {
  int nao2 = nao_ * nao_;
  int nao3 = nao2 * nao_;
  int sig  = G.sig();
  auto ggrt = ZMatrixMap(rho.data(), nao_, nao_);

  for(int t = 0; t <= tstp; t++) {
    auto gles = ZMatrixMap(G.curr_timestep_les_ptr(t,tstp), nao_, nao_);
    auto gret = ZMatrixMap(G.curr_timestep_ret_ptr(tstp,t), nao_, nao_);
    ggrt = gret - gles.adjoint();

    // Hubbard Bubble
    ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t), nao_, nao_).noalias()  = U_ * U_ * (ggrt.cwiseProduct(ggrt)).cwiseProduct(gles.transpose());

    // Extended Bubble
    tmp = ggrt.cwiseProduct(gles.transpose());
    tmp2 = W * tmp;
    tmp = tmp2 * W.transpose();
    ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t), nao_, nao_).noalias() += 2 * U_ * U_ * gamma_ * gamma_ * ggrt.cwiseProduct(tmp);

    // Extended Exchange
    for(int i = 0; i < nao_; i++) {
      tmp = W.row(i).transpose() * ggrt.row(i);
      tmp2 = tmp.cwiseProduct(gles.transpose());
      tmp = tmp2 * W;
      tmp2 = ggrt.transpose() * tmp;
      for(int j = 0; j < nao_; j++) {
        Sigma.curr_timestep_ret_ptr(tstp,t)[i*nao_ + j] += -U_ * U_ * gamma_ * gamma_ * tmp2(j,j);
      }
    }

    // Mixed Bubble with Ext on left
    tmp = ggrt.cwiseProduct(gles.transpose());
    tmp2 = W * tmp;
    tmp = ggrt.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;

    // Mixed Bubble with Ext on right
    tmp = ggrt.cwiseProduct(gles.transpose());
    tmp2 = tmp * W.transpose();
    tmp = ggrt.cwiseProduct(tmp2);
    ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t), nao_, nao_).noalias() += U_ * U_ * gamma_ * tmp;

    // Grt -> Ret
    ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,t), nao_, nao_).noalias() += ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tstp), nao_, nao_).adjoint();
  }
}


void hubb_gf2_ext::solve(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double *it2itr) {
  if(tstp == -1) {
    solve_mat(Sigma, G, it2itr);
  }
  else {
    solve_les(tstp, Sigma, G);
    solve_tv(tstp, Sigma, G, it2itr);
    solve_ret(tstp, Sigma, G);
  }
}


void hubb_gf2_ext::solve_hf(int tstp, cplx *hmf, herm_matrix_hodlr &G) {
  int nao2 = nao_*nao_;
  int nao3 = nao2*nao_;

  G.get_les(tstp, tstp, rho.data());
  rho *= cplx(0.,G.sig());

  // Hubb Hartree
  for(int i = 0; i < nao_; i++) {
    hmf[tstp*nao2 + i*nao_ + i] += U_ * rho(i,i);
  }

  // Ext Hartree
  ZMatrixMap(hmf + tstp*nao2, nao_, nao_).diagonal() += 2. * gamma_ * U_ * W * rho.diagonal();

  // Ext Fock
  ZMatrixMap(hmf + tstp*nao2, nao_, nao_) += -gamma_ * U_ * W.cwiseProduct(rho);
}


void hubb_gf2_ext::solve_hf_mat(double *hmf, herm_matrix_hodlr &G, double *it2cf, int *it2cfp, double *dlrrf) {
  int nao2 = nao_*nao_;
  int nao3 = nao2*nao_;

  if(use_dlr_) {
    G.get_mat_tau(1., 1., it2cf, it2cfp, dlrrf, rho.data());
    rho *= -1.;
  }
  else {
    rho = -DMatrixMap(G.matptr(ntau_-1), nao_, nao_);
  }

  // Hubb Hartree
  for(int i = 0; i < nao_; i++) {
    hmf[i*nao_ + i] += U_ * rho(i,i).real();
  }

  // Ext Hartree
  DMatrixMap(hmf, nao_, nao_).diagonal() += (2. * gamma_ * U_ * W * rho.diagonal()).real();

  // Ext Fock
  DMatrixMap(hmf, nao_, nao_) += (-gamma_ * U_ * W.cwiseProduct(rho)).real();

}

