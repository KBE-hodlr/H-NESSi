#include "SC_gf2.hpp"

using namespace hodlr;

SC_gf2::SC_gf2(dlr_info &dlr) :
dlr_(dlr)
{
  sigma3_ = ZMatrix::Identity(2,2);
  sigma3_(1,1) = -1;
}

// ###########################################################################################

void SC_gf2::solve_Delta(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  if(tstp == -1) { Delta_mat(t0, G, Sigma); }
  else { Delta_tstp(tstp, t0, G, Sigma); }
}

void SC_gf2::Delta_mat(function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  for(int tau = 0; tau < G.ntau(); tau++) {
    Sigma.map_mat(tau).noalias() += (t0.get_map(-1) * sigma3_ * G.map_mat(tau) * sigma3_ * t0.get_map(-1).conjugate()).real();
  }
}

void SC_gf2::Delta_tstp(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  ZMatrix sGs = ZMatrix::Zero(2,2);
  for(int t = 0; t <= tstp; t++) {
    sGs.noalias() = sigma3_ * G.map_ret_curr(tstp, t) * sigma3_;
    Sigma.map_ret_curr(tstp, t).noalias() += 0.5 * t0.get_map(tstp) * sGs * t0.get_map(t).conjugate();
    Sigma.map_ret_curr(tstp, t).noalias() += 0.5 * t0.get_map(tstp).conjugate() * sGs * t0.get_map(t);

    sGs.noalias() = sigma3_ * G.map_les_curr(t, tstp) * sigma3_;
    Sigma.map_les_curr(t, tstp).noalias() += 0.5 * t0.get_map(t) * sGs * t0.get_map(tstp).conjugate();
    Sigma.map_les_curr(t, tstp).noalias() += 0.5 * t0.get_map(t).conjugate() * sGs * t0.get_map(tstp);
  }

  for(int tau = 0; tau < G.ntau(); tau++) {
    sGs.noalias() = sigma3_ * G.map_tv(tstp, tau) * sigma3_;
    Sigma.map_tv(tstp, tau).noalias() += 0.5 * t0.get_map(tstp) * sGs * t0.get_map(-1).conjugate();
    Sigma.map_tv(tstp, tau).noalias() += 0.5 * t0.get_map(tstp).conjugate() * sGs * t0.get_map(-1);
  }
}

void SC_gf2::solve_Delta_LmR(int tstp, function &t0, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  ZMatrix sGs = ZMatrix::Zero(2,2);
  for(int t = 0; t <= tstp; t++) {
    sGs.noalias() = sigma3_ * G.map_ret_curr(tstp, t) * sigma3_;
    Sigma.map_ret_curr(tstp, t).noalias() += -0.5 * t0.get_map(tstp) * sGs * t0.get_map(t).conjugate() + 0.5 * t0.get_map(tstp).conjugate() * sGs * t0.get_map(t);

    sGs.noalias() = sigma3_ * G.map_les_curr(t, tstp) * sigma3_;
    Sigma.map_les_curr(t, tstp).noalias() += -0.5 * t0.get_map(t) * sGs * t0.get_map(tstp).conjugate() + 0.5 * t0.get_map(t).conjugate() * sGs * t0.get_map(tstp);
  }

  for(int tau = 0; tau < G.ntau(); tau++) {
    sGs.noalias() = sigma3_ * G.map_tv(tstp, tau) * sigma3_;
    Sigma.map_tv(tstp, tau).noalias() += -0.5 * t0.get_map(tstp) * sGs * t0.get_map(-1).conjugate() + 0.5 * t0.get_map(tstp).conjugate() * sGs * t0.get_map(-1);
  }
}

// ##########################################################################################

void SC_gf2::solve_Sigma(int tstp, function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  if(tstp == -1) { Sigma_mat(U, G, Sigma); }
  else { Sigma_tstp(tstp, U, G, Sigma); }
}

void SC_gf2::Sigma_mat(function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  DMatrix GM_reversed(G.ntau(), 4);
  G.get_mat_reversed(dlr_, GM_reversed);

  for(int tau = 0; tau < G.ntau(); tau++) {
    auto SM_map = Sigma.map_mat(tau);
    auto GM_map = G.map_mat(tau);
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        SM_map(i,j) += (U(-1,0,0) * U(-1,0,0) * GM_map(i,j) * GM_map(1-i,1-j) * GM_reversed(tau, (1-j)*2 + (1-i))).real();
        SM_map(i,j) -= (U(-1,0,0) * U(-1,0,0) * GM_map(i,1-j) * GM_map(1-i,j) * GM_reversed(tau, (1-j)*2 + (1-i))).real();
      }
    }
  }
}

void SC_gf2::Sigma_tstp(int tstp, function &U, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  for(int t = 0; t <= tstp; t++) {
    auto SR = Sigma.map_ret_curr(tstp, t);
    auto GR = G.map_ret_curr(tstp, t);
    auto SL = Sigma.map_les_curr(t, tstp);
    auto GL = G.map_les_curr(t, tstp);
    auto GG = GR - GL.adjoint();

    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        SL(i,j) += U(tstp,0,0) * U(t,0,0) * GL(i,j) * GL(1-i,1-j) * GG(1-j,1-i);
        SL(i,j) -= U(tstp,0,0) * U(t,0,0) * GL(i,1-j) * GG(1-j,1-i) * GL(1-i,j);
        SR(i,j) += U(tstp,0,0) * U(t,0,0) * (GG(i,j) * GG(1-i,1-j) * GL(1-j,1-i) + std::conj(GL(j,i) * GL(1-j,1-i) * GG(1-i,1-j)));
        SR(i,j) -= U(tstp,0,0) * U(t,0,0) * (GG(i,1-j) * GL(1-j,1-i) * GG(1-i,j) + std::conj(GL(1-j,i) * GG(1-i,1-j) * GL(j,1-i)));
      }
    }
  }

  ZMatrix GVT(G.ntau(), 4);
  G.get_vt(tstp, dlr_, GVT);
  for(int tau = 0; tau < G.ntau(); tau++) {
    auto GTV_map = G.map_tv(tstp, tau);
    auto STV_map = Sigma.map_tv(tstp, tau);
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 2; j++) {
        STV_map(i,j) += U(tstp,0,0) * U(-1,0,0) * GTV_map(i,j) * GTV_map(1-i,1-j) * GVT(tau,(1-j)*2 + (1-i));
        STV_map(i,j) -= U(tstp,0,0) * U(-1,0,0) * GTV_map(i,1-j) * GVT(tau, (1-j)*2 + (1-i)) * GTV_map(1-i,j);
      }
    }
  }
}

// ###########################################################################################

void SC_gf2::solve_Sigma_Fock(int tstp, function &U, herm_matrix_hodlr &G, function &H) {
  ZMatrix rho(2,2);
  G.density_matrix(tstp, dlr_, rho);

  for(int i = 0; i < 2; i++) {
    H(tstp,i,1-i) = -U(tstp,0,0) * rho(i,1-i);
  }
}

