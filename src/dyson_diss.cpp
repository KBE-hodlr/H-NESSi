#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

double dyson::dyson_timestep_diss(int tstp, herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);

  return err;
}

std::vector<double> dyson::dyson_timestep_errinfo_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  std::vector<double> ret(4);

  ret[1] = dyson_timestep_ret_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  ret[2] = dyson_timestep_tv_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  ret[3] = dyson_timestep_les_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);

  ret[0] = ret[1] + ret[2] + ret[3];
  return ret;
}

double dyson::dyson_timestep_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  err += dyson_timestep_tv_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  err += dyson_timestep_les_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_nobc_diss(int tstp, herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);
  err += dyson_timestep_tv_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_nobc_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_nobc_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  err += dyson_timestep_tv_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  err += dyson_timestep_les_nobc_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_2leg_diss(int tstp, herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_2leg_diss(tstp,G,mu,H.ptr(0),ellL.ptr(0),ellG.ptr(0),Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_2leg_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);
  err += dyson_timestep_les_2leg_diss(tstp,G,mu,H,ellL,ellG,Sigma,I,h);

  return err;
}

double dyson::dyson_start_tv_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_tv_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ret_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_ret_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ret_ntti_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  return dyson_start_ret_ntti_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_les_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_ntti_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_ntti_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_ntti_nobc_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_ntti_nobc_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_2leg_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_2leg_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}



double dyson::dyson_start_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  // We enforce Time-Translational Invariance for the first k timesteps

  // check TTI of H
  for(int t = 1; t <= k_; t++) {
    double diff = (ZMatrixMap(H, nao_, nao_) - ZMatrixMap(H + t*es_, nao_, nao_)).norm();
    if( diff > 1e-13 ) {
      throw std::invalid_argument("H is not TTI for 0 <= t <= k");
    }
  }

  // check TTI of Sigma Les
  for(int tp = 0; tp <= k_; tp++) {
    for(int t = 0; t <= tp; t++) {
      double diff = (ZMatrixMap(Sigma.curr_timestep_les_ptr(0,tp-t), nao_, nao_) - ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tp), nao_, nao_)).norm();
      if( diff > 1e-13 ) {
        throw std::invalid_argument("Sigma Les is not TTI for 0 <= t <= k " + std::to_string(t) + " " + std::to_string(tp));
      }
    }
  }

  // check TTI of Sigma Ret
  for(int t = 0; t <= k_; t++) {
    for(int tp = 0; tp <= t; tp++) {
      double diff = (ZMatrixMap(Sigma.curr_timestep_ret_ptr(t-tp,0), nao_, nao_) - ZMatrixMap(Sigma.curr_timestep_ret_ptr(t,tp), nao_, nao_)).norm();
      if( diff > 1e-13 ) {
        throw std::invalid_argument("Sigma Ret is not TTI for 0 <= t <= k");
      }
    }
  }


  double err = 0;
  err += dyson_start_tv_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  err += dyson_start_ret_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  err += dyson_start_les_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ntti_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  err += dyson_start_ret_ntti_diss(G, mu, H,ellL,ellG,Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_ntti_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_ntti_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_ntti_nobc_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  err += dyson_start_ret_ntti_diss(G, mu, H,ellL,ellG,Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti_nobc_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_ntti_nobc_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_ntti_nobc_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_2leg_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_ret_ntti_diss(G, mu, H,ellL,ellG,Sigma, I, h, imp_tp0);
  err += dyson_start_les_2leg_diss(G, mu, H,ellL,ellG,Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_2leg_diss(herm_matrix_hodlr &G, double mu, function &H, function &ellL, function &ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_2leg_diss(G, mu, H.ptr(0), ellL.ptr(0), ellG.ptr(0), Sigma, I, h);
}

} // namespace
