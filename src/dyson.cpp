#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace hodlr {

void dyson::write_timing(h5e::File &file, std::string label) {
  if(profile_) {
    h5e::dump(file, label + "/timing", timing);
  }
  else {
    std::cout << "write_timing() was called, however profiling was not enabled" << std::endl;
    std::cout << "To enable profiling, provide profile=true to the dyson constructor" << std::endl;
  }
  h5e::dump(file, label + "/rho_version", rho_version_);
}

void dyson::write_checkpoint_hdf5(h5e::File &file, std::string label) {
  h5e::dump(file, label + "/nao", nao_);
  h5e::dump(file, label + "/es", es_);
  h5e::dump(file, label + "/k", k_);
  h5e::dump(file, label + "/nt", nt_);
  h5e::dump(file, label + "/rho_version", rho_version_);
  h5e::dump(file, label + "/profile", profile_);
  if(profile_) h5e::dump(file, label + "/timing", timing);
}


dyson::dyson(int nt, int nao, int k, dlr_info &dlr, bool rho_version, bool profile) :
dlr_(dlr)
 {
  nao_ = nao;
  es_ = nao * nao;
  k_ = k;
  nt_ = nt;

  r_ = dlr.r();
  ntau_ = dlr.r();
  beta_ = dlr.beta();
  xi_ = dlr.xi();

  rho_version_ = rho_version;

  NTauTmp_.resize(r_ * es_);
  NTauTmp2_.resize(r_ * es_);
  DNTauTmp_.resize(r_ * es_);
  DNTauTmp2_.resize(r_ * es_);
  epsnao_tmp_.resize((nt_+1)/2 * nao_);
  epsnao_tmp_2_.resize((nt_+1)/2 * nao_);

  Q_.resize((nt_+1) * es_);
  X_.resize((nt_+1) * es_);
  M_.resize(k * k * es_);
  iden_.resize(es_);
  bound_.resize(es_);

  ZMatrixMap(iden_.data(), nao_, nao_).noalias() = ZMatrix::Identity(nao_, nao_);

  profile_ = profile;
  timing = DMatrix::Zero(profile_ ? nt : 0, 25);
}

void dyson::print_memory_usage(void) {
  
    double D = (double)sizeof(double)/(1e9);
    double Z = 2*D;
    double I = (double)sizeof(int)/(1e9);

    double total = 0;
    double tautmp = 0;
    double epsnaotmp = 0;
    double qxm = 0;
    double tim = 0;

    tautmp = Z * 2 * r_ * es_;
    epsnaotmp = Z * 2 * (nt_+1)/2 * nao_;
    qxm = Z * 2 * (nt_+1) * es_ + Z*k_*k_*es_;
    tim = D * (profile_ ? nt_ : 0) * 25;
    total = tautmp + epsnaotmp + qxm + tim;

    const int total_width = 10;
    const int precision = 2;

    std::cout << "==========================================" << std::endl;
    std::cout << "==        Memory usage for dyson        ==" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "==                                      ==" << std::endl;
    printf(      "==  DLR buffers        :%*.*f MB   ==", total_width, precision, tautmp*1000); std::cout << std::endl;
    printf(      "==  Mat Mul buffers    :%*.*f MB   ==", total_width, precision, epsnaotmp*1000); std::cout << std::endl;
    printf(      "==  Solver buffers     :%*.*f MB   ==", total_width, precision, qxm*1000); std::cout << std::endl;
    printf(      "==  Timing buffer      :%*.*f MB   ==", total_width, precision, tim*1000); std::cout << std::endl;
    std::cout << "==                                      ==" << std::endl;
    printf(      "==  Total              :%*.*f MB   ==", total_width, precision, total*1000); std::cout << std::endl;
    std::cout << "==                                      ==" << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;
}

dyson::dyson(h5e::File &in, std::string label, dlr_info &dlr) :
dlr_(dlr)
 {
  nao_ = in.getDataSet(label + std::string("/nao")).read<int>();
  es_ = in.getDataSet(label + std::string("/es")).read<int>();
  k_ = in.getDataSet(label + std::string("/k")).read<int>();
  nt_ = in.getDataSet(label + std::string("/nt")).read<int>();

  r_ = dlr.r();
  ntau_ = dlr.r();
  beta_ = dlr.beta();
  xi_ = dlr.xi();

  rho_version_ = in.getDataSet(label + std::string("/rho_version")).read<bool>();

  NTauTmp_.resize(r_ * es_);
  NTauTmp2_.resize(r_ * es_);
  DNTauTmp_.resize(r_ * es_);
  DNTauTmp2_.resize(r_ * es_);
  epsnao_tmp_.resize((nt_+1)/2 * nao_);
  epsnao_tmp_2_.resize((nt_+1)/2 * nao_);

  Q_.resize((nt_+1) * es_);
  X_.resize((nt_+1) * es_);
  M_.resize(k_ * k_ * es_);
  iden_.resize(es_);
  bound_.resize(es_);

  ZMatrixMap(iden_.data(), nao_, nao_).noalias() = ZMatrix::Identity(nao_, nao_);

  profile_ = in.getDataSet(label + std::string("/profile")).read<bool>();
  timing = profile_ ? in.getDataSet(label + std::string("/timing")).read<DMatrix>() : DMatrix::Zero(0,25);
}

double dyson::dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return err;
}

std::vector<double> dyson::dyson_timestep_errinfo(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
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

  ret[1] = dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  ret[2] = dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  ret[3] = dyson_timestep_les(tstp,G,mu,H,Sigma,I,h);

  ret[0] = ret[1] + ret[2] + ret[3];
  return ret;
}

double dyson::dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les(tstp,G,mu,H,Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_nobc(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les_nobc(tstp,G,mu,H,Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_2leg(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return err;
}

double dyson::dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  int size1 = G.size1(), k = I.k();  
  assert(G.size1() == Sigma.size1());
  assert(G.r() == Sigma.r());
  assert(G.nt() >= tstp);
  assert(Sigma.nt() >= tstp);
  assert(tstp > k);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;


  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les_2leg(tstp,G,mu,H,Sigma,I,h);

  return err;
}

double dyson::dyson_start_tv(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_tv(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ret(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_ret(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  return dyson_start_ret_ntti(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_les(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_ntti(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_ntti_nobc(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  return dyson_start_les_2leg(G, mu, H.ptr(0), Sigma, I, h);
}



double dyson::dyson_start(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret(G, mu, H, Sigma, I, h);
  err += dyson_start_les(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start(G, mu, H.ptr(0), Sigma, I, h);
}

double dyson::dyson_start_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_ntti(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti_nobc(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_ntti_nobc(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
}

double dyson::dyson_start_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_2leg(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return err;
}

double dyson::dyson_start_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  return dyson_start_2leg(G, mu, H.ptr(0), Sigma, I, h);
}

/*
void dyson::dipole_step(int tstp, function &dfield, herm_matrix_hodlr &Gu, herm_matrix_hodlr &Gd, double *dipole, double l, double n, double h, Integration::Integrator &I) {
 
  if( tstp <= k_) { // Need to use poly_diff coeff
    for(int d = 0; d < 3; d++) {
      dfield.ptr(tstp)[d] = 0;

      for(int i = 0; i <= k_; i++) {
        cplx yi = DMatrixMap(dipole + d * es_, nao_, nao_).cwiseProduct(
                      ZMatrixMap(Gu.curr_timestep_les_ptr(i,i), nao_, nao_).transpose()
                    + ZMatrixMap(Gd.curr_timestep_les_ptr(i,i), nao_, nao_).transpose()).sum();
        dfield.ptr(tstp)[d] += 1./h * I.poly_diff(tstp, i) * yi;
      }
      dfield.ptr(tstp)[d] *= cplx(0., Gu.sig() * 1.) * l * n * 2. * PI / 137.035999206;
    }
  }
  else { // Use bd_weights
    for(int d = 0; d < 3; d++) {
      dfield.ptr(tstp)[d] = 0;
      for(int i = 0; i <= k_; i++) {
        cplx yi = DMatrixMap(dipole + d * es_, nao_, nao_).cwiseProduct(
                      ZMatrixMap(Gu.curr_timestep_les_ptr(tstp-i,tstp-i), nao_, nao_).transpose()
                    + ZMatrixMap(Gd.curr_timestep_les_ptr(tstp-i,tstp-i), nao_, nao_).transpose()).sum();
        dfield.ptr(tstp)[d] += 1./h * I.bd_weights(i) * yi;
      }

      Gu.get_les(tstp-k_-1, tstp-k_-1, Q_.data());
      Gd.get_les(tstp-k_-1, tstp-k_-1, X_.data());
      cplx yi = DMatrixMap(dipole + d * es_, nao_, nao_).cwiseProduct(
                      ZMatrixMap(Q_.data(), nao_, nao_).transpose()
                    + ZMatrixMap(X_.data(), nao_, nao_).transpose()).sum();
      dfield.ptr(tstp)[d] += 1./h * I.bd_weights(k_+1) * yi;
      
      dfield.ptr(tstp)[d] *= cplx(0.,Gu.sig() * 1.) * l * n * 2. * PI / 137.035999206;
    }
  }
}
*/

} // namespace hodlr
