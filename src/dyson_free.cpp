#include "h_nessi/dyson.hpp"
#include <iomanip>
#include <cmath>

using namespace std::chrono;

namespace h_nessi {

#define EXPMAX 100

// 1/(1+exp(omega*beta))
// UT-3 (tier 1): beta > 0; bose: omega*beta != 0 (1/arg below; the callers check h - mu > 0 for bosons, DY-M10/F10)
double fermi(double beta, double omega) {
    HNESSI_ASSERT(beta > 0, "fermi", HNESSI_KV(beta));
    double arg = omega * beta;
    if (fabs(arg) > EXPMAX) {
        return (arg > 0.0 ? 0.0 : 1.0);
    } else {
        return 1.0 / (1.0 + exp(arg));
    }
}


// 1/(1+exp(omega*beta))
DColVector fermi(double beta, DColVector &omega){
   int size=omega.size();
   DColVector tmp(size);
   for(int i=0;i<size;i++){
      tmp(i)=fermi(beta,omega(i));
   }
   return tmp;
}


// exp(w*t)/(1+exp(w*b))
double fermi_exp(double beta, double tau, double omega) {
    if (omega < 0) {
        return exp(omega * tau) *
               fermi(beta, omega); // exp(w*t)/(1+exp(b*w)) always OK for w<0
    } else {
        return exp((tau - beta) * omega) *
               fermi(beta, -omega); // exp((t-b)*w)/(1+exp(-w*b))
    }
}


// exp(w*t)/(1+exp(w*b))
DColVector fermi_exp(double beta,double tau,DColVector &omega){
   int size=omega.size();
   DColVector tmp(size);

   for(int i=0;i<size;i++){
      tmp(i)=fermi_exp(beta,tau,omega(i));
   }
   return tmp;
}


// 1/(exp(w*b)-1)
double bose(double beta, double omega) {
    HNESSI_ASSERT(beta > 0 && omega * beta != 0, "bose", HNESSI_KV(beta), HNESSI_KV(omega));
    double arg = omega * beta;
    if (arg < 0)
        return (-1.0 - bose(beta, -omega));
    if (fabs(arg) > EXPMAX) {
        return 0.0;
    } else if (arg < 1e-10) {
        return 1.0 / arg;
    } else {
        return 1.0 / (exp(arg) - 1.0);
    }
}


// 1/(exp(w*b)-1)
DColVector bose(double beta,DColVector &omega){
   int size=omega.size();
   DColVector tmp(size);
   for(int i=0;i<size;i++){
      tmp(i)=bose(beta,omega(i));
   }
   return tmp;
}


// exp(w*t)/(exp(w*b)-1)
double bose_exp(double beta, double tau, double omega) {
    if (omega < 0)
        return exp(tau * omega) * bose(beta, omega);
    else
        return -exp((tau - beta) * omega) * bose(beta, -omega);
}


// exp(w*t)/(exp(w*b)-1)
DColVector bose_exp(double beta,double tau,DColVector &omega){
   int size=omega.size();
   DColVector tmp(size);

   for(int i=0;i<size;i++){
      tmp(i)=bose_exp(beta,tau,omega(i));
   }
   return tmp;
}


void dyson::green_from_H_mat(double *g0, double mu, ZMatrix H) {
  call_scope cs(*this);
  const char *where = "dyson::green_from_H_mat";
  if(cs.outer()) {
    HNESSI_REQUIRE(H.rows() == nao_ && H.cols() == nao_ && std::isfinite(mu), where, HNESSI_KV(H.rows()),
                   HNESSI_KV(H.cols()), HNESSI_KV(nao_), HNESSI_KV(mu));
    hermitize_(H.data() + es_, -1, -1, where);   // D-11 on the local copy (H is passed by value), recorded as h(-1)
    require_real_(H.data(), where);              // DY-F9 (D1): G^M is real
  }
  double tau;
  // Make Hamiltonian and solve eigen problem
  ZMatrix Hmu = mu * ZMatrixMap(iden_.data(), nao_, nao_) - H;
  Eigen::SelfAdjointEigenSolver<ZMatrix> eigensolver(Hmu);
  ZMatrix evec0(nao_, nao_);
  DColVector eval0(nao_), eval0m(nao_);
  evec0 = eigensolver.eigenvectors();
  eval0 = eigensolver.eigenvalues();
  eval0m = -eval0;
  require_bosonic_spectrum_(eval0, "dyson::green_from_H_mat");   // DY-M10/DY-F10 (also inside dyson_mat)

  // Matsubara
  for(int m = 0; m < r_; m++) {
    tau = dlr_.it0B(m);
    if(xi_ == -1){
      DMatrixMap(g0 + m*nao_*nao_, nao_, nao_) = (-evec0 * fermi_exp(beta_,tau,eval0).asDiagonal() * evec0.adjoint()).real();
    } else if(xi_ == 1){
      DMatrixMap(g0 + m*nao_*nao_, nao_, nao_) =  (evec0 * bose_exp(beta_,tau,eval0).asDiagonal() * evec0.adjoint()).real();
    }
  }
}


// DY-F1..DY-F5 (tier 2; H11: tmax <= k and G.tstpmk() == 0, the rows 0..tmax live in the circular window)
void dyson::check_free_args_(const char *where, herm_matrix_hodlr &G, double mu, const ZMatrix &H, double h, int tmax) {
  HNESSI_REQUIRE(G.sig() == xi_, where, HNESSI_KV(G.sig()), HNESSI_KV(xi_),
                 detail::note{"Fermion/Boson xi of G does not agree with the dlr_info of this dyson"});
  HNESSI_REQUIRE(G.r() == r_, where, HNESSI_KV(G.r()), HNESSI_KV(r_),
                 detail::note{"Ntau of G does not agree with r of libdlr"});
  HNESSI_REQUIRE(G.size1() == nao_ && G.size2() == nao_ && H.rows() == nao_ && H.cols() == nao_, where,
                 HNESSI_KV(G.size1()), HNESSI_KV(H.rows()), HNESSI_KV(H.cols()), HNESSI_KV(nao_));
  HNESSI_REQUIRE(0 <= tmax && tmax <= k_ && tmax < G.nt(), where, HNESSI_KV(tmax), HNESSI_KV(k_), HNESSI_KV(G.nt()),
                 detail::note{"green_from_H* fills only the bootstrap rows 0..tmax <= k (circular buffers, ISSUES H11); "
                              "for later times propagate with dyson_timestep"});
  HNESSI_REQUIRE_STATE(G.tstpmk() == 0, where, HNESSI_KV(G.tstpmk()),
                       detail::note{"green_from_H* writes rows 0..tmax as if the window were 0..k"});
  HNESSI_REQUIRE(std::isfinite(h) && h > 0 && std::isfinite(mu), where, HNESSI_KV(h), HNESSI_KV(mu));
}

void dyson::green_from_H(herm_matrix_hodlr &G, double mu, function &H, double h, int tmax, bool inc_mat) {
  call_scope cs(*this);
  const char *where = "dyson::green_from_H";
  if(cs.outer()) {
    check_function_(where, H, -1);
    check_free_args_(where, G, mu, H.get_map(-1), h, tmax);
    hermitize_(H.ptr(0), -1, -1, where);   // D-11, in place on slice -1 of the user's function
    if(inc_mat) require_real_(H.ptr(-1), where);   // DY-F9 (D1): G^M is stored real
  }
  green_from_H( G, mu, H.get_map(-1), h, tmax, inc_mat);
}

  // Gives free green's function from constant hamiltonian
// G^M(\tau) = s f_s(mu-h) exp((mu-h)\tau)
// G^{TV}(n,\tau) = -is U_{n,0} f_s(h-mu) exp((h-mu)\tau)
// G^R(n,j) = -iU_{n,j} = U_{n,0} (U_{j,0})^\dagger
// G^L(j,n) = -siU_{j,0} f_s(h-mu) (U_{n,0})^\dagger
void dyson::green_from_H(herm_matrix_hodlr &G, double mu, ZMatrix H, double h, int tmax, bool inc_mat) {
  call_scope cs(*this);
  const char *where = "dyson::green_from_H";
  if(cs.outer()) {
    check_free_args_(where, G, mu, H, h, tmax);
    hermitize_(H.data() + es_, -1, -1, where);   // D-11 on the local copy (H is passed by value), recorded as h(-1)
    // DY-F9 (D1, decided as dyson_mat): G^M is stored real, a complex h would silently lose Im G^M
    if(inc_mat) require_real_(H.data(), where);
  }

  int sign = G.sig();
  double tau, t;

  // Make Hamiltonian and solve eigen problem
  ZMatrix Hmu = mu * ZMatrixMap(iden_.data(), nao_, nao_) - H;
  Eigen::SelfAdjointEigenSolver<ZMatrix> eigensolver(Hmu);

  ZMatrix evec0(nao_, nao_);
  DColVector eval0(nao_), eval0m(nao_);

  evec0 = eigensolver.eigenvectors();
  eval0 = eigensolver.eigenvalues();
  eval0m = -eval0;
  require_bosonic_spectrum_(eval0, "dyson::green_from_H");   // DY-F10

  // Matsubara
  if(inc_mat) {
    for(int m = 0; m < G.r(); m++) {
      tau = dlr_.it0B(m);
      if(sign == -1){
        DMatrixMap(G.matptr(m), nao_, nao_) = (-evec0 * fermi_exp(beta_,tau,eval0).asDiagonal() * evec0.adjoint()).real();
      } else if(sign == 1){
        DMatrixMap(G.matptr(m), nao_, nao_) =  (evec0 * bose_exp(beta_,tau,eval0).asDiagonal() * evec0.adjoint()).real();
      }
    }
  }

  // Ut
  ZMatrixMap IHdt = ZMatrixMap(X_.data(), nao_, nao_);
  ZMatrixMap Udt = ZMatrixMap(M_.data(), nao_, nao_);
  IHdt = std::complex<double>(0,1.0) * h * Hmu;
  Hmu = IHdt.exp();
  Udt = Hmu;
  ZMatrixMap(Q_.data(), nao_, nao_) = ZMatrixMap(iden_.data(), nao_, nao_);
  for( int n = 1; n <= tmax; n++ ) {
    ZMatrixMap(Q_.data() + n*es_, nao_, nao_) = ZMatrixMap(Q_.data() + (n-1)*es_, nao_, nao_) * Udt;
  }

  // TV
  for(int m=0; m < G.r(); m++) {
    tau = dlr_.it0B(m);

    for(int n=0; n<=tmax; n++) {
      ZMatrixMap UtMap = ZMatrixMap(Q_.data() + n*es_, nao_, nao_);
      if(sign==-1){
        ZMatrixMap(G.tvptr(n,m), nao_, nao_) = std::complex<double>(0,1.0) *  UtMap * evec0 * fermi_exp(beta_,tau,eval0m).asDiagonal() * evec0.adjoint();
        ZMatrixMap(G.tvptr_trans(n,m), nao_, nao_) = ZMatrixMap(G.tvptr(n,m), nao_, nao_).transpose();
      } else if(sign==1){
        ZMatrixMap(G.tvptr(n,m), nao_, nao_) = std::complex<double>(0,-1.0) * UtMap * evec0 * bose_exp(beta_,tau,eval0m).asDiagonal() * evec0.adjoint();
        ZMatrixMap(G.tvptr_trans(n,m), nao_, nao_) = ZMatrixMap(G.tvptr(n,m), nao_, nao_).transpose();
      }
    }
  }

  // Ret and Less
  ZMatrixMap value = ZMatrixMap(X_.data(), nao_, nao_);
  if(sign == -1){
    value = evec0 * fermi(beta_,eval0m).asDiagonal() * evec0.adjoint();
  }else if(sign == 1){
    value = -1.0 * evec0 * bose(beta_,eval0m).asDiagonal() * evec0.adjoint();
  }

  for(int m = 0; m <= tmax; m++) {
    for(int n = 0; n <= m; n++) {
      ZMatrixMap Ut1 = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
      ZMatrixMap Ut2 = ZMatrixMap(Q_.data() + n*es_, nao_, nao_);
      ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) = std::complex<double>(0,-1.0) * Ut1 * Ut2.adjoint();
      ZMatrixMap(G.curr_timestep_les_ptr(n,m), nao_, nao_) = std::complex<double>(0,1.0) * Ut2 * value * Ut1.adjoint();
    }
  }
}

void dyson::green_from_H_dm(herm_matrix_hodlr &G, double mu, ZMatrix H, ZMatrix &rho, double h, int tmax) {
  call_scope cs(*this);
  const char *where = "dyson::green_from_H_dm";
  if(cs.outer()) {
    check_free_args_(where, G, mu, H, h, tmax);
    HNESSI_REQUIRE(rho.rows() == nao_ && rho.cols() == nao_, where, HNESSI_KV(rho.rows()), HNESSI_KV(rho.cols()),
                   HNESSI_KV(nao_));
    hermitize_(H.data() + es_, -1, -1, where);   // D-11 on the local copy (H is passed by value), recorded as h(-1)
    // DY-F7 (tier 2): rho is a density matrix (D-1: rho = -G^M(beta^-) = xi i G^<(0,0)): Hermitian, and its
    // eigenvalues are occupations (fermions: in [0,1], bosons: >= 0); tolerance herm_err_tol_ (relative for the
    // Hermiticity, absolute for the occupations)
    double rn = rho.norm(), ra = (rho - rho.adjoint()).norm();
    double rrel = rn == 0 ? 0. : ra / rn;
    HNESSI_REQUIRE(rrel <= herm_err_tol_, where, HNESSI_KV(rrel), HNESSI_KV(herm_err_tol_),
                   detail::note{"rho must be the (Hermitian) density matrix: rel = ||rho-rho^dagger||_F/||rho||_F"});
    Eigen::SelfAdjointEigenSolver<ZMatrix> rs(0.5 * (rho + rho.adjoint()), Eigen::EigenvaluesOnly);
    double occ_min = rs.eigenvalues().minCoeff(), occ_max = rs.eigenvalues().maxCoeff();
    HNESSI_REQUIRE(occ_min >= -herm_err_tol_ && (xi_ == 1 || occ_max <= 1 + herm_err_tol_), where, HNESSI_KV(occ_min),
                   HNESSI_KV(occ_max), HNESSI_KV(xi_),
                   detail::note{"the eigenvalues of rho are occupations (fermions: 0..1, bosons: >= 0); rho is the "
                                "density matrix -G^M(beta^-), not G^M(0^+)"});
  }

  int sign = G.sig();
  double tau, t;

  // Make Hamiltonian and solve eigen problem
  ZMatrix Hmu = mu * ZMatrixMap(iden_.data(), nao_, nao_) - H;
  Eigen::SelfAdjointEigenSolver<ZMatrix> eigensolver(Hmu);

  ZMatrix evec0(nao_, nao_);
  DColVector eval0(nao_), eval0m(nao_);

  evec0 = eigensolver.eigenvectors();
  eval0 = eigensolver.eigenvalues();
  eval0m = -eval0;
  require_bosonic_spectrum_(eval0, "dyson::green_from_H_dm");   // DY-F10

  // Ut
  ZMatrixMap IHdt = ZMatrixMap(X_.data(), nao_, nao_);
  ZMatrixMap Udt = ZMatrixMap(M_.data(), nao_, nao_);
  IHdt = std::complex<double>(0,1.0) * h * Hmu;
  Udt = IHdt.exp();
  ZMatrixMap(Q_.data(), nao_, nao_) = ZMatrixMap(iden_.data(), nao_, nao_);
  for( int n = 1; n <= tmax; n++ ) {
    ZMatrixMap(Q_.data() + n*es_, nao_, nao_) = ZMatrixMap(Q_.data() + (n-1)*es_, nao_, nao_) * Udt;
  }

  // TV
  for(int m=0; m < G.r(); m++) {
    tau = dlr_.it0B(m);

    for(int n=0; n<=tmax; n++) {
      ZMatrixMap UtMap = ZMatrixMap(Q_.data() + n*es_, nao_, nao_);
      if(sign==-1){
        ZMatrixMap(G.tvptr(n,m), nao_, nao_) = std::complex<double>(0,1.0) *  UtMap * evec0 * fermi_exp(beta_,tau,eval0m).asDiagonal() * evec0.adjoint();
        ZMatrixMap(G.tvptr_trans(n,m), nao_, nao_) = ZMatrixMap(G.tvptr(n,m), nao_, nao_).transpose();
      } else if(sign==1){
        ZMatrixMap(G.tvptr(n,m), nao_, nao_) = std::complex<double>(0,-1.0) * UtMap * evec0 * bose_exp(beta_,tau,eval0m).asDiagonal() * evec0.adjoint();
        ZMatrixMap(G.tvptr_trans(n,m), nao_, nao_) = ZMatrixMap(G.tvptr(n,m), nao_, nao_).transpose();
      }
    }
  }

  // Ret and Less
  // rho is the density matrix rho = -G^M(beta^-) = xi i G^<(0,0), so G^<(t_n,t_m) = -i xi U(t_n) rho U(t_m)^\dagger
  for(int m = 0; m <= tmax; m++) {
    for(int n = 0; n <= m; n++) {
      ZMatrixMap Ut1 = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
      ZMatrixMap Ut2 = ZMatrixMap(Q_.data() + n*es_, nao_, nao_);
      ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) = std::complex<double>(0,-1.0) * Ut1 * Ut2.adjoint();
      ZMatrixMap(G.curr_timestep_les_ptr(n,m), nao_, nao_) = std::complex<double>(0,-double(sign)) * Ut2 * rho * Ut1.adjoint();
    }
  }
}

void dyson::extrapolate_2leg(herm_matrix_hodlr &G, Integration::Integrator &I) {
  call_scope cs(*this);   // ST-10
  int tstp = G.tstpmk() + G.k();
  int l, j, jcut;

  if(G.can_extrap()) {
    // DY-X1, DY-X2 (tier 2)
    HNESSI_REQUIRE(G.k() == k_ && I.k() == k_ && G.size1() == nao_ && G.size2() == nao_ && G.r() == r_ &&
                   G.nt() <= nt_, "dyson::extrapolate_2leg", HNESSI_KV(G.k()), HNESSI_KV(I.k()), HNESSI_KV(k_),
                   HNESSI_KV(G.size1()), HNESSI_KV(nao_), HNESSI_KV(G.r()), HNESSI_KV(r_), HNESSI_KV(G.nt()),
                   HNESSI_KV(nt_));
    HNESSI_REQUIRE_STATE(k_+1 <= tstp && tstp < G.nt(), "dyson::extrapolate_2leg", HNESSI_KV(tstp), HNESSI_KV(k_),
                         HNESSI_KV(G.nt()), detail::note{"there is no row tstp = G.tstpmk()+k to extrapolate to"});
    //retarded
    memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
    for(l=0; l<tstp-k_; l++) {
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + (tstp-l)*nao_*nao_, nao_, nao_);
      for(j=0; j<=k_; j++) {
        int t = tstp-j-1 == tstp-k_-1 ? tstp : tstp-j-1;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,tstp-l-j-1), nao_, nao_);
      }
    }
    for(l=0; l<=k_; l++) {
      jcut = (l<=tstp-k_-1) ? k_ : (tstp-l-1);
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*nao_*nao_, nao_, nao_);
  
      for(j=0; j<=jcut; j++) {
        int t = tstp-j-1 == tstp-k_-1 ? tstp : tstp-j-1;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,l), nao_, nao_);
      }
  
      for(j=jcut+1; j<=k_; j++) {
        int t = l == tstp-k_-1 ? tstp : l;
        resMap.noalias() -= I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,tstp-j-1), nao_, nao_).adjoint();
      }
    }
    memcpy(G.curr_timestep_ret_ptr(tstp, 0), Q_.data(), (tstp+1)*es_*sizeof(cplx));


    //less
    memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
    for(l=0; l<=k_; l++) {
      jcut=std::min(k_, tstp-l-1);
  
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*es_, nao_, nao_);
  
      for(j=0; j<=jcut; j++) {
        int tp = tstp-1-j == tstp-k_-1 ? tstp : tstp-1-j;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(l,tp), nao_, nao_);
      }
      for(j=jcut+1; j<=k_; j++) {
        resMap.noalias() -= I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-1-j,l), nao_, nao_).adjoint();
      }
    }
    for(l=k_+1; l<=tstp; l++) {
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*es_, nao_, nao_);
  
      for(j=0; j<=k_; j++) {
        int tp = tstp-1-j == tstp-k_-1 ? tstp : tstp-1-j;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(l-j-1, tp), nao_, nao_);
      }
    }
    memcpy(G.curr_timestep_les_ptr(0, tstp), Q_.data(), (tstp+1)*es_*sizeof(cplx));
  }

  G.can_extrap() = false;
}

void dyson::extrapolate(herm_matrix_hodlr &G, Integration::Integrator &I) {
  call_scope cs(*this);   // ST-10
  int tstp = G.tstpmk() + G.k();
  int l, j, jcut;

  if(G.can_extrap()) {
    // DY-X1, DY-X2 (tier 2)
    HNESSI_REQUIRE(G.k() == k_ && I.k() == k_ && G.size1() == nao_ && G.size2() == nao_ && G.r() == r_ &&
                   G.nt() <= nt_, "dyson::extrapolate", HNESSI_KV(G.k()), HNESSI_KV(I.k()), HNESSI_KV(k_),
                   HNESSI_KV(G.size1()), HNESSI_KV(nao_), HNESSI_KV(G.r()), HNESSI_KV(r_), HNESSI_KV(G.nt()),
                   HNESSI_KV(nt_));
    HNESSI_REQUIRE_STATE(k_+1 <= tstp && tstp < G.nt(), "dyson::extrapolate", HNESSI_KV(tstp), HNESSI_KV(k_),
                         HNESSI_KV(G.nt()), detail::note{"there is no row tstp = G.tstpmk()+k to extrapolate to"});
    //right mixing
    memset(G.tvptr(tstp,0), 0, es_*ntau_*sizeof(cplx));
    memset(G.tvptr_trans(tstp,0), 0, es_*ntau_*sizeof(cplx));
    for(l=0; l<ntau_; l++) {
      ZMatrixMap resMap = ZMatrixMap(G.tvptr(tstp,l), nao_, nao_);
      ZMatrixMap resMap_trans = ZMatrixMap(G.tvptr_trans(tstp,l), nao_, nao_);
      for(j=0; j<=k_; j++) {
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.tvptr(tstp-j-1,l), nao_, nao_);
        resMap_trans.noalias() += I.ex_weights(j) * ZMatrixMap(G.tvptr(tstp-j-1,l), nao_, nao_).transpose();
      }
    }
  
    //retarded
    memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
    for(l=0; l<tstp-k_; l++) {
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + (tstp-l)*nao_*nao_, nao_, nao_);
      for(j=0; j<=k_; j++) {
        int t = tstp-j-1 == tstp-k_-1 ? tstp : tstp-j-1;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,tstp-l-j-1), nao_, nao_);
      }
    }
    for(l=0; l<=k_; l++) {
      jcut = (l<=tstp-k_-1) ? k_ : (tstp-l-1);
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*nao_*nao_, nao_, nao_);
  
      for(j=0; j<=jcut; j++) {
        int t = tstp-j-1 == tstp-k_-1 ? tstp : tstp-j-1;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,l), nao_, nao_);
      }
  
      for(j=jcut+1; j<=k_; j++) {
        int t = l == tstp-k_-1 ? tstp : l;
        resMap.noalias() -= I.ex_weights(j) * ZMatrixMap(G.curr_timestep_ret_ptr(t,tstp-j-1), nao_, nao_).adjoint();
      }
    }
    memcpy(G.curr_timestep_ret_ptr(tstp, 0), Q_.data(), (tstp+1)*es_*sizeof(cplx));
  
    //less
    memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
    G.get_tv_tau(tstp, 0, dlr_, Q_.data()+es_);
    ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -ZMatrixMap(Q_.data()+es_, nao_, nao_).adjoint();
    memset(Q_.data()+es_, 0, es_*sizeof(cplx));
  
    for(l=1; l<=k_; l++) {
      jcut=std::min(k_, tstp-l-1);
  
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*es_, nao_, nao_);
  
      for(j=0; j<=jcut; j++) {
        int tp = tstp-1-j == tstp-k_-1 ? tstp : tstp-1-j;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(l,tp), nao_, nao_);
      }
      for(j=jcut+1; j<=k_; j++) {
        resMap.noalias() -= I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-1-j,l), nao_, nao_).adjoint();
      }
    }
    for(l=k_+1; l<=tstp; l++) {
      ZMatrixMap resMap = ZMatrixMap(Q_.data() + l*es_, nao_, nao_);
  
      for(j=0; j<=k_; j++) {
        int tp = tstp-1-j == tstp-k_-1 ? tstp : tstp-1-j;
        resMap.noalias() += I.ex_weights(j) * ZMatrixMap(G.curr_timestep_les_ptr(l-j-1, tp), nao_, nao_);
      }
    }
    memcpy(G.curr_timestep_les_ptr(0, tstp), Q_.data(), (tstp+1)*es_*sizeof(cplx));
  }

  G.can_extrap() = false;
}
} // namespace
