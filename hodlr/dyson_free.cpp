#include "dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace hodlr {

#define EXPMAX 100

// 1/(1+exp(omega*beta))
double fermi(double beta, double omega) {
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
  double tau;
  // Make Hamiltonian and solve eigen problem
  ZMatrix Hmu = mu * ZMatrixMap(iden_.data(), nao_, nao_) - H;
  Eigen::SelfAdjointEigenSolver<ZMatrix> eigensolver(Hmu);
  ZMatrix evec0(nao_, nao_);
  DColVector eval0(nao_), eval0m(nao_);
  evec0 = eigensolver.eigenvectors();
  eval0 = eigensolver.eigenvalues();
  eval0m = -eval0;

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


void dyson::green_from_H(herm_matrix_hodlr &G, double mu, function &H, double h, int tmax, bool inc_mat) {
  green_from_H( G, mu, H.get_map(-1), h, tmax, inc_mat);
}

  // Gives free green's function from constant hamiltonian
// G^M(\tau) = s f_s(mu-h) exp((mu-h)\tau)
// G^{TV}(n,\tau) = -is U_{n,0} f_s(h-mu) exp((h-mu)\tau)
// G^R(n,j) = -iU_{n,j} = U_{n,0} (U_{j,0})^\dagger
// G^L(j,n) = -siU_{j,0} f_s(h-mu) (U_{n,0})^\dagger
void dyson::green_from_H(herm_matrix_hodlr &G, double mu, ZMatrix H, double h, int tmax, bool inc_mat) {
  if( G.sig() != xi_ ) {
    throw std::invalid_argument("Fermion/Boson xi does not agree");
  }
  if( G.r() != r_ ) {
    std::cout<<G.r() << " "<< r_<<std::endl;
    throw std::invalid_argument("Ntau of G does not agree with r of libdlr");
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
  if( G.sig() != xi_ ) {
    throw std::invalid_argument("Fermion/Boson xi does not agree");
  }
  if( G.r() != r_ ) {
    std::cout<<G.r() << " "<< r_<<std::endl;
    throw std::invalid_argument("Ntau of G does not agree with r of libdlr");
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
  for(int m = 0; m <= tmax; m++) {
    for(int n = 0; n <= m; n++) {
      ZMatrixMap Ut1 = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
      ZMatrixMap Ut2 = ZMatrixMap(Q_.data() + n*es_, nao_, nao_);
      ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) = std::complex<double>(0,-1.0) * Ut1 * Ut2.adjoint();
      ZMatrixMap(G.curr_timestep_les_ptr(n,m), nao_, nao_) = std::complex<double>(0,1.0) * Ut2 * rho * Ut1.adjoint();
    }
  }
}

void dyson::extrapolate_2leg(herm_matrix_hodlr &G, Integration::Integrator &I) {
  int tstp = G.tstpmk() + G.k();
  int l, j, jcut;

  if(G.can_extrap()) {
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
  int tstp = G.tstpmk() + G.k();
  int l, j, jcut;

  if(G.can_extrap()) {
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
