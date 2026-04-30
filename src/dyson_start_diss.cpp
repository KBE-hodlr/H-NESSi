#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

double dyson::dyson_start_tv_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  // We also enforce TTI for 0 <= t <= k
  // This is guaranteed using checks in the dyson_start() routine as well
  int m, l, n, i;
  double err = 0;

  int es = G.size1() * G.size2();
  int r = G.r();

  for(int t = 0; t <= k_; t++) {
    for(int i = 0; i < r_; i++) {
      ZMatrixMap(Sigma.tvptr_trans(t,i), nao_, nao_) = ZMatrixMap(Sigma.tvptr(t,i), nao_,nao_).transpose();
    }
  }

  cplx cplxi = cplx(0.,1.);
  ZMatrixMap QMap = ZMatrixMap(Q_.data(), nao_*k_, nao_);
  ZMatrixMap MMap = ZMatrixMap(M_.data(), nao_*k_, nao_*k_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  // Boundary Conditions
  ZMatrixMap(NTauTmp_.data(), r_, es_) = DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * DMatrixMap(G.matptr(0), r_, es_);
  for(m=0; m<r_; m++) {
    auto tvmap = ZMatrixMap(G.tvptr(0,m), nao_, nao_);
    auto tvmap_trans = ZMatrixMap(G.tvptr_trans(0,m), nao_, nao_);

    auto matmap = ZMatrixMap(NTauTmp_.data()+m*es_, nao_, nao_);
    err += (tvmap - (double)G.sig()*cplxi*matmap).norm();

    tvmap.noalias() = (double)G.sig()*cplxi*matmap;
    tvmap_trans.noalias() = (double)G.sig()*cplxi*matmap.transpose();
  }

  // At each m, get n=1...k
  for(m=0; m<r_; m++) {
    memset(M_.data(),0,k_*k_*es_*sizeof(cplx));
    memset(Q_.data(),0,k_*es_*sizeof(cplx));


    // Set up the kxk linear problem MX=Q
    for(n=1; n<=k_; n++) {
      // do the integral
      tv_it_conv(m, n, Sigma, G, Q_.data()+(n-1)*es_);

      auto QMapBlock = ZMatrixMap(Q_.data() + (n-1)*es_, nao_, nao_);

      for(l=0; l<=k_; l++) {
        // This is not the best practice, but we dont use this for l=0 and it compains about indexing -1 for l=0
        auto MMapBlock = MMap.block((n-1)*nao_, ((l==0?1:l)-1)*nao_, nao_, nao_);

        // Derivative term
        if(l == 0){ // Put into Q
          QMapBlock.noalias() -= cplxi*I.poly_diff(n,l)/h * ZMatrixMap(G.tvptr(0,m), nao_, nao_);
        }
        else{ // Put into M
          MMapBlock.noalias() += cplxi*I.poly_diff(n,l)/h * IMap;
        }

        // Delta energy term
        // h_o = h - i(\ell^> -\xi \ell^<)
        if(l==n){
          MMapBlock.noalias() += mu*IMap - ZMatrixMap(H + l*es_, nao_, nao_) + cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_));
        }

        // Integral term
        if(l==0){ // Put into Q
          QMapBlock.noalias() += h*I.gregory_weights(n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_) * ZMatrixMap(G.tvptr(l,m), nao_, nao_);
        }
        else{ // Put into M
          if(n>=l){ // Have Sig
            MMapBlock.noalias() -= h*I.gregory_weights(n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_);
          }
          else{ // Dont have Sig
            MMapBlock.noalias() += h*I.gregory_weights(n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_).adjoint();
          }
        }
      }
    }

    // Solve MX=Q
    Eigen::FullPivLU<ZMatrix> lu(MMap);
    ZMatrixMap(X_.data(), k_*nao_, nao_) = lu.solve(QMap);

    for(l=0; l<k_; l++){
      err += (ZColVectorMap(G.tvptr(l+1,m), es_) - ZColVectorMap(X_.data() + l*es_, es_)).norm();
      ZMatrixMap(G.tvptr(l+1,m), nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_);
      ZMatrixMap(G.tvptr_trans(l+1,m), nao_, nao_).noalias() = ZMatrixMap(G.tvptr(l+1,m), nao_, nao_).transpose();
    }
  }

  return err;
}

double dyson::dyson_start_ret_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  // We enforce TTI for 0 <= t <= k
  // This is guaranteed using checks in the dyson_start() routine

  double err = 0;
  cplx ncplxi = cplx(0,-1);

  for(int n = 0; n <= k_; n++) {
    G.get_tv_tau(n, 0, dlr_, M_.data());
    G.get_tv_tau(n, beta_, dlr_, Q_.data());
    ZMatrixMap(M_.data(), nao_, nao_) *= -1;
    ZMatrixMap(M_.data(), nao_, nao_) += G.sig() * ZMatrixMap(Q_.data(), nao_, nao_);

    if(n == 0) ZMatrixMap(M_.data(), nao_, nao_) = ncplxi * ZMatrixMap(iden_.data(), nao_, nao_);

    for(int l = 0; l <= k_-n; l++) {
      ZMatrixMap retMap = ZMatrixMap(G.curr_timestep_ret_ptr(n+l,l), nao_, nao_);
      err += (ZMatrixMap(M_.data(), nao_, nao_) - retMap).lpNorm<2>();
      retMap = ZMatrixMap(M_.data(), nao_, nao_);
    }
  }
  
 
  return err;
}

double dyson::dyson_start_ret_ntti_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  // Counters
  int m, l, n, i;

  double err = 0;

  cplx ncplxi = cplx(0, -1);
  ZMatrixMap QMap = ZMatrixMap(Q_.data(), nao_*k_, nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  // Keep G^R(:,0) for error later, used only if(imp_tp0)
  ZMatrix GR0(k_*nao_*nao_, 1);
  for(i = 0; i < k_; i++) {
    ZMatrixMap(GR0.data() + i*nao_*nao_, nao_, nao_) = ZMatrixMap(G.curr_timestep_ret_ptr(i+1, 0), nao_, nao_);
  }

  // Initial condition
  for(i=0; i<=k_; i++){
    ZMatrixMap(G.curr_timestep_ret_ptr(i,i), nao_, nao_).noalias() = ncplxi*IMap;
  }

  // Fill GR(:k+1,0)
  memset(M_.data(), 0, k_*k_*es_*sizeof(cplx));
  memset(Q_.data(), 0, k_*es_*sizeof(cplx));

  for(m=0; m<k_; m++) {
    ZMatrixMap MMap = ZMatrixMap(M_.data(), nao_*(k_-m), nao_*(k_-m));
    memset(M_.data(), 0, k_*k_*es_*sizeof(cplx));
    memset(Q_.data(), 0, k_*es_*sizeof(cplx));

    for(n=m+1; n<=k_; n++) {
      auto QMapBlock = QMap.block((n-m-1)*nao_, 0, nao_, nao_);

      for(l=0; l<=m; l++) {
        QMapBlock.noalias() -= -ncplxi/h * I.poly_diff(n,l) * -1.*ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_).adjoint() - h * I.poly_integ(m,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_) * -1. * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_).adjoint();
      }

      for(l = m+1; l <= k_; l++) {
        MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) += -ncplxi/h * I.poly_diff(n,l) * IMap;
        // h_o = h - i(\ell^> -\xi \ell^<)
        if(n==l) MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) += mu*IMap - ZMatrixMap(H + l*es_, nao_, nao_) - ncplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_));
        if(n>=l) MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) += -h*I.poly_integ(m,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_);
        else     MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) -= -h*I.poly_integ(m,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint();
      }

    }

    // Solve MX=Q for X
    Eigen::FullPivLU<ZMatrix> lu(ZMatrixMap(M_.data(), (k_-m)*nao_, (k_-m)*nao_));
    ZMatrixMap(X_.data(), (k_-m)*nao_, nao_) = lu.solve(ZMatrixMap(Q_.data(), (k_-m)*nao_, nao_));

    // Put X into G
    for(n=m+1; n<=k_; n++){
      if(!imp_tp0 or m!=0) err += (ZColVectorMap(G.curr_timestep_ret_ptr(n,m), es_) - ZColVectorMap(X_.data() + (n-m-1)*es_, es_)).norm();
      ZMatrixMap(G.curr_timestep_ret_ptr(n,m), nao_, nao_).noalias() = ZMatrixMap(X_.data() + (n-m-1)*es_, nao_, nao_);
    }
  }

  // We solve the t' equation downwards at all k slices to get G^R(t,0)
  if(imp_tp0) {
    for(m = 1; m <= k_; m++) {
      ZMatrixMap MMap0 = ZMatrixMap(M_.data(), nao_, nao_);
      ZMatrixMap QMap0 = ZMatrixMap(Q_.data(), nao_, nao_);
      // h_o = h - i(\ell^> -\xi \ell^<)
      MMap0 = -ncplxi/h * I.poly_diff(k_,k_) * IMap - ZMatrixMap(H, nao_, nao_) - ncplxi * (ZMatrixMap(ellG, nao_, nao_) - G.sig() * ZMatrixMap(ellL, nao_, nao_)) + mu*IMap - h * I.poly_integ(k_-m,k_,k_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(0,0), nao_, nao_);
      QMap0.setZero();
      for(int i = 0; i < k_-m; i++)  QMap0 -= ncplxi/h * I.poly_diff(k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(k_-i,m), nao_, nao_).adjoint();
      for(int i = k_-m; i < k_; i++) QMap0 += ncplxi/h * I.poly_diff(k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(m,k_-i), nao_, nao_);
      for(int i = 0; i < k_-m; i++)  QMap0 -= h * I.poly_integ(k_-m,k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(k_-i,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(k_-i,0), nao_, nao_);
      for(int i = k_-m; i < k_; i++) QMap0 += h * I.poly_integ(k_-m,k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(m,k_-i), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(k_-i,0), nao_, nao_);

      // Solve MX=Q for X
      Eigen::FullPivLU<ZMatrix> lu(ZMatrixMap(M_.data(), nao_, nao_));
      ZMatrixMap(X_.data(), nao_, nao_) = lu.solve(ZMatrixMap(Q_.data(), nao_, nao_));
      err += (ZColVectorMap(GR0.data() + (m-1)*nao_*nao_, nao_, nao_) - ZColVectorMap(X_.data(), es_)).norm();
      ZMatrixMap(G.curr_timestep_ret_ptr(m,0), nao_, nao_).noalias() = ZMatrixMap(X_.data(), nao_, nao_);
    }
  }

  return err;
}


double dyson::dyson_start_les_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  // We enforce TTI for 0 <= t <= k
  // This is guaranteed using checks in the dyson_start() routine
  double err = 0;
  for(int tp = 0; tp <= k_; tp++) {
    G.get_tv_tau(tp, 0, dlr_, M_.data());
    for( int l = 0; l <= k_-tp; l++) {
      err += (ZMatrixMap(G.curr_timestep_les_ptr(l,l+tp), nao_, nao_) + ZMatrixMap(M_.data(), nao_, nao_).adjoint()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(l,l+tp), nao_, nao_).noalias() = -ZMatrixMap(M_.data(), nao_, nao_).adjoint();
    }
  }

  return err;
}


double dyson::dyson_start_les_ntti_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  double err = 0;
  for(int tp = 0; tp <= k_; tp++) {
    G.get_tv_tau(tp, 0, dlr_, M_.data());
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,tp), nao_, nao_) + ZMatrixMap(M_.data(), nao_, nao_).adjoint()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,tp), nao_, nao_).noalias() = -ZMatrixMap(M_.data(), nao_, nao_).adjoint();
  }

  cplx cplxi = cplx(0,1);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  // store diagonal in case we do the diagonal correction.
  // needed for evaluating iteration error
  ZMatrix DIC = ZMatrix::Zero(k_*nao_, nao_);
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_) = ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_);
  }

  // Here we are using the t' equation and solving upwards from the IC at G^<(m,0)
  for(int m = 1; m <= k_; m++) {
    ZMatrix MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    ZMatrix XIC = ZMatrix::Zero(k_*nao_, nao_);
    ZMatrix QIC = ZMatrix::Zero(k_*nao_, nao_);

    for(int n = 1; n <= k_; n++) {
      auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);
      for(int l = 0; l <= k_; l++) {
        if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      }
      QBlock -= (h * I.poly_integ(0,n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,0), nao_, nao_).adjoint()).transpose();
      QBlock -= (cplxi/h * I.poly_diff(n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint()).transpose();
      // additional dissipative term
      // Using notation from Table 4.1 from thesis q_n = \xi 2 G^R_{mn}\ell^<_n
      // Q_n = -i [q_n - y_0 M_{0n}]
      if(m>=n) QBlock -= (2. * G.sig() * cplxi * ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) * ZMatrixMap(ellL + n*es_, nao_, nao_)).transpose();

      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
        // h_o = h - i(\ell^> -\xi \ell^<)
        // we use the adjoint equation, so h and i share same sign
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      if(rho_version_ != 0) err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      else                  err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
    }
  }

  if(rho_version_== 1) {

    // redo diagonal
    ZMatrix MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    ZMatrix XIC = ZMatrix::Zero(k_*nao_, nao_);
    ZMatrix QIC = ZMatrix::Zero(k_*nao_, nao_);


    for(int i = 1; i <= k_; i++) {
      for(int j = 1; j <= k_; j++) {
        auto MBlock = MIC.block((i-1)*nao_, (j-1)*nao_, nao_, nao_);
        MBlock = 1./h * I.poly_diff(i,j) * IMap;
      }
    }

    // Solving \partial_t G = -ih_oG -iG^\dagger h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = -ih_oG +iG h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-ih_oG-iI] -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [iG^\dagger h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-iG h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] - [iG h_o^\dagger - iI^dagger]^\dagger -2\xi i\ell^<
    // I         =  S^R G^< + S^< G^A
    // I^\dagger = -G^< S^A - G^R S^<
    for(int i = 1; i <= k_; i++) {
      auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint, so h and i share same sign
      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_) 
                                                                                + cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_) 
                                                                                           - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_)) 
                                                                                - mu*IMap);

      for(int l = 0; l <= i; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = 0; l <= i; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
      }
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(i, i, G, Sigma, M_.data());
      QBlock += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();

      ZMatrixMap(XIC.data(), nao_, nao_) = QBlock-QBlock.adjoint();
      QBlock = ZMatrixMap(XIC.data(), nao_, nao_);

      QBlock -= 1./h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
      // additional dissipative term
      QBlock -= 2. * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
    }

    Eigen::FullPivLU<ZMatrix> lu3(MIC);
    XIC = lu3.solve(QIC);
    for(int i = 1; i <= k_; i++) {
      err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
    }
  }

  if(rho_version_ == 2) {
    // Pre-calculate sizes
    int n2 = nao_ * nao_;
    int sys_size = k_ * n2;

    // Allocate the global system matrices
    ZMatrix L_global = ZMatrix::Zero(sys_size, sys_size);
    ZColVector RHS_global = ZColVector::Zero(sys_size); // USING YOUR ZColVector

    // Identities needed for Kronecker products
    ZMatrix I_nao = ZMatrix::Identity(nao_, nao_);
    ZMatrix I_n2 = ZMatrix::Identity(n2, n2);

    // 1. Build the Time Derivative Operator: D (x) I_n2
    ZMatrix D_mat = ZMatrix::Zero(k_, k_);
    for(int i = 1; i <= k_; i++) {
        for(int j = 1; j <= k_; j++) {
            D_mat(i-1, j-1) = 1.0 / h * I.poly_diff(i,j);
        }
    }
    L_global += Eigen::kroneckerProduct(D_mat, I_n2).eval();

    // 2. Build the Spatial Operators and RHS per timestep
    for(int i = 1; i <= k_; i++) {
        // Construct the effective non-Hermitian Hamiltonian for step i
        ZMatrix h_o_i = ZMatrixMap(H+i*es_, nao_, nao_) 
                      - cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_)) 
                      - mu * IMap;

        // Build the Base Liouvillian A_i (Row-Major)
        ZMatrix A_i = -cplxi * Eigen::kroneckerProduct(h_o_i, I_nao).eval() 
                  + cplxi * Eigen::kroneckerProduct(I_nao, h_o_i.conjugate()).eval();

        // Pull the l=i collision integral term (from loop 3) into the LHS
        double w_ii = I.poly_integ(0, i, i);
        std::complex<double> alpha = cplxi * w_ii * h;
        ZMatrix Sigma_R_ii = ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,i), nao_, nao_);

        A_i += -alpha * Eigen::kroneckerProduct(Sigma_R_ii, I_nao).eval() 
               + alpha * Eigen::kroneckerProduct(I_nao, Sigma_R_ii.conjugate()).eval();

        // Subtract total A_i from the diagonal block of L_global
        L_global.block((i-1)*n2, (i-1)*n2, n2, n2) -= A_i;

        // --- Compute RHS Constants for step i ---
        ZMatrix C_half = ZMatrix::Zero(nao_, nao_);
        
        // Loop 1: G^R * Sigma^< (Safe for l < i)
        for(int l = 0; l <= i; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
        }

        // Loop 2: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 3: l < i (l=i was successfully moved to LHS)
        for(int l = 0; l < i; l++) { 
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 4: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
        }

        ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
        les_it_int(i, i, G, Sigma, M_.data());
        C_half += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();
        
        // Apply the physical symmetry trick to generate the missing conjugate halves
        ZMatrix C_i = C_half - C_half.adjoint();

        // --- ADD FULLY FORMED TERMS ---
        // (These are added AFTER symmetrization so they are not artificially doubled)
        C_i -= 1.0 / h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
        C_i -= 2.0 * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
        
        // Flatten C_i into the global RHS column vector
        RHS_global.segment((i-1)*n2, n2) = Eigen::Map<ZColVector>(C_i.data(), n2);
    }
        
    // 3. Direct Exact Solve
    Eigen::FullPivLU<ZMatrix> lu(L_global);
    ZColVector G_flat = lu.solve(RHS_global);
    
    // 4. Map the flattened solution back into the central storage
    for(int i = 1; i <= k_; i++) {
        // Extract the N x N block. Because ZMatrix is RowMajor, Mapping the column vector 
        // segment back to ZMatrix correctly unpacks it row-by-row!
        ZMatrix G_solved = Eigen::Map<ZMatrix>(G_flat.data() + (i-1)*n2, nao_, nao_);
        
        // Strictly enforce anti-hermiticity one last time to kill floating point noise
        ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = 0.5 * (G_solved - G_solved.adjoint());
        err += (ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    }
  }

  return err;
}

double dyson::dyson_start_les_ntti_nobc_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  cplx cplxi = cplx(0,1);

  ZMatrix QIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix XIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  double err = 0;
  int m = 0;

  // FIRST COLUMN, T=0
  for(int n = 1; n <= k_; n++) {
    auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);

    for(int l = 0; l <= k_; l++) {
      if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
    }

    int l = 0;
    QBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
    QBlock += (cplxi/h * I.poly_diff(n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_)).transpose();
    // additional dissipative term would go here, but it is proportional to G^R(0,t')
    // if we interpret our integration as starting from t'=0^+, this evaluates to zero.

    ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
    les_it_int(m, n, G, Sigma, M_.data());
    QBlock += ZMatrixMap(M_.data(), nao_, nao_);
  }

  for(int n = 1; n <= k_; n++) {
    for(int l = 1; l <= k_; l++) {
      auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
      MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint equation, so h and i share same sign
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
      if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
      if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
    }
  }
  Eigen::FullPivLU<ZMatrix> lu(MIC);
  XIC = lu.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
  }

  // store diagonal in case we do the diagonal correction.
  // needed for evaluating iteration error
  ZMatrix DIC = ZMatrix::Zero(k_*nao_, nao_);
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_) = ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_);
  }

  // REST OF THE COLUMNS 
  for(m = 1; m <= k_; m++) {
    MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    XIC = ZMatrix::Zero(k_*nao_, nao_);
    QIC = ZMatrix::Zero(k_*nao_, nao_);

    for(int n = 1; n <= k_; n++) {
      auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);
      for(int l = 0; l <= k_; l++) {
        if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      }
      QBlock -= (h * I.poly_integ(0,n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,0), nao_, nao_).adjoint()).transpose();
      QBlock -= (cplxi/h * I.poly_diff(n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint()).transpose();
      // additional dissipative term
      // Using notation from Table 4.1 from thesis q_n = \xi 2 G^R_{mn}\ell^<_n
      // Q_n = -i [q_n - y_0 M_{0n}]
      if(m>=n) QBlock -= (2. * G.sig() * cplxi * ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) * ZMatrixMap(ellL + n*es_, nao_, nao_)).transpose();

      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }

    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
        // h_o = h - i(\ell^> -\xi \ell^<)
        // we use the adjoint equation, so h and i share same sign
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      if(rho_version_ != 0) err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      else                  err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
    }
  }

  if(rho_version_==1) {
    // redo diagonal
    MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    XIC = ZMatrix::Zero(k_*nao_, nao_);
    QIC = ZMatrix::Zero(k_*nao_, nao_);

    for(int i = 1; i <= k_; i++) {
      for(int j = 1; j <= k_; j++) {
        auto MBlock = MIC.block((i-1)*nao_, (j-1)*nao_, nao_, nao_);
        MBlock = 1./h * I.poly_diff(i,j) * IMap;
      }
    }
    // Solving \partial_t G = -ih_oG -iG^\dagger h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = -ih_oG +iG h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-ih_oG-iI] -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [iG^\dagger h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-iG h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] - [iG h_o^\dagger - iI^dagger]^\dagger -2\xi i\ell^<
    // I         =  S^R G^< + S^< G^A
    // I^\dagger = -G^< S^A - G^R S^<
    for(int i = 1; i <= k_; i++) {
      auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint, so h and i share same sign
      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_)
                                                                                + cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_)
                                                                                           - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_))
                                                                                - mu*IMap);
      for(int l = 0; l <= i; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = 0; l <= i; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
      }
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(i, i, G, Sigma, M_.data());
      QBlock += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();

      ZMatrixMap(XIC.data(), nao_, nao_) = QBlock-QBlock.adjoint();
      QBlock = ZMatrixMap(XIC.data(), nao_, nao_);

      QBlock -= 1./h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
      // additional dissipative term
      QBlock -= 2. * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
    }

    Eigen::FullPivLU<ZMatrix> lu3(MIC);
    XIC = lu3.solve(QIC);
    for(int i = 1; i <= k_; i++) {
      err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
    }
  }

  if(rho_version_ == 2) {
    // Pre-calculate sizes
    int n2 = nao_ * nao_;
    int sys_size = k_ * n2;

    // Allocate the global system matrices
    ZMatrix L_global = ZMatrix::Zero(sys_size, sys_size);
    ZColVector RHS_global = ZColVector::Zero(sys_size); // USING YOUR ZColVector

    // Identities needed for Kronecker products
    ZMatrix I_nao = ZMatrix::Identity(nao_, nao_);
    ZMatrix I_n2 = ZMatrix::Identity(n2, n2);

    // 1. Build the Time Derivative Operator: D (x) I_n2
    ZMatrix D_mat = ZMatrix::Zero(k_, k_);
    for(int i = 1; i <= k_; i++) {
        for(int j = 1; j <= k_; j++) {
            D_mat(i-1, j-1) = 1.0 / h * I.poly_diff(i,j);
        }
    }
    L_global += Eigen::kroneckerProduct(D_mat, I_n2).eval();

    // 2. Build the Spatial Operators and RHS per timestep
    for(int i = 1; i <= k_; i++) {
        // Construct the effective non-Hermitian Hamiltonian for step i
        ZMatrix h_o_i = ZMatrixMap(H+i*es_, nao_, nao_) 
                      - cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_)) 
                      - mu * IMap;

        // Build the Base Liouvillian A_i (Row-Major)
        ZMatrix A_i = -cplxi * Eigen::kroneckerProduct(h_o_i, I_nao).eval() 
                  + cplxi * Eigen::kroneckerProduct(I_nao, h_o_i.conjugate()).eval();

        // Pull the l=i collision integral term (from loop 3) into the LHS
        double w_ii = I.poly_integ(0, i, i);
        std::complex<double> alpha = cplxi * w_ii * h;
        ZMatrix Sigma_R_ii = ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,i), nao_, nao_);

        A_i += -alpha * Eigen::kroneckerProduct(Sigma_R_ii, I_nao).eval() 
               + alpha * Eigen::kroneckerProduct(I_nao, Sigma_R_ii.conjugate()).eval();

        // Subtract total A_i from the diagonal block of L_global
        L_global.block((i-1)*n2, (i-1)*n2, n2, n2) -= A_i;

        // --- Compute RHS Constants for step i ---
        ZMatrix C_half = ZMatrix::Zero(nao_, nao_);
        
        // Loop 1: G^R * Sigma^< (Safe for l < i)
        for(int l = 0; l <= i; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
        }

        // Loop 2: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 3: l < i (l=i was successfully moved to LHS)
        for(int l = 0; l < i; l++) { 
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 4: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
        }

        ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
        les_it_int(i, i, G, Sigma, M_.data());
        C_half += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();
        
        // Apply the physical symmetry trick to generate the missing conjugate halves
        ZMatrix C_i = C_half - C_half.adjoint();

        // --- ADD FULLY FORMED TERMS ---
        // (These are added AFTER symmetrization so they are not artificially doubled)
        C_i -= 1.0 / h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
        C_i -= 2.0 * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
        
        // Flatten C_i into the global RHS column vector
        RHS_global.segment((i-1)*n2, n2) = Eigen::Map<ZColVector>(C_i.data(), n2);
    }
        
    // 3. Direct Exact Solve
    Eigen::FullPivLU<ZMatrix> lu(L_global);
    ZColVector G_flat = lu.solve(RHS_global);
    
    // 4. Map the flattened solution back into the central storage
    for(int i = 1; i <= k_; i++) {
        // Extract the N x N block. Because ZMatrix is RowMajor, Mapping the column vector 
        // segment back to ZMatrix correctly unpacks it row-by-row!
        ZMatrix G_solved = Eigen::Map<ZMatrix>(G_flat.data() + (i-1)*n2, nao_, nao_);
        
        // Strictly enforce anti-hermiticity one last time to kill floating point noise
        ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = 0.5 * (G_solved - G_solved.adjoint());
        err += (ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    }
  }

  return err;
}

double dyson::dyson_start_les_2leg_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  cplx cplxi = cplx(0,1);

  ZMatrix QIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix XIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  double err = 0;
  int m = 0;

  // FIRST COLUMN, T=0
  for(int n = 1; n <= k_; n++) {
    auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);

    for(int l = 0; l <= k_; l++) {
      if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
    }

    int l = 0;
    QBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
    QBlock += (cplxi/h * I.poly_diff(n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_)).transpose();
    // additional dissipative term would go here, but it is proportional to G^R(0,t')
    // if we interpret our integration as starting from t'=0^+, this evaluates to zero.

    ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//    les_it_int(m, n, G, Sigma, M_.data());
    QBlock += ZMatrixMap(M_.data(), nao_, nao_);
  }

  for(int n = 1; n <= k_; n++) {
    for(int l = 1; l <= k_; l++) {
      auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
      MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint equation, so h and i share same sign
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
      if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
      if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
    }
  }
  Eigen::FullPivLU<ZMatrix> lu(MIC);
  XIC = lu.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
  }

  // store diagonal in case we do the diagonal correction.
  // needed for evaluating iteration error
  ZMatrix DIC = ZMatrix::Zero(k_*nao_, nao_);
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_) = ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_);
  }
  // REST OF THE COLUMNS 
  for(m = 1; m <= k_; m++) {
    MIC = ZMatrix::Zero((k_-m+1)*nao_, (k_-1+m)*nao_);
    XIC = ZMatrix::Zero((k_-m+1)*nao_, nao_);
    QIC = ZMatrix::Zero((k_-m+1)*nao_, nao_);

    for(int n = m; n <= k_; n++) {
      auto QBlock = QIC.block((n-m)*nao_, 0, nao_, nao_);
      for(int l = 0; l <= k_; l++) {
        if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      }
      for(int l = 0; l < m; l++) {
        if(n>=l) QBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(G.curr_timestep_les_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) QBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(G.curr_timestep_les_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
      for(int l = 0; l < m; l++) {
        QBlock -= (cplxi/h * I.poly_diff(n,l) * ZMatrixMap(G.curr_timestep_les_ptr(l,m), nao_, nao_).adjoint()).transpose();
      }
      // additional dissipative term
      // Using notation from Table 4.1 from thesis q_n = \xi 2 G^R_{mn}\ell^<_n
      // Q_n = -i [q_n - y_0 M_{0n}]
      // comment out since we are solving for n>=m so G^R_{mn}=0
      if(n==m) QBlock -= (2. * G.sig() * cplxi * ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) * ZMatrixMap(ellL + n*es_, nao_, nao_)).transpose();

      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }

    for(int n = m; n <= k_; n++) {
      for(int l = m; l <= k_; l++) {
        auto MBlock = MIC.block((n-m)*nao_, (l-m)*nao_, nao_, nao_);
        MBlock -= cplxi/h * I.poly_diff(n,l) * IMap;
        // h_o = h - i(\ell^> -\xi \ell^<)
        // we use the adjoint equation, so h and i share same sign
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }

    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      if(rho_version_ != 0) err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-m)*es_, nao_, nao_).transpose()).norm();
      else                  err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-m)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-m)*es_, nao_, nao_).transpose();
    }
  }

  if(rho_version_==1) {
    // redo diagonal
    MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    XIC = ZMatrix::Zero(k_*nao_, nao_);
    QIC = ZMatrix::Zero(k_*nao_, nao_);


    for(int i = 1; i <= k_; i++) {
      for(int j = 1; j <= k_; j++) {
        auto MBlock = MIC.block((i-1)*nao_, (j-1)*nao_, nao_, nao_);
        MBlock = 1./h * I.poly_diff(i,j) * IMap;
      }
    }

    // Solving \partial_t G = -ih_oG -iG^\dagger h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = -ih_oG +iG h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-ih_oG-iI] -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [iG^\dagger h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-iG h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] - [iG h_o^\dagger - iI^dagger]^\dagger -2\xi i\ell^<
    // I         =  S^R G^< + S^< G^A
    // I^\dagger = -G^< S^A - G^R S^<
    for(int i = 1; i <= k_; i++) {
      auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint, so h and i share same sign
      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_)
                                                                                + cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_)
                                                                                           - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_))
                                                                                - mu*IMap);
      for(int l = 0; l <= i; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = 0; l <= i; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
      }
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
  //    les_it_int(i, i, G, Sigma, M_.data());
      QBlock += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();

      ZMatrixMap(XIC.data(), nao_, nao_) = QBlock-QBlock.adjoint();
      QBlock = ZMatrixMap(XIC.data(), nao_, nao_);

      QBlock -= 1./h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
      // additional dissipative term
      QBlock -= 2. * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
    }

    Eigen::FullPivLU<ZMatrix> lu3(MIC);
    XIC = lu3.solve(QIC);
    for(int i = 1; i <= k_; i++) {
      err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
    }
  }

  // ========================================================================================
  // EXACT DIRECT SOLVER FORMALISM (LIOUVILLE SPACE)
  // ========================================================================================
  // Original Equation of Motion:
  // \partial_t G^< = [iG^< h_o^\dagger - iI^\dagger] - [iG^< h_o^\dagger - iI^\dagger]^\dagger - 2\xi i\ell^<
  // 
  // Where the collision integral is:
  // I         = S^R G^< + S^< G^A
  // I^\dagger = -G^< S^A - G^R S^<
  //
  // Let C_half = iG^< h_o^\dagger - iI^\dagger 
  //            = iG^< h_o^\dagger + iG^< S^A + iG^R S^<
  // By computing C_half and taking (C_half - C_half^\dagger), we perfectly generate 
  // both the Left and Right acting Hamiltonian and Collision terms while strictly 
  // enforcing anti-Hermiticity (G^< = -G^<^\dagger).
  //
  // ----------------------------------------------------------------------------------------
  // VECTORIZATION IDENTITY (ROW-MAJOR)
  // ----------------------------------------------------------------------------------------
  // To solve this implicitly without a fixed-point loop, we map the N x N matrices into 
  // an N^2 x 1 column vector. Because Eigen maps RowMajor matrices by stacking rows, we 
  // MUST use the Row-Major Kronecker identity:
  //      vec_R(A * X * B) = (A (x) B^T) * vec_R(X)
  // Note: Since (B^\dagger)^T = B^*, right-multiplying by B^\dagger maps to (I (x) B^*).
  //
  // ----------------------------------------------------------------------------------------
  // 1. THE SPATIAL HAMILTONIAN (LHS)
  // ----------------------------------------------------------------------------------------
  // The commutator generated by the symmetry trick is: -ih_o G^< + iG^< h_o^\dagger
  // Applying the row-major identity:
  //      vec_R(-ih_o G^< I)  = -i(h_o (x) I) * vec_R(G^<)
  //      vec_R(i I G^< h_o^\dagger) =  i(I (x) h_o^*) * vec_R(G^<)
  // 
  // Base Liouvillian: A_i = -i(h_o (x) I) + i(I (x) h_o^*)
  //
  // ----------------------------------------------------------------------------------------
  // 2. EXTRACTING THE IMPLICIT COLLISION TERM (LHS)
  // ----------------------------------------------------------------------------------------
  // The l=i term of the collision integral contains the unknown G^<(t_i, t_i). 
  // Inside C_half, this term is: Q = i * w_ii * h * G^< S^A
  // Let \alpha = i * w_ii * h. 
  // When the symmetry trick (Q - Q^\dagger) is applied, it generates:
  //      Q - Q^\dagger = \alpha G^< S^A - (\alpha G^< S^A)^\dagger
  //                    = \alpha G^< S^A - \alpha^* (S^A)^\dagger (G^<)^\dagger
  // Because \alpha is purely imaginary (\alpha^* = -\alpha) and G^< is anti-Hermitian:
  //                    = \alpha G^< S^A - (-\alpha) S^R (-G^<)
  //                    = \alpha G^< S^A - \alpha S^R G^<
  //
  // To make the solver fully implicit, we move this to the LHS block matrix A_i.
  // Applying the row-major vectorization identity to (-\alpha S^R G^< + \alpha G^< S^A):
  //      vec_R(-\alpha S^R G^< I) = -\alpha(S^R (x) I) * vec_R(G^<)
  //      vec_R(\alpha I G^< S^A)  =  \alpha(I (x) (S^A)^T) * vec_R(G^<) 
  //                               =  \alpha(I (x) (S^R)^*) * vec_R(G^<)
  //
  // Correction to Liouvillian: \Delta A_i = -\alpha(S^R (x) I) + \alpha(I (x) (S^R)^*)
  //
  // ----------------------------------------------------------------------------------------
  // 3. THE EXPLICIT TERMS (RHS)
  // ----------------------------------------------------------------------------------------
  // The remaining terms are safely computed explicitly and stored in C_half.
  // OPTIMIZATION: For the l=i loop of (iG^R S^<), we know exactly that G^R(t,t) = -i1.
  //      i * w_ii * h * (-i1) * S^< = 1.0 * w_ii * h * S^<
  // 
  // Finally, the known j=0 derivative boundary and the -2\xi i\ell^< driving term are 
  // inherently anti-Hermitian. They are added AFTER the (C_half - C_half^\dagger) 
  // symmetrization so their magnitudes are not artificially doubled.
  // ========================================================================================
  if(rho_version_ == 2) {
    // Pre-calculate sizes
    int n2 = nao_ * nao_;
    int sys_size = k_ * n2;

    // Allocate the global system matrices
    ZMatrix L_global = ZMatrix::Zero(sys_size, sys_size);
    ZColVector RHS_global = ZColVector::Zero(sys_size); // USING YOUR ZColVector

    // Identities needed for Kronecker products
    ZMatrix I_nao = ZMatrix::Identity(nao_, nao_);
    ZMatrix I_n2 = ZMatrix::Identity(n2, n2);

    // 1. Build the Time Derivative Operator: D (x) I_n2
    ZMatrix D_mat = ZMatrix::Zero(k_, k_);
    for(int i = 1; i <= k_; i++) {
        for(int j = 1; j <= k_; j++) {
            D_mat(i-1, j-1) = 1.0 / h * I.poly_diff(i,j);
        }
    }
    L_global += Eigen::kroneckerProduct(D_mat, I_n2).eval();

    // 2. Build the Spatial Operators and RHS per timestep
    for(int i = 1; i <= k_; i++) {
        // Construct the effective non-Hermitian Hamiltonian for step i
        ZMatrix h_o_i = ZMatrixMap(H+i*es_, nao_, nao_) 
                      - cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_)) 
                      - mu * IMap;

        // Build the Base Liouvillian A_i (Row-Major)
        ZMatrix A_i = -cplxi * Eigen::kroneckerProduct(h_o_i, I_nao).eval() 
                  + cplxi * Eigen::kroneckerProduct(I_nao, h_o_i.conjugate()).eval();

        // Pull the l=i collision integral term (from loop 3) into the LHS
        double w_ii = I.poly_integ(0, i, i);
        std::complex<double> alpha = cplxi * w_ii * h;
        ZMatrix Sigma_R_ii = ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,i), nao_, nao_);

        A_i += -alpha * Eigen::kroneckerProduct(Sigma_R_ii, I_nao).eval() 
               + alpha * Eigen::kroneckerProduct(I_nao, Sigma_R_ii.conjugate()).eval();

        // Subtract total A_i from the diagonal block of L_global
        L_global.block((i-1)*n2, (i-1)*n2, n2, n2) -= A_i;

        // --- Compute RHS Constants for step i ---
        ZMatrix C_half = ZMatrix::Zero(nao_, nao_);
        
        // Loop 1: G^R * Sigma^< (Safe for l < i)
        for(int l = 0; l <= i; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
        }

        // Loop 2: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 3: l < i (l=i was successfully moved to LHS)
        for(int l = 0; l < i; l++) { 
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 4: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
        }
        
        // Apply the physical symmetry trick to generate the missing conjugate halves
        ZMatrix C_i = C_half - C_half.adjoint();

        // --- ADD FULLY FORMED TERMS ---
        // (These are added AFTER symmetrization so they are not artificially doubled)
        C_i -= 1.0 / h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
        C_i -= 2.0 * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
        
        // Flatten C_i into the global RHS column vector
        RHS_global.segment((i-1)*n2, n2) = Eigen::Map<ZColVector>(C_i.data(), n2);
    }
        
    // 3. Direct Exact Solve
    Eigen::FullPivLU<ZMatrix> lu(L_global);
    ZColVector G_flat = lu.solve(RHS_global);
    
    // 4. Map the flattened solution back into the central storage
    for(int i = 1; i <= k_; i++) {
        // Extract the N x N block. Because ZMatrix is RowMajor, Mapping the column vector 
        // segment back to ZMatrix correctly unpacks it row-by-row!
        ZMatrix G_solved = Eigen::Map<ZMatrix>(G_flat.data() + (i-1)*n2, nao_, nao_);
        
        // Strictly enforce anti-hermiticity one last time to kill floating point noise
        ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = 0.5 * (G_solved - G_solved.adjoint());
        err += (ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    }
  }

  return err;
}

/*
double dyson::dyson_start_les_2leg_diss(herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  cplx cplxi = cplx(0,1);

  ZMatrix QIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix XIC = ZMatrix::Zero(k_*nao_, nao_);
  ZMatrix MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  double err = 0;
  int m = 0;

  // FIRST COLUMN, T=0
  for(int n = 1; n <= k_; n++) {
    auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);

    for(int l = 0; l <= k_; l++) {
      if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
      else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
    }

    int l = 0;
    QBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
    QBlock += (cplxi/h * I.poly_diff(n,l) * ZMatrixMap(G.curr_timestep_les_ptr(m,l), nao_, nao_)).transpose();
    // additional dissipative term would go here, but it is proportional to G^R(0,t')
    // if we interpret our integration as starting from t'=0^+, this evaluates to zero.

    ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//    les_it_int(m, n, G, Sigma, M_.data());
    QBlock += ZMatrixMap(M_.data(), nao_, nao_);
  }

  for(int n = 1; n <= k_; n++) {
    for(int l = 1; l <= k_; l++) {
      auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
      MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint equation, so h and i share same sign
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
      if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
      if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
    }
  }
  Eigen::FullPivLU<ZMatrix> lu(MIC);
  XIC = lu.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
  }

  // store diagonal in case we do the diagonal correction.
  // needed for evaluating iteration error
  ZMatrix DIC = ZMatrix::Zero(k_*nao_, nao_);
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_) = ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_);
  }
  // REST OF THE COLUMNS 
  for(m = 1; m <= k_; m++) {
    MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    XIC = ZMatrix::Zero(k_*nao_, nao_);
    QIC = ZMatrix::Zero(k_*nao_, nao_);

    for(int n = 1; n <= k_; n++) {
      auto QBlock = QIC.block((n-1)*nao_, 0, nao_, nao_);
      for(int l = 0; l <= k_; l++) {
        if(m>=l && n>=l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m<l && n>=l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,n), nao_, nao_)).transpose();
        else if(m>=l && n<l) QBlock -= (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(m,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(m<l && n<l) QBlock += (h * I.poly_integ(0,m,l) * ZMatrixMap(G.curr_timestep_ret_ptr(l,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(n,l), nao_, nao_).adjoint()).transpose();
      }
      QBlock -= (h * I.poly_integ(0,n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,0), nao_, nao_).adjoint()).transpose();
      QBlock -= (cplxi/h * I.poly_diff(n,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,m), nao_, nao_).adjoint()).transpose();
      // additional dissipative term
      // Using notation from Table 4.1 from thesis q_n = \xi 2 G^R_{mn}\ell^<_n
      // Q_n = -i [q_n - y_0 M_{0n}]
      if(m>=n) QBlock -= (2. * G.sig() * cplxi * ZMatrixMap(G.curr_timestep_ret_ptr(m,n), nao_, nao_) * ZMatrixMap(ellL + n*es_, nao_, nao_)).transpose();

      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
        // h_o = h - i(\ell^> -\xi \ell^<)
        // we use the adjoint equation, so h and i share same sign
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n==l) MBlock -= cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_)).transpose();
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      if(rho_version_ != 0) err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      else                  err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
    }
  }

  if(rho_version_==1) {
    // redo diagonal
    MIC = ZMatrix::Zero(k_*nao_, k_*nao_);
    XIC = ZMatrix::Zero(k_*nao_, nao_);
    QIC = ZMatrix::Zero(k_*nao_, nao_);


    for(int i = 1; i <= k_; i++) {
      for(int j = 1; j <= k_; j++) {
        auto MBlock = MIC.block((i-1)*nao_, (j-1)*nao_, nao_, nao_);
        MBlock = 1./h * I.poly_diff(i,j) * IMap;
      }
    }

    // Solving \partial_t G = -ih_oG -iG^\dagger h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = -ih_oG +iG h_o^\dagger -iI -iI^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-ih_oG-iI] -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [iG^\dagger h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] + [-iG h_o^\dagger + iI^dagger]^\dagger -2\xi i\ell^<
    //                      = [iG h_o^\dagger-iI^\dagger] - [iG h_o^\dagger - iI^dagger]^\dagger -2\xi i\ell^<
    // I         =  S^R G^< + S^< G^A
    // I^\dagger = -G^< S^A - G^R S^<
    for(int i = 1; i <= k_; i++) {
      auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
      // h_o = h - i(\ell^> -\xi \ell^<)
      // we use the adjoint, so h and i share same sign
      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_)
                                                                                + cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_)
                                                                                           - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_))
                                                                                - mu*IMap);
      for(int l = 0; l <= i; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = 0; l <= i; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = i+1; l <= k_; l++) {
        QBlock -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
      }
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
  //    les_it_int(i, i, G, Sigma, M_.data());
      QBlock += cplxi*ZMatrixMap(M_.data(), nao_, nao_).transpose();

      ZMatrixMap(XIC.data(), nao_, nao_) = QBlock-QBlock.adjoint();
      QBlock = ZMatrixMap(XIC.data(), nao_, nao_);

      QBlock -= 1./h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
      // additional dissipative term
      QBlock -= 2. * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
    }

    Eigen::FullPivLU<ZMatrix> lu3(MIC);
    XIC = lu3.solve(QIC);
    for(int i = 1; i <= k_; i++) {
      err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
    }
  }

  // ========================================================================================
  // EXACT DIRECT SOLVER FORMALISM (LIOUVILLE SPACE)
  // ========================================================================================
  // Original Equation of Motion:
  // \partial_t G^< = [iG^< h_o^\dagger - iI^\dagger] - [iG^< h_o^\dagger - iI^\dagger]^\dagger - 2\xi i\ell^<
  // 
  // Where the collision integral is:
  // I         = S^R G^< + S^< G^A
  // I^\dagger = -G^< S^A - G^R S^<
  //
  // Let C_half = iG^< h_o^\dagger - iI^\dagger 
  //            = iG^< h_o^\dagger + iG^< S^A + iG^R S^<
  // By computing C_half and taking (C_half - C_half^\dagger), we perfectly generate 
  // both the Left and Right acting Hamiltonian and Collision terms while strictly 
  // enforcing anti-Hermiticity (G^< = -G^<^\dagger).
  //
  // ----------------------------------------------------------------------------------------
  // VECTORIZATION IDENTITY (ROW-MAJOR)
  // ----------------------------------------------------------------------------------------
  // To solve this implicitly without a fixed-point loop, we map the N x N matrices into 
  // an N^2 x 1 column vector. Because Eigen maps RowMajor matrices by stacking rows, we 
  // MUST use the Row-Major Kronecker identity:
  //      vec_R(A * X * B) = (A (x) B^T) * vec_R(X)
  // Note: Since (B^\dagger)^T = B^*, right-multiplying by B^\dagger maps to (I (x) B^*).
  //
  // ----------------------------------------------------------------------------------------
  // 1. THE SPATIAL HAMILTONIAN (LHS)
  // ----------------------------------------------------------------------------------------
  // The commutator generated by the symmetry trick is: -ih_o G^< + iG^< h_o^\dagger
  // Applying the row-major identity:
  //      vec_R(-ih_o G^< I)  = -i(h_o (x) I) * vec_R(G^<)
  //      vec_R(i I G^< h_o^\dagger) =  i(I (x) h_o^*) * vec_R(G^<)
  // 
  // Base Liouvillian: A_i = -i(h_o (x) I) + i(I (x) h_o^*)
  //
  // ----------------------------------------------------------------------------------------
  // 2. EXTRACTING THE IMPLICIT COLLISION TERM (LHS)
  // ----------------------------------------------------------------------------------------
  // The l=i term of the collision integral contains the unknown G^<(t_i, t_i). 
  // Inside C_half, this term is: Q = i * w_ii * h * G^< S^A
  // Let \alpha = i * w_ii * h. 
  // When the symmetry trick (Q - Q^\dagger) is applied, it generates:
  //      Q - Q^\dagger = \alpha G^< S^A - (\alpha G^< S^A)^\dagger
  //                    = \alpha G^< S^A - \alpha^* (S^A)^\dagger (G^<)^\dagger
  // Because \alpha is purely imaginary (\alpha^* = -\alpha) and G^< is anti-Hermitian:
  //                    = \alpha G^< S^A - (-\alpha) S^R (-G^<)
  //                    = \alpha G^< S^A - \alpha S^R G^<
  //
  // To make the solver fully implicit, we move this to the LHS block matrix A_i.
  // Applying the row-major vectorization identity to (-\alpha S^R G^< + \alpha G^< S^A):
  //      vec_R(-\alpha S^R G^< I) = -\alpha(S^R (x) I) * vec_R(G^<)
  //      vec_R(\alpha I G^< S^A)  =  \alpha(I (x) (S^A)^T) * vec_R(G^<) 
  //                               =  \alpha(I (x) (S^R)^*) * vec_R(G^<)
  //
  // Correction to Liouvillian: \Delta A_i = -\alpha(S^R (x) I) + \alpha(I (x) (S^R)^*)
  //
  // ----------------------------------------------------------------------------------------
  // 3. THE EXPLICIT TERMS (RHS)
  // ----------------------------------------------------------------------------------------
  // The remaining terms are safely computed explicitly and stored in C_half.
  // OPTIMIZATION: For the l=i loop of (iG^R S^<), we know exactly that G^R(t,t) = -i1.
  //      i * w_ii * h * (-i1) * S^< = 1.0 * w_ii * h * S^<
  // 
  // Finally, the known j=0 derivative boundary and the -2\xi i\ell^< driving term are 
  // inherently anti-Hermitian. They are added AFTER the (C_half - C_half^\dagger) 
  // symmetrization so their magnitudes are not artificially doubled.
  // ========================================================================================
  if(rho_version_ == 2) {
    // Pre-calculate sizes
    int n2 = nao_ * nao_;
    int sys_size = k_ * n2;

    // Allocate the global system matrices
    ZMatrix L_global = ZMatrix::Zero(sys_size, sys_size);
    ZColVector RHS_global = ZColVector::Zero(sys_size); // USING YOUR ZColVector

    // Identities needed for Kronecker products
    ZMatrix I_nao = ZMatrix::Identity(nao_, nao_);
    ZMatrix I_n2 = ZMatrix::Identity(n2, n2);

    // 1. Build the Time Derivative Operator: D (x) I_n2
    ZMatrix D_mat = ZMatrix::Zero(k_, k_);
    for(int i = 1; i <= k_; i++) {
        for(int j = 1; j <= k_; j++) {
            D_mat(i-1, j-1) = 1.0 / h * I.poly_diff(i,j);
        }
    }
    L_global += Eigen::kroneckerProduct(D_mat, I_n2).eval();

    // 2. Build the Spatial Operators and RHS per timestep
    for(int i = 1; i <= k_; i++) {
        // Construct the effective non-Hermitian Hamiltonian for step i
        ZMatrix h_o_i = ZMatrixMap(H+i*es_, nao_, nao_) 
                      - cplxi * (ZMatrixMap(ellG+i*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+i*es_, nao_, nao_)) 
                      - mu * IMap;

        // Build the Base Liouvillian A_i (Row-Major)
        ZMatrix A_i = -cplxi * Eigen::kroneckerProduct(h_o_i, I_nao).eval() 
                  + cplxi * Eigen::kroneckerProduct(I_nao, h_o_i.conjugate()).eval();

        // Pull the l=i collision integral term (from loop 3) into the LHS
        double w_ii = I.poly_integ(0, i, i);
        std::complex<double> alpha = cplxi * w_ii * h;
        ZMatrix Sigma_R_ii = ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,i), nao_, nao_);

        A_i += -alpha * Eigen::kroneckerProduct(Sigma_R_ii, I_nao).eval() 
               + alpha * Eigen::kroneckerProduct(I_nao, Sigma_R_ii.conjugate()).eval();

        // Subtract total A_i from the diagonal block of L_global
        L_global.block((i-1)*n2, (i-1)*n2, n2, n2) -= A_i;

        // --- Compute RHS Constants for step i ---
        ZMatrix C_half = ZMatrix::Zero(nao_, nao_);
        
        // Loop 1: G^R * Sigma^< (Safe for l < i)
        for(int l = 0; l <= i; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
        }

        // Loop 2: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 3: l < i (l=i was successfully moved to LHS)
        for(int l = 0; l < i; l++) { 
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
        }
        
        // Loop 4: Advanced components (starts at i+1)
        for(int l = i+1; l <= k_; l++) {
            C_half -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
        }
        
        // Apply the physical symmetry trick to generate the missing conjugate halves
        ZMatrix C_i = C_half - C_half.adjoint();

        // --- ADD FULLY FORMED TERMS ---
        // (These are added AFTER symmetrization so they are not artificially doubled)
        C_i -= 1.0 / h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
        C_i -= 2.0 * G.sig() * cplxi * ZMatrixMap(ellL+i*es_, nao_, nao_);
        
        // Flatten C_i into the global RHS column vector
        RHS_global.segment((i-1)*n2, n2) = Eigen::Map<ZColVector>(C_i.data(), n2);
    }
        
    // 3. Direct Exact Solve
    Eigen::FullPivLU<ZMatrix> lu(L_global);
    ZColVector G_flat = lu.solve(RHS_global);
    
    // 4. Map the flattened solution back into the central storage
    for(int i = 1; i <= k_; i++) {
        // Extract the N x N block. Because ZMatrix is RowMajor, Mapping the column vector 
        // segment back to ZMatrix correctly unpacks it row-by-row!
        ZMatrix G_solved = Eigen::Map<ZMatrix>(G_flat.data() + (i-1)*n2, nao_, nao_);
        
        // Strictly enforce anti-hermiticity one last time to kill floating point noise
        ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = 0.5 * (G_solved - G_solved.adjoint());
        err += (ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    }
  }

  return err;
}
*/

} // namespace





















