#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

double dyson::dyson_timestep_ret_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){

  std::chrono::time_point<std::chrono::system_clock> retstart, retend, intstart, intend;
  std::chrono::duration<double, std::micro> retdur(0), intdur(0), intcurrdur(0), intblkdur(0), intcorrdur(0), intdirdur(0), intdircorrdur(0);

  retstart = std::chrono::system_clock::now();


  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  int m, l, n, i;
  double err = 0;
  cplx ncplxi = cplx(0,-1);

  // Matrix Maps
  ZMatrixMap MMap = ZMatrixMap(M_.data(), k_*nao_, k_*nao_);
  ZMatrixMap QMap = ZMatrixMap(Q_.data(), k_*nao_, nao_);
  ZMatrixMap XMap = ZMatrixMap(X_.data(), k_*nao_, nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  memset(M_.data(), 0, k_*k_*es_*sizeof(cplx));
  memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
  memset(bound_.data(), 0, es_*sizeof(cplx));

  int next_block_apply_at = -1;
  int next_block = Sigma.built_blocks()-1;
  if ( next_block != -1 ) {
    next_block_apply_at = tstp-Sigma.blkr1(next_block);
  }

  // Initial condition goes into directly stored slice
  ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).noalias() = ncplxi * IMap;


  // Must solve bootstrapping equation for steps 1 through k
  // X*M = Q, which we solve as (M^T)*(X^T) = (Q^T)
  // Build M^T and Q^T directly
  for(n = 1; n <= k_; n++){
    ZMatrixMap QMapBlock = ZMatrixMap(Q_.data() + (n-1)*es_, nao_, nao_);

    for(l = 0; l <= k_; l++) {
      auto MMapBlock = MMap.block((n-1)*nao_,(l==0)? 0:(l-1)*nao_, nao_, nao_);

      // Known derivative term goes into Q
      if(l == 0) {
        QMapBlock.noalias() += ncplxi / h * I.poly_diff(n,l) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).transpose();
      }
      else {  // Other derivative terms go into M
        MMapBlock.noalias() -= ncplxi / h * I.poly_diff(n,l) * IMap;
      }

      // Delta Energy term
      if(l == n) {
        // h_o = h - i(\ell^> -\xi \ell^<)
        // EOM contains -G^R(t,t-t')[\epsilon^\mu(t-t')]
        //           =  -G^R(t,t-t')[h - \mu - i(\ell^> -\xi \ell^<)]
        MMapBlock.noalias() += mu * IMap - ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
        MMapBlock.noalias() -= ncplxi * ZMatrixMap(ellG + (tstp-n)*es_, nao_, nao_).transpose();
        MMapBlock.noalias() -= ncplxi * -1 * G.sig() * ZMatrixMap(ellL + (tstp-n)*es_, nao_, nao_).transpose();
      }

      // Integral terms
      if(l == 0) { // We know both G and Sigma so this goes into Q
          QMapBlock.noalias() += h*I.gregory_weights(n,l) * (ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_)
                                * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,tstp-n), nao_, nao_)).transpose();
      }
      else { // Goes into M
        if(tstp-l >= tstp-n){ // We have Sig
          MMapBlock.noalias() -= h*I.gregory_weights(n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp-l,tstp-n), nao_, nao_).transpose();
        }
        else { // Don't have Sig
            MMapBlock.noalias() += h*I.gregory_weights(n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp-n,tstp-l), nao_, nao_).conjugate();
        }
      }
    }
  }

  // Solve XM=Q for X (M^T * X^T = Q^T)
  // We already constructed Q and M to be transposed
  Eigen::FullPivLU<ZMatrix> lu(MMap);
  XMap = lu.solve(QMap);

  // Place results into directly stored slice, calculate error compared to previous iteration
  for(l=0; l<k_; l++) {
    err += (ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp-l-1) , nao_, nao_) - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp-l-1) , nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }


  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  ZMatrixMap XMapSmall = ZMatrixMap(X_.data(), nao_, nao_);

  intstart = std::chrono::system_clock::now();
  // Do first part of integral
  memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
  for( n = k_+1; n <= tstp; n++) {
    ZMatrixMap QMapSmall = ZMatrixMap(Q_.data()+(tstp-n)*es_, nao_, nao_);
    for( l = 0; l <= k_; l++) {
      QMapSmall.noalias() += I.gregory_weights(n,l) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tstp-l), nao_, nao_)
                                          * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp-l, tstp-n), nao_, nao_);
    }
  }
  intend = std::chrono::system_clock::now();
  intcurrdur += intend-intstart;
  intdur += intend-intstart;

  // Timestepping yay
  for(n = k_+1; n <= tstp; n++) {
    ZMatrixMap QMapSmall = ZMatrixMap(Q_.data() + (tstp-n)*es_, nao_, nao_);
    QMapSmall *= h;
    // Now we need to include the known derivative values into the RHS
    for(l = 1; l <= k_+1; l++) {
        QMapSmall.noalias() += I.bd_weights(l) * ncplxi / h * ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp-(n-l)), nao_, nao_);
    }

    // M contains four parts: H, mu, unknown derivative term, and unknown integral term
    // h_o = h - i(\ell^> -\xi \ell^<)
    // EOM contains -G^R(t,t-t')[\epsilon^\mu(t-t')]
    //           =  -G^R(t,t-t')[h - \mu - i(\ell^> -\xi \ell^<)]
    MMapSmall.noalias() = -ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
    MMapSmall.noalias() -= ncplxi * (ZMatrixMap(ellG + (tstp-n)*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL + (tstp-n)*es_, nao_, nao_)).transpose();
    MMapSmall.noalias() -= h * I.omega(0) * ZMatrixMap(Sigma.retptr_col(tstp-n,tstp-n), nao_, nao_).transpose();
    MMapSmall.noalias() += (mu - I.bd_weights(0) * ncplxi / h) * IMap;

    // Solve XM=Q (M^T * X^T = Q^T) for X
    // Calculate error and put into directly stored slice
    Eigen::FullPivLU<ZMatrix> lu2(MMapSmall); // we already did the transposition of M above
    XMapSmall.noalias() = lu2.solve(QMapSmall.transpose()).transpose();
    err += (ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp-n) , nao_, nao_) - XMapSmall).norm();
    ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tstp-n), nao_, nao_).noalias() = XMapSmall;

    // Now we have a new value of G which needs to be put into integrals
    intstart = std::chrono::system_clock::now();
    if(n == next_block_apply_at) {
      for( int j = 0; j < nao_; j++) {
        for( int k = 0; k < nao_; k++) {
          int blkrows = Sigma.blklen(next_block);
          int blkcols = Sigma.ret().data().blocks()[next_block][k][j].cols();
          int epsrank = Sigma.ret().data().blocks()[next_block][k][j].epsrank();
          ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);
          ZMatrixMap tmp2(epsnao_tmp_2_.data(), epsrank, nao_);

          tmp.noalias() = ZMatrixMap(Sigma.ret().data().blocks()[next_block][k][j].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(G.curr_timestep_ret_ptr(tstp, Sigma.blkr1(next_block)) + k, blkrows, nao_, Eigen::InnerStride<>(nao_));

          tmp2.noalias() = DColVectorMap(Sigma.ret().data().blocks()[next_block][k][j].Sdata(), epsrank).asDiagonal() * tmp;

          Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkc1(next_block)*es_ + j, blkcols, nao_, Eigen::InnerStride<>(nao_)).noalias() += ZMatrixMap(Sigma.ret().data().blocks()[next_block][k][j].Vdata(), blkcols, epsrank).conjugate() * tmp2;
        }
      }

      next_block--;
      if ( next_block != -1 ) {
        next_block_apply_at = tstp-Sigma.blkr1(next_block);
      }
    }
    intend = std::chrono::system_clock::now();
    intblkdur += intend-intstart;
    intdur +=    intend-intstart;

    // Lastly we have to complete the integral for the next timestep, this amounts to 
    // adding in the directly stored elements.
    // First argument is the case where the svd block boundary lies above the integral region
    // second argument is the case where the curr_timestep() tensor is the cutoff
    if( n != tstp) {

      intstart = std::chrono::system_clock::now();
      int lstart = std::max(tstp-Sigma.r2_dir(tstp-n-1), k_+1);
      for(int ll = tstp-n; ll < tstp-lstart+1; ll++) {
        double weight = ll > tstp-n+k_-1 ? 1. : I.gregory_weights(n+1, tstp-ll);
        ZMatrixMap(Q_.data() + (tstp-n-1) * es_, nao_, nao_) += weight * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, ll), nao_, nao_) * ZMatrixMap(Sigma.retptr_col(ll, tstp-n-1), nao_, nao_);
      }
      intend = std::chrono::system_clock::now();
      intdirdur += intend-intstart;
      intdur += intend-intstart;

      // There is one caveat, which is if there are less than k+1 stored in a column, we have to
      // adjust some of the values from the svd blocks, which at the moment have coefficient 1
      int lstartcorr = std::max(n+1-k_, k_+1);
      int lendcorr = tstp-Sigma.r2_dir(tstp-n-1)-1;
      intstart = std::chrono::system_clock::now();
      for( l = lstartcorr; l <= lendcorr; l++) {
        ZMatrixMap(Q_.data() + (tstp-n-1) * es_, nao_, nao_) += (I.gregory_weights(n+1, l) - 1.)* ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tstp-l), nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(tstp-l,tstp-n-1), nao_, nao_);
      }
      intend = std::chrono::system_clock::now();
      intcorrdur += intend-intstart;
      intdur += intend-intstart;
    }
  }

  retend = std::chrono::system_clock::now();
  retdur = retend-retstart;

  if(profile_) timing(tstp, 0) = retdur.count();
  if(profile_) timing(tstp, 1) = intdur.count();
  if(profile_) timing(tstp, 2) = intcurrdur.count();
  if(profile_) timing(tstp, 3) = intblkdur.count();
  if(profile_) timing(tstp, 4) = intdirdur.count();
  if(profile_) timing(tstp, 5) = intdircorrdur.count();
  if(profile_) timing(tstp, 6) = intcorrdur.count();

  return err;
}

double dyson::dyson_timestep_les_nobc_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  std::chrono::time_point<std::chrono::system_clock> lesstart, lesend, int1start, int1end, int2start, int2end, int3start, int3end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0), int3dur(0), int3dur_curr(0), int3dur_corr(0), int3dur_block(0);

  lesstart = std::chrono::system_clock::now();

  int next_block_apply_at = Sigma.blkc2(0);
  int next_block = 0;

  double err = 0;
  cplx cplxi = cplx(0,1);

  // Matrix Maps
  ZMatrixMap MMap = ZMatrixMap(M_.data(), k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  ZMatrixMap SMap = ZMatrixMap(bound_.data(), nao_, nao_); // Used for boundary cases where we need to extract Sigma from SVD

  // Integrals go into Q via increment.  Q must be 0.
  memset(Q_.data(),0,sizeof(cplx)*(tstp+1)*es_);
  memset(X_.data(),0,sizeof(cplx)*(tstp+1)*es_);
  memset(M_.data(),0,sizeof(cplx)*k_*k_*es_);


  // INTEGRAL 1 AND 2 GO
  int2start = std::chrono::system_clock::now();
  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = std::chrono::system_clock::now();
  int2dur = int2end-int2start;

  int1start = std::chrono::system_clock::now();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = std::chrono::system_clock::now();
  int1dur = int1end-int1start;

  // Initial Condition. Do not use tv, remember integral is transposed
  // integral
  memset(Q_.data(),0,sizeof(cplx)*es_);
  les_it_int(0, tstp, G, Sigma, Q_.data());

  // derivative
  for(int l = 1; l <= k_; l++) {
    ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(0,tstp-l), nao_, nao_).transpose();
  }
  G.get_les(0,tstp-(k_+1), X_.data());
  ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(k_+1) * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  
  // M 
  // h_o = h - i(\ell^> -\xi \ell^<)
  // we take the hermitian conjugate in the adjoint equation - ellL, ellG, h, are all self-adjoint
  // \pm 2iG^R(t, t')\ell^<(t') is zero, as t < t'
  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap
                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_)
                              + cplxi * (ZMatrixMap(ellG + tstp*nao_*nao_, nao_, nao_) - G.sig() * ZMatrixMap(ellL + tstp*nao_*nao_, nao_, nao_))
                              - mu * IMap
                              + h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).adjoint()).transpose();

  // Last integral
  // curr_timestep
  for(int l = tstp-k_; l < tstp; l++) {
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * I.gregory_weights(tstp,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(G.curr_timestep_les_ptr(0,l), nao_, nao_).transpose();
  }
  // direct
  for(int l = 0; l < std::min(tstp-k_, G.blkr1(0)); l++) {
    G.get_les(0,l,X_.data());
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * I.gregory_weights(tstp,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  }
  // block corrections
  for(int l = G.blkr1(0); l < std::min(k_+1, tstp-k_); l++) {
    G.get_les(0,l,X_.data());
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * (I.gregory_weights(tstp,l)-1.) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  }
  // blocks
  for(int l = 0; l < G.nbox(); l++) {
    int b = std::pow(2,l)-1;
    if(b >= G.built_blocks()) break;

    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        int blkrows = G.blklen(b);
        int blkcols = G.les().data().blocks()[b][i][k].cols();
        int epsrank = G.les().data().blocks()[b][i][k].epsrank();
        int lstart = G.blkr1(b);

        ZMatrix tmp(epsrank,nao_);

        tmp = ZMatrixMap(G.les().data().blocks()[b][i][k].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Sigma.curr_timestep_ret_ptr(tstp, lstart) + k, blkrows, nao_, Eigen::InnerStride<>(nao_)).conjugate();

        tmp = -h * DColVectorMap(G.les().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

        ZMatrixMap(Q_.data(), nao_, nao_).col(i) += (ZMatrixMap(G.les().data().blocks()[b][i][k].Vdata(), 1, epsrank).conjugate() * tmp).transpose();
      }
    }
  }
  Eigen::FullPivLU<ZMatrix> luIC(ZMatrixMap(M_.data(), nao_, nao_));
  ZMatrixMap(X_.data(), nao_, nao_) = luIC.solve(ZMatrixMap(Q_.data(), nao_, nao_));
  memset(M_.data(),0,sizeof(cplx)*k_*k_*es_);

  // Set up the kxk linear problem MX=Q
  for(int m = 1; m <= k_; m++) {
    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);

    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // Additional Dissipative Term
    // \mp 2i\ell^<(t)G^A(t, T)
    // \mp 2i\ell^<(t)G^R(T, t)^\dagger
    QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();

    for(int l = 0; l <= k_; l++) {
      auto MMapBlock = MMap.block((m-1)*nao_,(l==0)? 0:(l-1)*nao_, nao_, nao_);

      // Derivative term
      if(l==0){ // We put this in Q
        QMapBlock.noalias() -= cplxi/h * I.poly_diff(m,l) * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // It goes into M
        MMapBlock.noalias() += cplxi/h * I.poly_diff(m,l) * IMap;
      }

      // Delta energy term
      // h_o = h - i(\ell^> -\xi \ell^<)
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
        MMapBlock.noalias() += cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_));
      }

      // Integral term
      if(l==0){ // Goes into Q
        Sigma.get_ret(m, l, SMap.data());
        QMapBlock.noalias() += h*I.gregory_weights(m,l) * SMap * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // Goes into M
        Sigma.get_ret(m, l, SMap.data());
        MMapBlock.noalias() -= h*I.gregory_weights(m,l) * SMap;
      }
    }
  }

  // Solve Mx=Q
  Eigen::FullPivLU<ZMatrix> lu(MMap);
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));

  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  int3start = std::chrono::system_clock::now();
  // Do first part of retles integral
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    if(Sigma.blkc2(b) <= k_) {
      if(Sigma.blkr2(b) > k_) {
        for(int i = 0; i < nao_; i++) {
          for(int k = 0; k < nao_; k++) {
            int blkrows = Sigma.blklen(b);
            int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
            int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
            ZMatrix tmp(epsrank, nao_);

            tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

            tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

            Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
          }
        }
      }
      next_block++;
      next_block_apply_at = Sigma.blkc2(next_block);
    }
    else {
      break;
    }
  }
  int3end = std::chrono::system_clock::now();
  int3dur += int3end-int3start;
  int3dur_block += int3end-int3start;

  // Timestepping
  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  for(int m = k_+1; m <= tstp; m++) {

    // Check if next box can be applied
    int3start = std::chrono::system_clock::now();
    if(next_block_apply_at == m-1) {
      int b = next_block;
      for(int i = 0; i < nao_; i++) {
        for(int k = 0; k < nao_; k++) {
          int blkrows = Sigma.blklen(b);
          int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
          int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
          ZMatrix tmp(epsrank, nao_);

          tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

          tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

          Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
        }
      }
      next_block++;
      if(next_block < Sigma.nbox()) next_block_apply_at = Sigma.blkc2(next_block);
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_block += int3end-int3start;

    // finish direct
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk()) {
      for(int tbar = Sigma.c1_dir(m); tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data()+tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_col(m,tbar), nao_, nao_).transpose();
      }
    }
    else {
      for(int tbar = 0; tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_curr += int3end-int3start;

    // do edge corrections on left
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk() && m >= Sigma.blkr1(0)) {
      for(int tbar = 0; tbar <= std::min(k_, Sigma.c1_dir(m)-1); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.ret_left_edge().data() + (m*(k_+1) + tbar)*nao_*nao_, nao_, nao_).transpose();
      }
      // do edge corrections on right
      for(int tbar = std::max(m-k_,k_+1); tbar < Sigma.c1_dir(m); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_corr += int3end-int3start;


    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // BULK OF ROW
    if(m != tstp) {
      // Set up M
      // h_o = h - i(\ell^> -\xi \ell^<)
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));
        
      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));

      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      // h_o = h - i(\ell^> -\xi \ell^<)
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      QMapBlock.noalias() += -cplxi * (ZMatrixMap(ellG+tstp*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+tstp*es_, nao_, nao_)) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
      // dissipative term
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() += - 2. * G.sig() * cplxi * ZMatrixMap(ellL+tstp*es_, nao_, nao_);
      // Derivatives into Q
      for(int l = 1; l <= k_; l++) {
        ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-l, tstp-l), nao_, nao_);
      }
      G.get_les(tstp-k_-1, tstp-k_-1, M_.data());
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(k_+1) * ZMatrixMap(M_.data(), nao_, nao_);
      MMapSmall.noalias() = 1./h*I.bd_weights(0) * IMap;
      ZMatrixMap(Q_.data() + m*es_, nao_, nao_) = ZMatrixMap(Q_.data(), nao_, nao_);
    }

    // Solve MX=Q
    Eigen::FullPivLU<ZMatrix> lu2(MMapSmall);
    ZMatrixMap(X_.data() + m*es_, nao_, nao_) = lu2.solve(ZMatrixMap(Q_.data() + m*es_, nao_, nao_)).transpose();
  }

  // Write elements into G
  for(int l = 0; l <= tstp; l++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_) - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }
  lesend = std::chrono::system_clock::now();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();
  if(profile_) timing(tstp, 21) = int3dur.count();
  if(profile_) timing(tstp, 22) = int3dur_block.count();
  if(profile_) timing(tstp, 23) = int3dur_curr.count();
  if(profile_) timing(tstp, 24) = int3dur_corr.count();

  return err;
}

double dyson::dyson_timestep_les_2leg_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  std::chrono::time_point<std::chrono::system_clock> lesstart, lesend, int1start, int1end, int2start, int2end, int3start, int3end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0), int3dur(0), int3dur_curr(0), int3dur_corr(0), int3dur_block(0);

  lesstart = std::chrono::system_clock::now();

  int next_block_apply_at = Sigma.blkc2(0);
  int next_block = 0;

  double err = 0;
  cplx cplxi = cplx(0,1);

  // Matrix Maps
  ZMatrixMap MMap = ZMatrixMap(M_.data(), k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  ZMatrixMap SMap = ZMatrixMap(bound_.data(), nao_, nao_); // Used for boundary cases where we need to extract Sigma from SVD

  // Integrals go into Q via increment.  Q must be 0.
  memset(Q_.data(),0,sizeof(cplx)*(tstp+1)*es_);
  memset(X_.data(),0,sizeof(cplx)*(tstp+1)*es_);
  memset(M_.data(),0,sizeof(cplx)*k_*k_*es_);


  // INTEGRAL 1 AND 2 GO
  int2start = std::chrono::system_clock::now();
//  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = std::chrono::system_clock::now();
  int2dur = int2end-int2start;

  int1start = std::chrono::system_clock::now();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = std::chrono::system_clock::now();
  int1dur = int1end-int1start;

  // Initial Condition. Do not use tv, remember integral is transposed
  // integral
  memset(Q_.data(),0,sizeof(cplx)*es_);
//  les_it_int(0, tstp, G, Sigma, Q_.data());

  // derivative
  for(int l = 1; l <= k_; l++) {
    ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(0,tstp-l), nao_, nao_).transpose();
  }
  G.get_les(0,tstp-(k_+1), X_.data());
  ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(k_+1) * ZMatrixMap(X_.data(), nao_, nao_).transpose();

  // M 
  // h_o = h - i(\ell^> -\xi \ell^<)
  // we take the hermitian conjugate in the adjoint equation - ellL, ellG, h, are all self-adjoint
  // \pm 2iG^R(t, t')\ell^<(t') is zero, as t < t'
  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap
                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_)
                              + cplxi * (ZMatrixMap(ellG + tstp*nao_*nao_, nao_, nao_) - G.sig() * ZMatrixMap(ellL + tstp*nao_*nao_, nao_, nao_))
                              - mu * IMap
                              + h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).adjoint()).transpose();

    
  // Last integral
  // curr_timestep
  for(int l = tstp-k_; l < tstp; l++) {
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * I.gregory_weights(tstp,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(G.curr_timestep_les_ptr(0,l), nao_, nao_).transpose();
  }
  // direct
  for(int l = 0; l < std::min(tstp-k_, G.blkr1(0)); l++) {
    G.get_les(0,l,X_.data());
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * I.gregory_weights(tstp,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  }
  // block corrections
  for(int l = G.blkr1(0); l < std::min(k_+1, tstp-k_); l++) {
    G.get_les(0,l,X_.data());
    ZMatrixMap(Q_.data(), nao_, nao_) -= h * (I.gregory_weights(tstp,l)-1.) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,l), nao_, nao_).conjugate()
                                                      * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  }
  // blocks
  for(int l = 0; l < G.nbox(); l++) {
    int b = std::pow(2,l)-1;
    if(b >= G.built_blocks()) break;

    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        int blkrows = G.blklen(b);
        int blkcols = G.les().data().blocks()[b][i][k].cols();
        int epsrank = G.les().data().blocks()[b][i][k].epsrank();
        int lstart = G.blkr1(b);

        ZMatrix tmp(epsrank,nao_);

        tmp = ZMatrixMap(G.les().data().blocks()[b][i][k].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Sigma.curr_timestep_ret_ptr(tstp, lstart) + k, blkrows, nao_, Eigen::InnerStride<>(nao_)).conjugate();

        tmp = -h * DColVectorMap(G.les().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

        ZMatrixMap(Q_.data(), nao_, nao_).col(i) += (ZMatrixMap(G.les().data().blocks()[b][i][k].Vdata(), 1, epsrank).conjugate() * tmp).transpose();
      }
    }
  }
  Eigen::FullPivLU<ZMatrix> luIC(ZMatrixMap(M_.data(), nao_, nao_));
  ZMatrixMap(X_.data(), nao_, nao_) = luIC.solve(ZMatrixMap(Q_.data(), nao_, nao_));

  memset(M_.data(),0,sizeof(cplx)*k_*k_*es_);
  // Set up the kxk linear problem MX=Q
  for(int m = 1; m <= k_; m++) {
    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);

    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // Additional Dissipative Term
    // \mp 2i\ell^<(t)G^A(t, T)
    // \mp 2i\ell^<(t)G^R(T, t)^\dagger
    QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();

    for(int l = 0; l <= k_; l++) {
      auto MMapBlock = MMap.block((m-1)*nao_,(l==0)? 0:(l-1)*nao_, nao_, nao_);

      // Derivative term
      if(l==0){ // We put this in Q
        QMapBlock.noalias() -= cplxi/h * I.poly_diff(m,l) * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // It goes into M
        MMapBlock.noalias() += cplxi/h * I.poly_diff(m,l) * IMap;
      }

      // Delta energy term
      // h_o = h - i(\ell^> -\xi \ell^<)
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
        MMapBlock.noalias() += cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_));
      }

      // Integral term
      if(l==0){ // Goes into Q
        Sigma.get_ret(m, l, SMap.data());
        QMapBlock.noalias() += h*I.gregory_weights(m,l) * SMap * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // Goes into M
        Sigma.get_ret(m, l, SMap.data());
        MMapBlock.noalias() -= h*I.gregory_weights(m,l) * SMap;
      }
    }
  }

  // Solve Mx=Q
  Eigen::FullPivLU<ZMatrix> lu(MMap);
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));

  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  int3start = std::chrono::system_clock::now();
  // Do first part of retles integral
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    if(Sigma.blkc2(b) <= k_) {
      if(Sigma.blkr2(b) > k_) {
        for(int i = 0; i < nao_; i++) {
          for(int k = 0; k < nao_; k++) {
            int blkrows = Sigma.blklen(b);
            int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
            int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
            ZMatrix tmp(epsrank, nao_);

            tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

            tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

            Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
          }
        }
      }
      next_block++;
      next_block_apply_at = Sigma.blkc2(next_block);
    }
    else {
      break;
    }
  }
  int3end = std::chrono::system_clock::now();
  int3dur += int3end-int3start;

  // Timestepping
  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  for(int m = k_+1; m <= tstp; m++) {

    // Check if next box can be applied
    int3start = std::chrono::system_clock::now();
    if(next_block_apply_at == m-1) {
      int b = next_block;
      for(int i = 0; i < nao_; i++) {
        for(int k = 0; k < nao_; k++) {
          int blkrows = Sigma.blklen(b);
          int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
          int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
          ZMatrix tmp(epsrank, nao_);

          tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

          tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

          Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
        }
      }
      next_block++;
      if(next_block < Sigma.nbox()) next_block_apply_at = Sigma.blkc2(next_block);
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_block += int3end-int3start;

    // finish direct
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk()) {
      for(int tbar = Sigma.c1_dir(m); tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data()+tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_col(m,tbar), nao_, nao_).transpose();
      }
    }
    else {
      for(int tbar = 0; tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_curr += int3end-int3start;

    // do edge corrections on left
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk() && m >= Sigma.blkr1(0)) {
      for(int tbar = 0; tbar <= std::min(k_, Sigma.c1_dir(m)-1); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.ret_left_edge().data() + (m*(k_+1) + tbar)*nao_*nao_, nao_, nao_).transpose();
      }
      // do edge corrections on right
      // k+1 case is for when m is small.  Right and left correction regions overlap.  Taken care of by left
      for(int tbar = std::max(m-k_,k_+1); tbar < Sigma.c1_dir(m); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_corr += int3end-int3start;


    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // BULK OF ROW
    if(m != tstp) {
      // Set up M
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      // h_o = h - i(\ell^> -\xi \ell^<)
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));
        
      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      // h_o = h - i(\ell^> -\xi \ell^<)
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));

      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      // h_o = h - i(\ell^> -\xi \ell^<)
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      QMapBlock.noalias() += -cplxi * (ZMatrixMap(ellG+tstp*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+tstp*es_, nao_, nao_)) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
      // dissipative term
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() += - 2. * G.sig() * cplxi * ZMatrixMap(ellL+tstp*es_, nao_, nao_);
      // Derivatives into Q
      for(int l = 1; l <= k_; l++) {
        ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-l, tstp-l), nao_, nao_);
      }
      G.get_les(tstp-k_-1, tstp-k_-1, M_.data());
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(k_+1) * ZMatrixMap(M_.data(), nao_, nao_);
      MMapSmall.noalias() = 1./h*I.bd_weights(0) * IMap;
      ZMatrixMap(Q_.data() + m*es_, nao_, nao_) = ZMatrixMap(Q_.data(), nao_, nao_);
    }

    // Solve MX=Q
    Eigen::FullPivLU<ZMatrix> lu2(MMapSmall);
    ZMatrixMap(X_.data() + m*es_, nao_, nao_) = lu2.solve(ZMatrixMap(Q_.data() + m*es_, nao_, nao_)).transpose();
  }

  // Write elements into G
  for(int l = 0; l <= tstp; l++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_) - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }
  lesend = std::chrono::system_clock::now();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();
  if(profile_) timing(tstp, 21) = int3dur.count();
  if(profile_) timing(tstp, 22) = int3dur_block.count();
  if(profile_) timing(tstp, 23) = int3dur_curr.count();
  if(profile_) timing(tstp, 24) = int3dur_corr.count();

  return err;
}

double dyson::dyson_timestep_les_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  std::chrono::time_point<std::chrono::system_clock> lesstart, lesend, int1start, int1end, int2start, int2end, int3start, int3end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0), int3dur(0), int3dur_curr(0), int3dur_corr(0), int3dur_block(0);

  lesstart = std::chrono::system_clock::now();

  int next_block_apply_at = Sigma.blkc2(0);
  int next_block = 0;

  double err = 0;
  cplx cplxi = cplx(0,1);

  // Matrix Maps
  ZMatrixMap MMap = ZMatrixMap(M_.data(), k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  ZMatrixMap SMap = ZMatrixMap(bound_.data(), nao_, nao_); // Used for boundary cases where we need to extract Sigma from SVD


  // Integrals go into Q via increment.  Q must be 0.
  memset(Q_.data(),0,sizeof(cplx)*(tstp+1)*es_);

  // INTEGRAL 1 AND 2 GO - Integrals are transposed
  int2start = std::chrono::system_clock::now();
  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = std::chrono::system_clock::now();
  int2dur = int2end-int2start;

  int1start = std::chrono::system_clock::now();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = std::chrono::system_clock::now();
  int1dur = int1end-int1start;

  // Initial Condition. remember that we transpose X at the end when filling G
  G.get_tv_tau(tstp, 0, dlr_, M_.data());
  ZMatrixMap(X_.data(), nao_, nao_).noalias() = -ZMatrixMap(M_.data(), nao_, nao_).conjugate();
  memset(M_.data(),0,sizeof(cplx)*k_*k_*es_);

  // Set up the kxk linear problem MX=Q
  for(int m = 1; m <= k_; m++) {
    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);

    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // Additional Dissipative Term
    // \mp 2i\ell^<(t)G^A(t, T)
    // \mp 2i\ell^<(t)G^R(T, t)^\dagger
    QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();

    for(int l = 0; l <= k_; l++) {
      auto MMapBlock = MMap.block((m-1)*nao_,(l==0)? 0:(l-1)*nao_, nao_, nao_);

      // Derivative term
      if(l==0){ // We put this in Q
        QMapBlock.noalias() -= cplxi/h * I.poly_diff(m,l) * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // It goes into M
        MMapBlock.noalias() += cplxi/h * I.poly_diff(m,l) * IMap;
      }

      // Delta energy term
      // h_o = h - i(\ell^> -\xi \ell^<)
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
        MMapBlock.noalias() += cplxi * (ZMatrixMap(ellG+l*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+l*es_, nao_, nao_));
      }

      // Integral term
      if(l==0){ // Goes into Q
        Sigma.get_ret(m, l, SMap.data());
        QMapBlock.noalias() += h*I.gregory_weights(m,l) * SMap * ZMatrixMap(X_.data(), nao_, nao_).transpose();
      }
      else{ // Goes into M
        Sigma.get_ret(m, l, SMap.data());
        MMapBlock.noalias() -= h*I.gregory_weights(m,l) * SMap;
      }
    }
  }
  // Solve Mx=Q
  Eigen::FullPivLU<ZMatrix> lu(MMap);
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  int3start = std::chrono::system_clock::now();
  // Do first part of retles integral
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    if(Sigma.blkc2(b) <= k_) {
      if(Sigma.blkr2(b) > k_) {
        for(int i = 0; i < nao_; i++) {
          for(int k = 0; k < nao_; k++) {
            int blkrows = Sigma.blklen(b);
            int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
            int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
            ZMatrix tmp(epsrank, nao_);

            tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

            tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

            Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
          }
        }
      }
      next_block++;
      next_block_apply_at = Sigma.blkc2(next_block);
    }
    else {
      break;
    }
  }
  int3end = std::chrono::system_clock::now();
  int3dur += int3end-int3start;
  int3dur_block += int3end-int3start;

  // Timestepping
  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  for(int m = k_+1; m <= tstp; m++) {

    // Check if next box can be applied
    int3start = std::chrono::system_clock::now();
    if(next_block_apply_at == m-1) {
      int b = next_block;
      for(int i = 0; i < nao_; i++) {
        for(int k = 0; k < nao_; k++) {
          int blkrows = Sigma.blklen(b);
          int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
          int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
          ZMatrix tmp(epsrank, nao_);

          tmp = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

          tmp = h * DColVectorMap(Sigma.ret().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

          Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)) += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
        }
      }
      next_block++;
      if(next_block < Sigma.nbox()) next_block_apply_at = Sigma.blkc2(next_block);
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_block += int3end-int3start;

    // finish direct
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk()) {
      for(int tbar = Sigma.c1_dir(m); tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data()+tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_col(m,tbar), nao_, nao_).transpose();
      }
    }
    else {
      for(int tbar = 0; tbar < m; tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_curr += int3end-int3start;

    // do edge corrections on left
    int3start = std::chrono::system_clock::now();
    if(m < Sigma.tstpmk() && m >= Sigma.blkr1(0)) {
      for(int tbar = 0; tbar <= std::min(k_, Sigma.c1_dir(m)-1); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.ret_left_edge().data() + (m*(k_+1) + tbar)*nao_*nao_, nao_, nao_).transpose();
      }
      // do edge corrections on right
      for(int tbar = std::max(m-k_,k_+1); tbar < Sigma.c1_dir(m); tbar++) {
        ZMatrixMap(Q_.data() + m*es_, nao_, nao_) += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = std::chrono::system_clock::now();
    int3dur += int3end-int3start;
    int3dur_corr += int3end-int3start;


    auto QMapBlock = ZMatrixMap(Q_.data() + m*es_, nao_, nao_);
    // integrals are transposed
    QMapBlock = QMapBlock.transpose().eval();

    // BULK OF ROW
    if(m != tstp) {
      // Set up M
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      // h_o = h - i(\ell^> -\xi \ell^<)
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));

      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      // h_o = h - i(\ell^> -\xi \ell^<)
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
      MMapSmall.noalias() += cplxi * (ZMatrixMap(ellG+m*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+m*es_, nao_, nao_));

      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }

      // Additional Dissipative Term
      // \mp 2i\ell^<(t)G^A(t, T)
      // \mp 2i\ell^<(t)G^R(T, t)^\dagger
      QMapBlock.noalias() += - G.sig() * 2 * cplxi * ZMatrixMap(ellL+m*es_, nao_, nao_) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, m), nao_, nao_).adjoint();
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      // h_o = h - i(\ell^> -\xi \ell^<)
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      QMapBlock.noalias() += -cplxi * (ZMatrixMap(ellG+tstp*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+tstp*es_, nao_, nao_)) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
      // dissipative term
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() += - 2. * G.sig() * cplxi * ZMatrixMap(ellL+tstp*es_, nao_, nao_);
      // Derivatives into Q
      for(int l = 1; l <= k_; l++) {
        ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-l, tstp-l), nao_, nao_);
      }
      G.get_les(tstp-k_-1, tstp-k_-1, M_.data());
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(k_+1) * ZMatrixMap(M_.data(), nao_, nao_);
      MMapSmall.noalias() = 1./h*I.bd_weights(0) * IMap;
      ZMatrixMap(Q_.data() + m*es_, nao_, nao_) = ZMatrixMap(Q_.data(), nao_, nao_);
    }

    // Solve MX=Q
    Eigen::FullPivLU<ZMatrix> lu2(MMapSmall);
    ZMatrixMap(X_.data() + m*es_, nao_, nao_) = lu2.solve(ZMatrixMap(Q_.data() + m*es_, nao_, nao_)).transpose();
  }

  // Write elements into G
  for(int l = 0; l <= tstp; l++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_) - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(l,tstp), nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }
  lesend = std::chrono::system_clock::now();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();
  if(profile_) timing(tstp, 21) = int3dur.count();
  if(profile_) timing(tstp, 22) = int3dur_block.count();
  if(profile_) timing(tstp, 23) = int3dur_curr.count();
  if(profile_) timing(tstp, 24) = int3dur_corr.count();

  return err;
}

double dyson::dyson_timestep_tv_diss(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, cplx *ellL, cplx *ellG, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.r() == Sigma.r());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  std::chrono::time_point<std::chrono::system_clock> tvstart, tvend, int1start, int1end, int2start, int2end;
  std::chrono::duration<double, std::micro> tvdur(0), int1dur(0), int2dur(0);

  tvstart = std::chrono::system_clock::now();

  // We allow users to call set_tstp_zero for [T-k...T]
  // If we don't expect them to update tv_trans, we need to 
  // make sure that it is up-to-date for [T-k-1...T]
  // -1 coming from the fact this may be the first time 
  // function is called at T
  for(int t = tstp-k_-1; t <= tstp; t++) {
    for(int i = 0; i < r_; i++) {
      ZMatrixMap(Sigma.tvptr_trans(t,i), nao_, nao_) = ZMatrixMap(Sigma.tvptr(t,i), nao_,nao_).transpose();
    }
  }

  double err = 0;

  cplx cplxi = cplx(0.,1.);
  auto IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  auto XMap = ZMatrixMap(Q_.data(), nao_, nao_);
  auto MMap = ZMatrixMap(M_.data(), nao_, nao_);

  // Do integrals and put them into G.tv(tstp,:)
  int1start = std::chrono::system_clock::now();
  tv_ret_int(tstp, Sigma, G, I, h);
  int1end = std::chrono::system_clock::now();
  int1dur = int1end-int1start;

  int2start = std::chrono::system_clock::now();
  tv_it_conv(tstp, Sigma, G, G.tvptr(tstp,0));
  int2end = std::chrono::system_clock::now();
  int2dur = int2end-int2start;

  // Put derivatives into G.tv(tstp,m)
  for( int l = 1; l <= k_+1; l++) {
    auto GTVMap = ZColVectorMap(G.tvptr(tstp,0), r_*es_);
    GTVMap.noalias() += -cplxi/h*I.bd_weights(l) * ZColVectorMap(G.tvptr(tstp-l,0), r_*es_);
  }

  // Make M
  // h_o = h - i(\ell^> -\xi \ell^<)
  MMap.noalias() = (cplxi/h*I.bd_weights(0) + mu) * IMap
                                             - ZMatrixMap(H+tstp*es_, nao_, nao_)
                                             + cplxi * (ZMatrixMap(ellG+tstp*es_, nao_, nao_) - G.sig() * ZMatrixMap(ellL+tstp*es_, nao_, nao_))
                                             - h*I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_);

  Eigen::FullPivLU<ZMatrix> lu(MMap);

  // Solve MX=Q
  for(int m=0; m<r_; m++) {
    XMap = lu.solve(ZMatrixMap(G.tvptr(tstp, m), nao_, nao_));
    err += (ZMatrixMap(NTauTmp_.data()+m*es_, nao_, nao_) - XMap).norm();
    ZMatrixMap(G.tvptr(tstp,m), nao_, nao_).noalias() = XMap;
    ZMatrixMap(G.tvptr_trans(tstp,m), nao_, nao_).noalias() = XMap.transpose();
  }

  tvend = std::chrono::system_clock::now();
  tvdur = tvend-tvstart;


  if(profile_) timing(tstp, 7) = tvdur.count();
  if(profile_) timing(tstp, 8) = int1dur.count();
  if(profile_) timing(tstp, 9) = int2dur.count();

  return err;
}

} // namespace
