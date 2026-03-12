#include "dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace hodlr {

#include "dyson_free.hpp"
#include "dyson_it_integrals.hpp"


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


double dyson::dyson_timestep_ret(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
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
// DEBUG
//        MMapBlock.noalias() += mu * IMap - ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).conjugate();
        MMapBlock.noalias() += mu * IMap - ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
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
// DEBUG
    MMapSmall.noalias() = -ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).conjugate();
//    MMapSmall.noalias() = -ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
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




double dyson::dyson_timestep_les_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
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
// DEBUG
//  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap 
//                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_).adjoint()
//                              - mu * IMap
//                              + h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).adjoint()).transpose();
  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap 
                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_)
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
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
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
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);

      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
  
      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
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

double dyson::dyson_timestep_les_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
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
// DEBUG
//  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap 
//                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_).adjoint()
//                              - mu * IMap
//                              + h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp,tstp), nao_, nao_).adjoint()).transpose();
  ZMatrixMap(M_.data(), nao_, nao_) = (cplxi/h * I.bd_weights(0) * IMap 
                              + ZMatrixMap(H + tstp*nao_*nao_, nao_, nao_)
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
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
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
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);

      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
  
      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
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

double dyson::dyson_timestep_les(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
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
      if(m==l){
        MMapBlock.noalias() += mu*IMap - ZMatrixMap(H+l*es_, nao_, nao_);
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
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);

      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 0) { // HORIZONTAL FOR RHO
      // Set up M 
      cplx *sigptrmm = m >= Sigma.tstpmk() ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + (cplxi/h*I.bd_weights(0) + mu)*IMap - h*I.omega(0)*ZMatrixMap(sigptrmm, nao_, nao_);
 
      // Derivatives into Q 
      for(int l = 1; l <= k_+1; l++) {
        QMapBlock.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else if(rho_version_ == 1) { // DIAGONAL FOR RHO
      // finish integral
      QMapBlock.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // hamiltonian
      QMapBlock.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (QMapBlock + QMapBlock.adjoint());
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



double dyson::dyson_timestep_tv(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
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
  MMap.noalias() = (cplxi/h*I.bd_weights(0) + mu) * IMap
                                             - ZMatrixMap(H+tstp*es_, nao_, nao_)
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

double dyson::dyson_start_tv(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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
        if(l==n){
          MMapBlock.noalias() += mu*IMap - ZMatrixMap(H + l*es_, nao_, nao_);
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


double dyson::dyson_start_ret(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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

double dyson::dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
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
        if(n==l) MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) += mu*IMap - ZMatrixMap(H + l*es_, nao_, nao_);
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

  if(imp_tp0) {
    for(m = 1; m <= k_; m++) {
      ZMatrixMap MMap0 = ZMatrixMap(M_.data(), nao_, nao_);
      ZMatrixMap QMap0 = ZMatrixMap(Q_.data(), nao_, nao_);
//DEBUG
      MMap0 = -ncplxi/h * I.poly_diff(k_,k_) * IMap - ZMatrixMap(H, nao_, nao_).adjoint() + mu*IMap - h * I.poly_integ(k_-m,k_,k_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(0,0), nao_, nao_);
//      MMap0 = -ncplxi/h * I.poly_diff(k_,k_) * IMap - ZMatrixMap(H, nao_, nao_) + mu*IMap - h * I.poly_integ(k_-m,k_,k_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(0,0), nao_, nao_);
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

double dyson::dyson_start_les(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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

double dyson::dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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
      
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
// DEBUG
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).conjugate() - mu*IMap;
//        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      if(rho_version_ == 1) err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
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
  
    for(int i = 1; i <= k_; i++) {
      auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
// DEBUG
      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_).adjoint() - mu*IMap);
//      QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_) - mu*IMap);
  
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
    }
  
    Eigen::FullPivLU<ZMatrix> lu3(MIC);
    XIC = lu3.solve(QIC);
    for(int i = 1; i <= k_; i++) {
      err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
    }
  }

	return err;
}

double dyson::dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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

    ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
    les_it_int(m, n, G, Sigma, M_.data());
    QBlock += ZMatrixMap(M_.data(), nao_, nao_);
  }

  for(int n = 1; n <= k_; n++) {
    for(int l = 1; l <= k_; l++) {
      auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
      MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
// DEBUG
//      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).conjugate() - mu*IMap;
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
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
      
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
// DEBUG
//        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).conjugate() - mu*IMap;
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
    }
  }

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

  for(int i = 1; i <= k_; i++) {
    auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
// DEBUG
//    QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_).adjoint() - mu*IMap);
    QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_) - mu*IMap);

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
  }

  Eigen::FullPivLU<ZMatrix> lu3(MIC);
  XIC = lu3.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
  }

	return err;
}

double dyson::dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
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

    ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//    les_it_int(m, n, G, Sigma, M_.data());
    QBlock += ZMatrixMap(M_.data(), nao_, nao_);
  }

  for(int n = 1; n <= k_; n++) {
    for(int l = 1; l <= k_; l++) {
      auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
      MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
// DEBUG
//      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).conjugate() - mu*IMap;
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
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
      
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
//      les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
// DEBUG
//        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).conjugate() - mu*IMap;
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
    Eigen::FullPivLU<ZMatrix> lu2(MIC);
    XIC = lu2.solve(QIC);
    for(int i = m; i <= k_; i++) {
      err += i==m ? 0 : (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
    }
  }

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

  for(int i = 1; i <= k_; i++) {
    auto QBlock = QIC.block((i-1)*nao_, 0, nao_, nao_);
// DEBUG
//    QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_).adjoint() - mu*IMap);
    QBlock += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_) - mu*IMap);

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
  }

  Eigen::FullPivLU<ZMatrix> lu3(MIC);
  XIC = lu3.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(XIC.data()+(i-1)*es_, nao_, nao_) - ZMatrixMap(DIC.data() + (i-1)*es_, nao_, nao_)).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
  }

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








double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    assert(xi_==G.sig() && nao_==G.size1() && nao_==G.size2());
    double sig = double(xi_);

    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H.get_map(-1));
      DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DNTauTmp_.data(),G.GMConvTens());
    }

    double *tmp = reinterpret_cast<double*>(Sigma.tvptr(0,0));
    DMatrixMap(tmp, es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DNTauTmp_.data(), G.GMConvTens(), tmp, DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();

    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return ret;
}


double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, DMatrix &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    assert(xi_==G.sig() && nao_==G.size1() && nao_==G.size2());
    double sig = double(xi_);

    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H);
      DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DNTauTmp_.data(),G.GMConvTens());
    }

    double *tmp = reinterpret_cast<double*>(Sigma.tvptr(0,0));
    DMatrixMap(tmp, es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DNTauTmp_.data(), G.GMConvTens(), tmp, DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();
    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return ret;
}

} // namespace hodlr
