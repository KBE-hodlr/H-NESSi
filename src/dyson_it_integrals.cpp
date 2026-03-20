#include "h_nessi/dyson.hpp"
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

void dyson::gamma_integral(int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res) {
  memset(res, 0, nao_*nao_*sizeof(cplx));
  ZMatrixMap resMap(res, nao_, nao_);
  cplx cplxi = cplx(0.,1.);

  for(int t = 0; t <= tstp; t++) {
    resMap += -cplxi * h * I.gregory_weights(tstp,t) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, t), nao_, nao_) * ZMatrixMap(Delta.curr_timestep_les_ptr(t, tstp), nao_, nao_);
  }
  for(int t = 0; t <= tstp; t++) {
    resMap -= -cplxi * h * I.gregory_weights(tstp, t) * ZMatrixMap(G.curr_timestep_les_ptr(t, tstp), nao_, nao_).adjoint() * ZMatrixMap(Delta.curr_timestep_ret_ptr(tstp, t), nao_, nao_).adjoint();
  }

  ZMatrixMap(NTauTmp_.data(), r_, es_) = DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * ZMatrixMap(Delta.tvptr(tstp,0), r_, es_);
  ZMatrixMap(NTauTmp2_.data(), r_, es_) = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);
  for(int tau = 0; tau < r_; tau++) {
    resMap += -ZMatrixMap(G.tvptr(tstp, tau), nao_, nao_) * ZMatrixMap(NTauTmp2_.data() + tau*es_, nao_, nao_).adjoint();
  }

  for(int t = tstp+1; t <= k_; t++) {
    resMap += -cplxi * h * I.gregory_weights(tstp, t) * ZMatrixMap(G.curr_timestep_ret_ptr(t, tstp), nao_, nao_).adjoint() * ZMatrixMap(Delta.curr_timestep_les_ptr(tstp, t), nao_, nao_).adjoint();
  }
  for(int t = tstp+1; t <= k_; t++) {
    resMap -= -cplxi * h * I.gregory_weights(tstp, t) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, t), nao_, nao_) * ZMatrixMap(Delta.curr_timestep_ret_ptr(t, tstp), nao_, nao_);
  }
}

void dyson::gamma_integral_mat(herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res) {

  memset(res, 0, nao_*nao_*sizeof(cplx));
  ZMatrixMap resMap(res, nao_, nao_);
  cplx cplxi = cplx(0.,1.);

  ZMatrixMap(NTauTmp_.data(), r_, es_) = DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * DMatrixMap(G.matptr(0), r_, es_);
  ZMatrixMap(NTauTmp2_.data(), r_, es_) = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);
  for(int tau = 0; tau < r_; tau++) {
    resMap += ZMatrixMap(NTauTmp2_.data() + tau*nao_*nao_, nao_, nao_) * DMatrixMap(Delta.matptr(tau), nao_, nao_).adjoint();
  }
}



// C_{ajbk} \Sigma_{bik} = I_{aij}
// Sigma is assumed to be transposed, ie \Sigma^\lceil_{ik} is stored as \Sigma^\lceil_{ki}
void dyson::tv_it_conv(cplx *Sigma, cplx *res, double *GMConvTens) {
  // I_{aji} = C_{ajbk} Sigma_{bki}
  ZMatrixMap(NTauTmp2_.data(), r_*nao_, nao_).noalias() = DMatrixMap(GMConvTens, r_*nao_, r_*nao_) * ZMatrixMap(Sigma, r_*nao_, nao_);

  // I_{aji} -> I_{aij}
  // For some stupid reason Eigen does not allow you to call transposeInPlace() on a Map
  for(int i = 0; i<r_; i++) {
    ZMatrixMap(res+i*es_, nao_, nao_) += ZMatrixMap(NTauTmp2_.data()+i*es_, nao_,nao_).transpose();
  }
}

void dyson::tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res) {
  dyson::tv_it_conv(Sigma.tvptr_trans(tstp,0), res, G.GMConvTens());
}

// C_{mjbk} \Sigma_{bik} = I_{ij}
// Sigma is assumed to be transposed, ie \Sigma^\lceil_{ik} is stored as \Sigma^\lceil_{ki}
void dyson::tv_it_conv(int m, cplx *Sigma, cplx *res,double *GMConvTens) {
  // I_{ij} = C_{jbk} Sigma_{bki}
  ZMatrixMap(res, nao_, nao_).noalias() += (DMatrixMap(GMConvTens + m*nao_*nao_*r_, nao_, r_*nao_) * ZMatrixMap(Sigma, r_*nao_, nao_)).transpose();
}

void dyson::tv_it_conv(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res) {
  dyson::tv_it_conv(m, Sigma.tvptr_trans(tstp,0), res, G.GMConvTens());
}

// I_{tij} = -i \int_0^beta d\tau \Sigma^\lceil_{ik}(t,\tau) G^\rceil_{kj}(\tau,T)
//         \approx -i -xi \Sigma_{taik}M_{ab} G_{(Ntau-b)kj}^*
// Sigma is assumed to be transposed, ie \Sigma^\lceil_{ik} is stored as \Sigma^\lceil_{ki}
// Result goes into res VIA INCREMENT
// Integral is transposed in ij indicies
void dyson::les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res) {
  assert(tstp <= G.nt());
  assert(tstp <= Sigma.nt());
  assert(tstp <= nt_);
  assert(G.size1() == Sigma.size1());
  assert(G.size1() == nao_);
  assert(G.sig() == xi_);
  assert(G.sig() == Sigma.sig());
  assert(G.r() == Sigma.r());
  assert(G.r() == r_);

  cplx mimxi = cplx(0, xi_);

  // G^\lceil_{kj}(\tau, T) = -xi G^\rceil_{jk}(\beta-\tau, T)*
  ZMatrixMap(NTauTmp_.data(), r_, es_).noalias() = mimxi * DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * ZMatrixMap(G.tvptr_trans(tstp,0), r_, es_).conjugate();

  // X_{akj} = M_{ab} G^\lceil_{kj}(a, T)
  ZMatrixMap(NTauTmp2_.data(), r_, es_).noalias() = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);

  for( int t = 0; t <= tstp; t++) {
    // res_{ij} = ST_{aki} X_{akj}
    ZMatrixMap(res + t*es_, nao_, nao_).noalias() += ZMatrixMap(NTauTmp2_.data(), r_*nao_, nao_).transpose() * ZMatrixMap(Sigma.tvptr_trans(t,0), r_*nao_, nao_);
  }
}

// I_{tij} = -i \int_0^beta d\tau \Sigma^\lceil_{ik}(t,\tau) G^\rceil_{kj}(\tau,T)
//         \approx -i -xi \Sigma_{taik}M_{ab} G_{(Ntau-b)kj}^*
// Sigma is assumed to be transposed, ie \Sigma^\lceil_{ik} is stored as \Sigma^\lceil_{ki}
// Result goes into res VIA INCREMENT
// Integral is transposed in ij indicies
void dyson::les_it_int(int t, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res) {
  assert(tstp <= G.nt());
  assert(tstp <= Sigma.nt());
  assert(tstp <= nt_);
  assert(G.size1() == Sigma.size1());
  assert(G.size1() == nao_);
  assert(G.sig() == xi_);
  assert(G.sig() == Sigma.sig());
  assert(G.r() == Sigma.r());
  assert(G.r() == r_);

  cplx mimxi = cplx(0, xi_);

  // G^\lceil_{kj}(\tau, T) = -xi G^\rceil_{jk}(\beta-\tau, T)*
  ZMatrixMap(NTauTmp_.data(), r_, es_) = mimxi * DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * ZMatrixMap(G.tvptr_trans(tstp,0), r_, es_).conjugate();

  // X_{akj} = M_{ab} G^\lceil_{kj}(a, T)
  ZMatrixMap(NTauTmp2_.data(), r_, es_) = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);

  // res_{ij} = S_{aki} X_{akj}
  ZMatrixMap(res, nao_, nao_) += ZMatrixMap(NTauTmp2_.data(), r_*nao_, nao_).transpose() * ZMatrixMap(Sigma.tvptr_trans(t,0), r_*nao_, nao_);
}

void dyson::tv_ret_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, Integration::Integrator &I, double h) {
  memset(NTauTmp2_.data(),0,r_*es_*sizeof(cplx));

  for( int t = 0; t < tstp; t++ ) {
    ZMatrixMap(NTauTmp_.data(), es_, r_).noalias() = (h * I.gregory_weights(tstp,t)) * ZMatrixMap(G.tvptr(t,0), r_, es_).transpose();
    ZMatrixMap(NTauTmp2_.data(), nao_, nao_*r_) += ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, t), nao_, nao_) * ZMatrixMap(NTauTmp_.data(), nao_, nao_*r_);
  }

  std::memcpy(NTauTmp_.data(), G.tvptr(tstp,0), r_*es_*sizeof(cplx));
  ZMatrixMap(G.tvptr(tstp,0), r_, es_).noalias() = ZMatrixMap(NTauTmp2_.data(), es_, r_).transpose();
}

// Integral is transposed in ij indicies
void dyson::les_lesadv_int_0_tstp(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, double h, Integration::Integrator &I) {

  ZMatrixMap(X_.data(), nao_, (tstp+1)*nao_).noalias() = h * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, 0), (tstp+1)*nao_, nao_).adjoint();
  ZMatrixMap(M_.data(), nao_, (k_+1)*nao_).noalias() = h * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, 0), (k_+1)*nao_, nao_).adjoint();

  std::chrono::time_point<std::chrono::system_clock> intstart, intend;
  std::chrono::duration<double, std::micro> dir, dircorr, blocks, blockstrans, blockscorrhor, blockscorrvert, currvert, currhor;

  cplx *resptr = Q_.data();

  // 1 direct
  // 2 direct corrections
  // 3 blocks
  // 4 blocks transpose
  // 5 blocks corrections horiz
  // 6 blocks corrections vert
  // 7 curr timestep vert
  // 8 curr timestep horiz

  // 1 This part of the integral does the triangular regions that are directly stored
  intstart = std::chrono::system_clock::now();
  int b = 0;
  for(b = 0; b < G.nbox() && G.blkr1(b) <= G.tstpmk(); b++) {
    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + Sigma.blkr1(b-1)*es_ + i, Sigma.blkdirheight(b), nao_, Eigen::InnerStride<>(nao_)).noalias() += h *
        ZMatrixMap(Sigma.les_dir_square().data() + i*nao_*Sigma.len_les_dir_square() + k*Sigma.len_les_dir_square() + Sigma.les_dir_square_first_index(b), Sigma.blkdirheight(b), Sigma.blkdirheight(b)).transpose() * 
        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(G.curr_timestep_ret_ptr(tstp, Sigma.blkr1(b-1))+k, Sigma.blkdirheight(b), nao_, Eigen::InnerStride<>(nao_)).conjugate();
      }
    }
  }
  for(int i = 0; i < nao_; i++) {
    for(int k = 0; k < nao_; k++) {
      Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + Sigma.blkr1(b-1)*es_ + i, Sigma.tstpmk()-Sigma.blkr1(b-1), nao_, Eigen::InnerStride<>(nao_)).noalias() += h *
      ZMatrixMap(Sigma.les_dir_square().data() + i*nao_*Sigma.len_les_dir_square() + k*Sigma.len_les_dir_square() + Sigma.les_dir_square_first_index(b), Sigma.blkdirheight(b), Sigma.blkdirheight(b)).block(0,0,Sigma.tstpmk()-Sigma.blkr1(b-1),Sigma.tstpmk()-Sigma.blkr1(b-1)).transpose() * 
      Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(G.curr_timestep_ret_ptr(tstp, Sigma.blkr1(b-1))+k, Sigma.tstpmk()-Sigma.blkr1(b-1), nao_, Eigen::InnerStride<>(nao_)).conjugate();
    }
  }
  intend = std::chrono::system_clock::now();
  dir = intend-intstart;

  // 2 corrections for parts of integral in directly stored regions
  // corrections for this only happen in upper right corner.  
  // Most of the time this means we correct all rows up to the end of the directly stored region (G.blkr1(0)-1)
  // Edge case where height of upper right square is smaller than k_
  // Both of these cases are covered in G.c1_dir(t) > k_
  // Also stop if you are about to hit curr_timestep tensor
  intstart = std::chrono::system_clock::now();
  for(int t = 0; t <= tstp; t++) {
    if(t == G.tstpmk() || G.c1_dir(t) > k_) break;
    auto corrMap = ZMatrixMap(resptr + t*es_, nao_, nao_);
    // corrmin is just the first column of directly stored information on a given row
    int corrmin = G.c1_dir(t);
    // corrmax is normally just k_, but edge case where tstpmk is closer than that
    int corrmax = std::min(k_, G.tstpmk()-1);
    // this is the case where upper dir block is smaller than k and the last few corrections are taken care of in vert block corrections
    corrmax = std::min(corrmax, G.r2_dir(t));
    for(int tbar = corrmin; tbar <= corrmax; tbar++) {
      int blk = G.t_to_dirlvl(t);
      int col = G.ntri(t)-1;
      int row = tbar-corrmin;
      corrMap.noalias() += h * (-1. + I.gregory_weights(tstp, tbar)) * 
                  ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tbar), nao_, nao_).conjugate() * 
                  Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Sigma.les_dir_square().data() + G.les_dir_square_first_index(blk) + row*G.blkdirheight(blk) + col, nao_, nao_, Eigen::InnerStride<>(G.len_les_dir_square())).transpose();
    }    
  }
  intend = std::chrono::system_clock::now();
  dircorr = intend-intstart;

  // Step 3 is to apply the blocks
  intstart = std::chrono::system_clock::now();
  auto startR = high_resolution_clock::now();
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    for( int i = 0; i < nao_; i++) {
      for( int k = 0; k < nao_; k++) {
        int blkrows = Sigma.blklen(b);
        int blkcols = Sigma.les().data().blocks()[b][i][k].cols();
        int epsrank = Sigma.les().data().blocks()[b][i][k].epsrank();
        int blkr1 = Sigma.blkr1(b);
        ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);
        ZMatrixMap tmp2(epsnao_tmp_2_.data(), epsrank, nao_);

        tmp.noalias() = ZMatrixMap(Sigma.les().data().blocks()[b][i][k].Udata(), blkrows, epsrank).transpose() * ZMatrixMap(X_.data() + k*(tstp+1)*nao_ + blkr1*nao_, blkrows, nao_);

        tmp2.noalias() = DColVectorMap(Sigma.les().data().blocks()[b][i][k].Sdata(), epsrank).asDiagonal() * tmp;

        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + Sigma.blkc1(b)*es_ + i, blkcols, nao_, Eigen::InnerStride<>(nao_)).noalias() += ZMatrixMap(Sigma.les().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).conjugate() * tmp2;
      }
    }
  }
  auto endR = high_resolution_clock::now();
  auto duration=duration_cast<microseconds>(endR - startR);
  intend = std::chrono::system_clock::now();
  blocks = intend-intstart;

  // Step 4 is to apply the transposed blocks
  intstart = std::chrono::system_clock::now();
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    for( int k = 0; k < nao_; k++) {
      for( int i = 0; i < nao_; i++) {
        int blkrows = Sigma.blklen(b);
        int blkcols = Sigma.les().data().blocks()[b][k][i].cols();
        int epsrank = Sigma.les().data().blocks()[b][k][i].epsrank();
        int blkr1 = Sigma.blkr1(b);
        ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);
        ZMatrixMap tmp2(epsnao_tmp_2_.data(), epsrank, nao_);

        tmp.noalias() = ZMatrixMap(Sigma.les().data().blocks()[b][k][i].Vdata(), blkcols, epsrank).transpose() * ZMatrixMap(X_.data() + k*(tstp+1)*nao_ + Sigma.blkc1(b)*nao_, blkcols, nao_);

        tmp2.noalias() = DColVectorMap(Sigma.les().data().blocks()[b][k][i].Sdata(), epsrank).asDiagonal() * tmp;

        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)).noalias() -= ZMatrixMap(Sigma.les().data().blocks()[b][k][i].Udata(), blkrows, epsrank).conjugate() * tmp2;
      }
    }
  }
  intend = std::chrono::system_clock::now();
  blockstrans = intend-intstart;

  intstart = std::chrono::system_clock::now();
  // Step 5 is horizontal corrections
  if( tstp < 2*k_+1) { // les_left_edge has omega, but for these timesteps we need the weights that overlap.
    for(int t = Sigma.blkr1(0); t < Sigma.tstpmk(); t++) {
      int max = std::min(k_, Sigma.c1_dir(t)-1);
      for(int tbar = 0; tbar <= max; tbar++) {
        Sigma.get_les(tbar, t, bound_.data());
        ZMatrixMap(resptr + t*es_, nao_, nao_) -= h * (-1. + I.gregory_weights(tstp,tbar)) * (ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tbar), nao_, nao_) * ZMatrixMap(bound_.data(), nao_, nao_)).conjugate();   
      }
    }
  }
  else {
    if( (G.tstpmk() - (G.r2_dir(0)+1)) > 0) {
      for(int i = 0; i < nao_; i++) {
        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + (G.r2_dir(0)+1)*es_ + i, G.tstpmk() - (G.r2_dir(0)+1), nao_, Eigen::InnerStride<>(nao_)).noalias()
        += ZMatrixMap(Sigma.les_left_edge().data() + i*nao_*(k_+1)*Sigma.nt() + (G.r2_dir(0)+1)*nao_*(k_+1), G.tstpmk() - (G.r2_dir(0)+1), nao_*(k_+1))
          *ZMatrixMap(M_.data(), nao_*(k_+1), nao_);
      }
    }
  }
  intend = std::chrono::system_clock::now();
  blockscorrhor = intend-intstart;

  // Step 6 is vertical corrections
  intstart = std::chrono::system_clock::now();
  for(int t = 0; t <= k_+1; t++) {
    // if correction region runs out of triangular region
    // need to make sure not to run into curr_timestep
    // This is only needed if the size of directly stored triangles is smaller than k+1
    // corrections only occur to integrals for t smaller than k+1
    int correction_max = std::min(k_, Sigma.tstpmk()-1);
    for(int tbar = Sigma.r2_dir(t)+1; tbar <= correction_max; tbar++) {
      Sigma.get_les(t,tbar,bound_.data());
      ZMatrixMap(resptr + t*es_, nao_, nao_) += h * (I.gregory_weights(tstp, tbar)-1.) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tbar), nao_, nao_).conjugate() * ZMatrixMap(bound_.data(), nao_, nao_).transpose();
    }
  }
  intend = std::chrono::system_clock::now();
  blockscorrvert = intend-intstart;

  // Step 7 is to apply the curr_timestep region vertical
  intstart = std::chrono::system_clock::now();
  for(int t = 0; t <= tstp; t++) {
    for(int tbar = std::max(Sigma.tstpmk(), t); tbar <= tstp; tbar++) {
      ZMatrixMap(resptr + t*es_, nao_, nao_) += h * I.gregory_weights(tstp, tbar) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tbar), nao_, nao_).conjugate() * ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tbar), nao_, nao_).transpose();
    }
  }
  intend = std::chrono::system_clock::now();
  currvert = intend-intstart;

  // Step 8 is curr_timestep horizontal
  intstart = std::chrono::system_clock::now();
  for(int t = Sigma.tstpmk(); t <= tstp; t++) {
    // the diagonal element is at tbar=t.  This element is done in Step 7, where tbar = std::max(Sigma.tstpmk(), t) = t
    // this is why the loop ends at tbar < t instead of tbar <= t
    for(int tbar = 0; tbar < t; tbar++) {
      ZMatrixMap(resptr + t*es_, nao_, nao_) -= h * I.gregory_weights(tstp, tbar) * (ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tbar), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(tbar, t), nao_, nao_)).conjugate();
    }
  }
  intend = std::chrono::system_clock::now();
  currhor = intend-intstart;

  if(profile_) timing(tstp, 13) = dir.count();
  if(profile_) timing(tstp, 14) = dircorr.count();
  if(profile_) timing(tstp, 15) = blocks.count();
  if(profile_) timing(tstp, 16) = blockstrans.count();
  if(profile_) timing(tstp, 17) = blockscorrhor.count();
  if(profile_) timing(tstp, 18) = blockscorrvert.count();
  if(profile_) timing(tstp, 19) = currvert.count();
  if(profile_) timing(tstp, 20) = currhor.count();

}
} // namespace
