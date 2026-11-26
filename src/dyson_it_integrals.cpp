#include "h_nessi/dyson.hpp"
#include <iomanip>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std::chrono;

namespace h_nessi {

namespace {
// Row-major Eigen Tensor views of the library's row-major buffers (index order = storage order)
template <int N> using ZTensorMap = Eigen::TensorMap<Eigen::Tensor<cplx, N, Eigen::RowMajor>>;
template <int N> using ZTensorConstMap = Eigen::TensorMap<const Eigen::Tensor<cplx, N, Eigen::RowMajor>>;
template <int N> using DTensorConstMap = Eigen::TensorMap<const Eigen::Tensor<double, N, Eigen::RowMajor>>;
template <int N> using IndexPairs = Eigen::array<Eigen::IndexPair<int>, N>;
}  // namespace

// DY-G1, DY-G3 (tier 2): G and Delta share the geometry and match this dyson
void dyson::check_gamma_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta) {
  HNESSI_REQUIRE(G.size1() == nao_ && G.size2() == nao_ && G.r() == r_ && G.sig() == xi_, where, HNESSI_KV(G.size1()),
                 HNESSI_KV(nao_), HNESSI_KV(G.r()), HNESSI_KV(r_), HNESSI_KV(G.sig()), HNESSI_KV(xi_));
  HNESSI_REQUIRE(Delta.nt() == G.nt() && Delta.nlvl() == G.nlvl() && Delta.k() == G.k() &&
                 Delta.size1() == G.size1() && Delta.size2() == G.size2() && Delta.r() == G.r() &&
                 Delta.sig() == G.sig(), where, HNESSI_KV(Delta.nt()), HNESSI_KV(G.nt()), HNESSI_KV(Delta.k()),
                 HNESSI_KV(G.k()), HNESSI_KV(Delta.size1()), HNESSI_KV(G.size1()), HNESSI_KV(Delta.r()),
                 HNESSI_KV(G.r()), HNESSI_KV(Delta.sig()), HNESSI_KV(G.sig()),
                 detail::note{"G and Delta must have the same geometry"});
}

void dyson::gamma_integral(int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res) {
  call_scope cs(*this);
  if(cs.outer()) {
    const char *where = "dyson::gamma_integral";
    check_gamma_args_(where, G, Delta);   // DY-G1
    HNESSI_REQUIRE(I.k() == k_ && G.k() == k_ && std::isfinite(h) && h > 0, where, HNESSI_KV(I.k()), HNESSI_KV(G.k()),
                   HNESSI_KV(k_), HNESSI_KV(h));
    // DY-G2: row tstp of both objects is in their windows; for tstp < k the rows tstp+1..k are read as well
    HNESSI_REQUIRE(0 <= tstp && tstp < G.nt(), where, HNESSI_KV(tstp), HNESSI_KV(G.nt()));
    HNESSI_REQUIRE_STATE(G.tstpmk() <= tstp && tstp <= G.tstpmk()+k_ && Delta.tstpmk() <= tstp &&
                         tstp <= Delta.tstpmk()+k_ && (tstp >= k_ || (G.tstpmk() == 0 && Delta.tstpmk() == 0)),
                         where, HNESSI_KV(tstp), HNESSI_KV(G.tstpmk()), HNESSI_KV(Delta.tstpmk()), HNESSI_KV(k_),
                         detail::note{"tstp must be in the current windows of G and Delta (for tstp < k: the bootstrap "
                                      "windows 0..k)"});
  }
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
    // mixed term -i(-i\int G^\rceil \Delta^\lceil) = \xi \int G^\rceil(T,\tau) [\Delta^\rceil(T,\beta-\tau)]^\dagger
    resMap += double(xi_) * ZMatrixMap(G.tvptr(tstp, tau), nao_, nao_) * ZMatrixMap(NTauTmp2_.data() + tau*es_, nao_, nao_).adjoint();
  }

  for(int t = tstp+1; t <= k_; t++) {
    resMap += -cplxi * h * I.gregory_weights(tstp, t) * ZMatrixMap(G.curr_timestep_ret_ptr(t, tstp), nao_, nao_).adjoint() * ZMatrixMap(Delta.curr_timestep_les_ptr(tstp, t), nao_, nao_).adjoint();
  }
  for(int t = tstp+1; t <= k_; t++) {
    resMap -= -cplxi * h * I.gregory_weights(tstp, t) * ZMatrixMap(G.curr_timestep_les_ptr(tstp, t), nao_, nao_) * ZMatrixMap(Delta.curr_timestep_ret_ptr(t, tstp), nao_, nao_);
  }
}

void dyson::gamma_integral_mat(herm_matrix_hodlr &G, herm_matrix_hodlr &Delta, double h, Integration::Integrator &I, cplx *res) {
  call_scope cs(*this);
  if(cs.outer()) check_gamma_args_("dyson::gamma_integral_mat", G, Delta);   // DY-G3

  memset(res, 0, nao_*nao_*sizeof(cplx));
  ZMatrixMap resMap(res, nao_, nao_);
  cplx cplxi = cplx(0.,1.);

  ZMatrixMap(NTauTmp_.data(), r_, es_) = DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * DMatrixMap(G.matptr(0), r_, es_);
  ZMatrixMap(NTauTmp2_.data(), r_, es_) = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);
  for(int tau = 0; tau < r_; tau++) {
    resMap += ZMatrixMap(NTauTmp2_.data() + tau*nao_*nao_, nao_, nao_) * DMatrixMap(Delta.matptr(tau), nao_, nao_).adjoint();
  }
}



// The imaginary-time integrals below read Sigma^tv and G^tv in their storage layout tv[t][a][i][j] = X^tv_{ij}(t,tau_a)
// (doc 11 §5.5). The orbital transposes they need are absorbed by Eigen Tensor contractions over (DLR index, orbital)
// pairs; there is no transposed copy of tv (WP6c, ISSUES H27). One contraction per call, default (single-thread)
// device: these functions run inside OpenMP regions (k-points), and Eigen's ThreadPoolDevice is 20-150x slower for these
// shapes (doc 62).

// DY-I2 + ST-2 + dlr pairing (tier 2) for the public tv_it_conv/les_it_int entry points
void dyson::check_it_args_(const char *where, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G) {
  HNESSI_REQUIRE(G.size1() == nao_ && G.size2() == nao_ && Sigma.size1() == nao_ && Sigma.size2() == nao_, where,
                 HNESSI_KV(G.size1()), HNESSI_KV(Sigma.size1()), HNESSI_KV(nao_));
  HNESSI_REQUIRE(G.r() == r_ && Sigma.r() == r_ && G.sig() == xi_ && Sigma.sig() == xi_, where, HNESSI_KV(G.r()),
                 HNESSI_KV(Sigma.r()), HNESSI_KV(r_), HNESSI_KV(G.sig()), HNESSI_KV(Sigma.sig()), HNESSI_KV(xi_),
                 detail::note{"G and Sigma must match the dlr_info of this dyson"});
  HNESSI_REQUIRE(0 <= tstp && tstp < Sigma.nt() && tstp < G.nt(), where, HNESSI_KV(tstp), HNESSI_KV(Sigma.nt()),
                 HNESSI_KV(G.nt()));
}

// I_{aij} = sum_{b,k} C_{ajbk} Sigma^tv_{ik}(T,tau_b)   (C = G.GMConvTens(), r x n x r x n; doc 11 §5.6)
// Result goes into res VIA INCREMENT, res[a][i][j] (r x n x n)
void dyson::tv_it_conv(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res) {
  call_scope cs(*this);
  if(cs.outer()) {
    check_it_args_("dyson::tv_it_conv", tstp, Sigma, G);
    G.require_gmconv_current("dyson::tv_it_conv");   // ST-2
  }
  DTensorConstMap<4> C(G.GMConvTens(), r_, nao_, r_, nao_);   // C(a,j,b,k)
  ZTensorConstMap<3> S(Sigma.tvptr(tstp,0), r_, nao_, nao_);   // S(b,i,k)
  ZTensorMap<3> R(res, r_, nao_, nao_);                        // R(a,i,j)
  // C(a,j,b,k) S(b,i,k) -> (a,j,i) -> (a,i,j)
  R += C.cast<cplx>().contract(S, IndexPairs<2>{{{2,0}, {3,2}}}).shuffle(Eigen::array<int,3>{{0,2,1}});
}

// Single DLR node m: I_{ij} = sum_{b,k} C_{mjbk} Sigma^tv_{ik}(T,tau_b); res (n x n) VIA INCREMENT
void dyson::tv_it_conv(int m, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx *res) {
  call_scope cs(*this);
  if(cs.outer()) {
    check_it_args_("dyson::tv_it_conv", tstp, Sigma, G);
    HNESSI_REQUIRE(0 <= m && m < r_, "dyson::tv_it_conv", HNESSI_KV(m), HNESSI_KV(r_));
    // ST-2: only the O(1) validity flag here; hashing G^M (O(r*es)) would cost as much as this single-node call
    G.require_gmconv_current("dyson::tv_it_conv", false);
  }
  HNESSI_ASSERT(0 <= m && m < r_, "dyson::tv_it_conv", HNESSI_KV(m), HNESSI_KV(r_));   // DY-I1 (tier 1)
  DTensorConstMap<3> Cm(G.GMConvTens() + m*nao_*r_*nao_, nao_, r_, nao_);   // C(m,j,b,k)
  ZTensorConstMap<3> S(Sigma.tvptr(tstp,0), r_, nao_, nao_);              // S(b,i,k)
  ZTensorMap<2> R(res, nao_, nao_);                                       // R(i,j)
  // S(b,i,k) C(m,j,b,k) -> (i,j)
  R += S.contract(Cm.cast<cplx>(), IndexPairs<2>{{{0,1}, {2,2}}});
}

// X_{aik} = sum_b ipmat_{ba} (-i) G^lceil_{ki}(tau_b,T), with G^lceil_{ki}(tau,T) = -xi [G^rceil_{ik}(T,beta-tau)]^*
// (the prelude of both les_it_int overloads; X is stored in NTauTmp2_ as r x n x n)
void dyson::les_it_int_prelude_(int tstp, herm_matrix_hodlr &G) {
  cplx mimxi = cplx(0, xi_);
  // NTauTmp[b][i][k] = i xi conj(G^rceil_{ik}(T, beta-tau_b)) = -i G^lceil_{ki}(tau_b, T)
  ZMatrixMap(NTauTmp_.data(), r_, es_).noalias() = mimxi * DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * ZMatrixMap(G.tvptr(tstp,0), r_, es_).conjugate();
  ZMatrixMap(NTauTmp2_.data(), r_, es_).noalias() = DMatrixConstMap(dlr_.ipmat(), r_, r_).transpose() * ZMatrixMap(NTauTmp_.data(), r_, es_);
}

// I_{ji}(t) = -i \int_0^beta d\tau \Sigma^\rceil_{jk}(t,\tau) G^\lceil_{ki}(\tau,T)  for t = 0..tstp
//           = sum_{a,k} Sigma^tv_{jk}(t,tau_a) X_{aik}
// Result goes into res VIA INCREMENT, res[t][i][j] = I_{ji}(t): the integral is transposed in ij (doc 03 §4.10)
void dyson::les_it_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res) {
  call_scope cs(*this);
  if(cs.outer()) check_it_args_("dyson::les_it_int", tstp, Sigma, G);   // DY-I3
  // LX-15 (tier 1; the tier-2 checks run at the public entry)
  HNESSI_ASSERT(0 <= tstp && tstp < G.nt() && tstp < Sigma.nt() && tstp < nt_ && G.size1() == Sigma.size1() &&
                G.size1() == nao_ && G.sig() == xi_ && G.sig() == Sigma.sig() && G.r() == Sigma.r() && G.r() == r_,
                "dyson::les_it_int", HNESSI_KV(tstp), HNESSI_KV(G.nt()), HNESSI_KV(Sigma.nt()), HNESSI_KV(nt_));

  les_it_int_prelude_(tstp, G);
  ZTensorConstMap<4> S(Sigma.tvptr(0,0), tstp+1, r_, nao_, nao_);   // S(t,a,j,k), rows 0..tstp
  ZTensorConstMap<3> X(NTauTmp2_.data(), r_, nao_, nao_);            // X(a,i,k)
  ZTensorMap<3> R(res, tstp+1, nao_, nao_);                          // R(t,i,j)
  // one contraction over all t: S(t,a,j,k) X(a,i,k) -> (t,j,i) -> (t,i,j)
  R += S.contract(X, IndexPairs<2>{{{1,0}, {3,2}}}).shuffle(Eigen::array<int,3>{{0,2,1}});
}

// Single row t of the above: res[i][j] += I_{ji}(t) (n x n, transposed in ij)
void dyson::les_it_int(int t, int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, cplx* res) {
  call_scope cs(*this);
  if(cs.outer()) {   // DY-I4
    check_it_args_("dyson::les_it_int", tstp, Sigma, G);
    HNESSI_REQUIRE(0 <= t && t < Sigma.nt(), "dyson::les_it_int", HNESSI_KV(t), HNESSI_KV(Sigma.nt()));
  }
  // LX-15 (tier 1; the tier-2 checks run at the public entry)
  HNESSI_ASSERT(0 <= tstp && tstp < G.nt() && tstp < Sigma.nt() && tstp < nt_ && G.size1() == Sigma.size1() &&
                G.size1() == nao_ && G.sig() == xi_ && G.sig() == Sigma.sig() && G.r() == Sigma.r() && G.r() == r_,
                "dyson::les_it_int", HNESSI_KV(tstp), HNESSI_KV(G.nt()), HNESSI_KV(Sigma.nt()), HNESSI_KV(nt_));

  les_it_int_prelude_(tstp, G);
  ZTensorConstMap<3> X(NTauTmp2_.data(), r_, nao_, nao_);   // X(a,i,k)
  ZTensorConstMap<3> S(Sigma.tvptr(t,0), r_, nao_, nao_);   // S(a,j,k)
  ZTensorMap<2> R(res, nao_, nao_);                         // R(i,j)
  // X(a,i,k) S(a,j,k) -> (i,j)
  R += X.contract(S, IndexPairs<2>{{{0,0}, {2,2}}});
}

void dyson::tv_ret_int(int tstp, herm_matrix_hodlr &Sigma, herm_matrix_hodlr &G, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  if(cs.outer()) {   // DY-I5 (tier 2)
    const char *where = "dyson::tv_ret_int";
    check_it_args_(where, tstp, Sigma, G);
    HNESSI_REQUIRE(0 < tstp && I.k() == k_ && std::isfinite(h) && h > 0, where, HNESSI_KV(tstp), HNESSI_KV(I.k()),
                   HNESSI_KV(k_), HNESSI_KV(h));
    HNESSI_REQUIRE_STATE(Sigma.tstpmk() <= tstp && tstp <= Sigma.tstpmk()+Sigma.k(), where, HNESSI_KV(tstp),
                         HNESSI_KV(Sigma.tstpmk()), detail::note{"row tstp of Sigma must be in its current window"});
  }
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
  call_scope cs(*this);
  if(cs.outer()) {   // DY-I6 (tier 2): G and Sigma geometry and windows are mixed in this integral (doc 21 §4)
    const char *where = "dyson::les_lesadv_int_0_tstp";
    check_solver_args_(where, G, Sigma, I, h, 0.);
    HNESSI_REQUIRE_STATE(G.tstpmk() == Sigma.tstpmk() && tstp == G.tstpmk()+k_, where, HNESSI_KV(tstp),
                         HNESSI_KV(G.tstpmk()), HNESSI_KV(Sigma.tstpmk()), HNESSI_KV(k_),
                         detail::note{"G and Sigma must be at the same step and tstp == G.tstpmk()+k"});
  }

  ZMatrixMap(X_.data(), nao_, (tstp+1)*nao_).noalias() = h * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, 0), (tstp+1)*nao_, nao_).adjoint();
  ZMatrixMap(M_.data(), nao_, (k_+1)*nao_).noalias() = h * ZMatrixMap(G.curr_timestep_ret_ptr(tstp, 0), (k_+1)*nao_, nao_).adjoint();

  std::chrono::time_point<std::chrono::steady_clock> intstart, intend;
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
  intstart = std::chrono::steady_clock::now();
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
  intend = std::chrono::steady_clock::now();
  dir = intend-intstart;

  // 2 corrections for parts of integral in directly stored regions
  // corrections for this only happen in upper right corner.  
  // Most of the time this means we correct all rows up to the end of the directly stored region (G.blkr1(0)-1)
  // Edge case where height of upper right square is smaller than k_
  // Both of these cases are covered in G.c1_dir(t) > k_
  // Also stop if you are about to hit curr_timestep tensor
  intstart = std::chrono::steady_clock::now();
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
  intend = std::chrono::steady_clock::now();
  dircorr = intend-intstart;

  // Step 3 is to apply the blocks
  intstart = std::chrono::steady_clock::now();
  auto startR = high_resolution_clock::now();
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    for( int i = 0; i < nao_; i++) {
      for( int k = 0; k < nao_; k++) {
        int blkrows = Sigma.blklen(b);
        int blkcols = Sigma.les().data().blocks()[b][i][k].cols();
        int epsrank = Sigma.les().data().blocks()[b][i][k].epsrank();
        int blkr1 = Sigma.blkr1(b);
        // DY-T9 (tier 1): the block fits the dyson scratch
        HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), "dyson::les_lesadv_int_0_tstp", HNESSI_KV(epsrank),
                      HNESSI_KV(epsnao_tmp_.size()));
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
  intend = std::chrono::steady_clock::now();
  blocks = intend-intstart;

  // Step 4 is to apply the transposed blocks
  intstart = std::chrono::steady_clock::now();
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    for( int k = 0; k < nao_; k++) {
      for( int i = 0; i < nao_; i++) {
        int blkrows = Sigma.blklen(b);
        int blkcols = Sigma.les().data().blocks()[b][k][i].cols();
        int epsrank = Sigma.les().data().blocks()[b][k][i].epsrank();
        int blkr1 = Sigma.blkr1(b);
        // DY-T9 (tier 1): the block fits the dyson scratch
        HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), "dyson::les_lesadv_int_0_tstp", HNESSI_KV(epsrank),
                      HNESSI_KV(epsnao_tmp_.size()));
        ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);
        ZMatrixMap tmp2(epsnao_tmp_2_.data(), epsrank, nao_);

        tmp.noalias() = ZMatrixMap(Sigma.les().data().blocks()[b][k][i].Vdata(), blkcols, epsrank).transpose() * ZMatrixMap(X_.data() + k*(tstp+1)*nao_ + Sigma.blkc1(b)*nao_, blkcols, nao_);

        tmp2.noalias() = DColVectorMap(Sigma.les().data().blocks()[b][k][i].Sdata(), epsrank).asDiagonal() * tmp;

        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(resptr + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)).noalias() -= ZMatrixMap(Sigma.les().data().blocks()[b][k][i].Udata(), blkrows, epsrank).conjugate() * tmp2;
      }
    }
  }
  intend = std::chrono::steady_clock::now();
  blockstrans = intend-intstart;

  intstart = std::chrono::steady_clock::now();
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
  intend = std::chrono::steady_clock::now();
  blockscorrhor = intend-intstart;

  // Step 6 is vertical corrections
  intstart = std::chrono::steady_clock::now();
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
  intend = std::chrono::steady_clock::now();
  blockscorrvert = intend-intstart;

  // Step 7 is to apply the curr_timestep region vertical
  intstart = std::chrono::steady_clock::now();
  for(int t = 0; t <= tstp; t++) {
    for(int tbar = std::max(Sigma.tstpmk(), t); tbar <= tstp; tbar++) {
      ZMatrixMap(resptr + t*es_, nao_, nao_) += h * I.gregory_weights(tstp, tbar) * ZMatrixMap(G.curr_timestep_ret_ptr(tstp,tbar), nao_, nao_).conjugate() * ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tbar), nao_, nao_).transpose();
    }
  }
  intend = std::chrono::steady_clock::now();
  currvert = intend-intstart;

  // Step 8 is curr_timestep horizontal
  intstart = std::chrono::steady_clock::now();
  for(int t = Sigma.tstpmk(); t <= tstp; t++) {
    // the diagonal element is at tbar=t.  This element is done in Step 7, where tbar = std::max(Sigma.tstpmk(), t) = t
    // this is why the loop ends at tbar < t instead of tbar <= t
    for(int tbar = 0; tbar < t; tbar++) {
      ZMatrixMap(resptr + t*es_, nao_, nao_) -= h * I.gregory_weights(tstp, tbar) * (ZMatrixMap(G.curr_timestep_ret_ptr(tstp, tbar), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(tbar, t), nao_, nao_)).conjugate();
    }
  }
  intend = std::chrono::steady_clock::now();
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
