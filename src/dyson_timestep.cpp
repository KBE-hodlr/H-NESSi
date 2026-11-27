#include "h_nessi/dyson.hpp"
#include <iomanip>
#include <cmath>
#include <cstring>
#include <limits>

using namespace std::chrono;

namespace h_nessi {

namespace {
// DY-T11 (tier 1): a step's linear system is non-singular. A non-finite system is left to the always-on DY-P1 check of
// the returned error (numerical_error), so it is not asserted here.
inline bool lu_ok(const Eigen::FullPivLU<ZMatrix> &lu) { return lu.isInvertible() || !lu.matrixLU().allFinite(); }

// P1: LU with partial (row) pivoting of the n x n row-major A in place: packed L\U with unit L, piv[c] = row
// interchanged with row c at step c (the PartialPivLU algorithm; pivot = largest |a_rc|, r >= c). Hand-written for
// small runtime n: 4-6x faster than Eigen's FullPivLU/PartialPivLU on 1x1..8x8 (agent_scratch/p1/mb). Returns the
// DY-T11 verdict: false if some |u_cc| <= n eps max_j |u_jj| (the FullPivLU rank threshold) and all of A is finite
// (a non-finite system is left to the DY-P1 check of the returned error, as before).
bool small_lu_factor(cplx *A, int *piv, int n) {
  for(int c = 0; c < n; c++) {
    int p = c;
    double best = std::norm(A[c*n+c]);
    for(int r = c+1; r < n; r++) {
      double v = std::norm(A[r*n+c]);
      if(v > best) { best = v; p = r; }
    }
    piv[c] = p;
    if(p != c) for(int j = 0; j < n; j++) std::swap(A[c*n+j], A[p*n+j]);
    if(best == 0) continue;   // singular column: nothing to eliminate (reported below)
    const cplx inv = 1. / A[c*n+c];
    for(int r = c+1; r < n; r++) {
      const cplx f = A[r*n+c] * inv;
      A[r*n+c] = f;
      for(int j = c+1; j < n; j++) A[r*n+j] -= f * A[c*n+j];
    }
  }
  double umax = 0, umin = INFINITY;
  bool finite = true;
  for(int i = 0; i < n*n; i++) finite = finite && std::isfinite(A[i].real()) && std::isfinite(A[i].imag());
  for(int c = 0; c < n; c++) { double u = std::abs(A[c*n+c]); umax = std::max(umax, u); umin = std::min(umin, u); }
  return !finite || umin > n * std::numeric_limits<double>::epsilon() * umax;
}

// P1: B <- A^{-1} B for the factors of small_lu_factor (B n x n row-major)
void small_lu_solve(const cplx *LU, const int *piv, cplx *B, int n) {
  for(int c = 0; c < n; c++)
    if(piv[c] != c) for(int j = 0; j < n; j++) std::swap(B[c*n+j], B[piv[c]*n+j]);
  for(int r = 1; r < n; r++)
    for(int c = 0; c < r; c++) {
      const cplx f = LU[r*n+c];
      for(int j = 0; j < n; j++) B[r*n+j] -= f * B[c*n+j];
    }
  for(int r = n-1; r >= 0; r--) {
    for(int c = r+1; c < n; c++) {
      const cplx f = LU[r*n+c];
      for(int j = 0; j < n; j++) B[r*n+j] -= f * B[c*n+j];
    }
    const cplx inv = 1. / LU[r*n+r];
    for(int j = 0; j < n; j++) B[r*n+j] *= inv;
  }
}
}  // namespace

// P1 (dyson.hpp): X = (M^{-1} op(B))^T with the factors of M cached in slot `slot` (contract in dyson.hpp)
void dyson::lu_solve_cached_(int slot, bool cache, const cplx *M, const cplx *B, bool transB, cplx *X, const char *where) {
  HNESSI_ASSERT(0 <= slot && slot < (int)lu_valid_.size(), where, HNESSI_KV(slot), HNESSI_KV(lu_valid_.size()));
  const int s = cache ? slot : (int)lu_valid_.size();   // uncached: scratch slot 2nt
  cplx *fac = lu_fac_.data() + (size_t)s*es_;
  int *piv = lu_piv_.data() + (size_t)s*nao_;
  if(!cache || !lu_valid_[slot]) {
    std::memcpy(fac, M, es_*sizeof(cplx));
    bool ok = small_lu_factor(fac, piv, nao_);
    (void)ok;
    HNESSI_ASSERT(ok, where, HNESSI_KV(slot), HNESSI_KV(nao_));   // DY-T11 (tier 1)
    if(cache) lu_valid_[slot] = 1;
  } else {
    // DY-T12 (tier 1): the reused factors are those of the newly built M (the archived h(t'), Sigma^R(t',t') were not
    // modified after their time step; dyson.hpp). Uses the scratch slot; not evaluated in Release.
    HNESSI_ASSERT(lu_cache_consistent_(slot, M), where, HNESSI_KV(slot),
                  detail::note{"h(t) or Sigma^R(t,t) of an archived time t was modified after its time step (cached step "
                               "matrix differs); call a dyson_start*/green_from_H* to restart, or do not modify them"});
  }
  if(transB) {
    for(int i = 0; i < nao_; i++) for(int j = 0; j < nao_; j++) X[i*nao_+j] = B[j*nao_+i];
  } else {
    std::memcpy(X, B, es_*sizeof(cplx));
  }
  small_lu_solve(fac, piv, X, nao_);
  for(int i = 0; i < nao_; i++) for(int j = i+1; j < nao_; j++) std::swap(X[i*nao_+j], X[j*nao_+i]);
}

// DY-T12 helper (tier 1 only): factorise M into the scratch slot and compare with the cached factors of `slot`
bool dyson::lu_cache_consistent_(int slot, const cplx *M) {
  const size_t sc = lu_valid_.size();
  cplx *f = lu_fac_.data() + sc*es_;
  int *p = lu_piv_.data() + sc*nao_;
  std::memcpy(f, M, es_*sizeof(cplx));
  small_lu_factor(f, p, nao_);
  return std::memcmp(f, lu_fac_.data() + (size_t)slot*es_, es_*sizeof(cplx)) == 0 &&
         std::memcmp(p, lu_piv_.data() + (size_t)slot*nao_, nao_*sizeof(int)) == 0;
}

double dyson::dyson_timestep_ret(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  call_scope cs(*this);
  const char *where = "dyson::dyson_timestep_ret";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp-1, false);

  // P1: every timer below runs only if profile_ (pnow_() returns a zero time point otherwise)
  clock_tp_ retstart, retend, intstart, intend;
  std::chrono::duration<double, std::micro> retdur(0), intdur(0), intcurrdur(0), intblkdur(0), intcorrdur(0), intdirdur(0), intdircorrdur(0);

  retstart = pnow_();
  zero_lvl_(tstp, 0);
  lu_params_(mu, h, Sigma, H);   // P1: clear the step-matrix cache if mu, h, Sigma or the h array changed


  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  int l, n;
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

  // P1: row tstp of the curr_timestep buffers is contiguous in the second time argument (doc 11 §3.3):
  // G^R(tstp,tp) = Grow + tp*es and Sigma^R(tstp-l,tp) = Srow[l] + tp*es for 0 <= tp <= nt
  cplx *Grow = G.curr_timestep_ret_ptr(tstp, 0);
  cplx *Srow[KMAX+1];
  for(l = 0; l <= k_; l++) Srow[l] = Sigma.curr_timestep_ret_ptr(tstp-l, 0);

  // Initial condition goes into directly stored slice
  ZMatrixMap(Grow + tstp*es_, nao_, nao_).noalias() = ncplxi * IMap;


  // Must solve bootstrapping equation for steps 1 through k
  // X*M = Q, which we solve as (M^T)*(X^T) = (Q^T)
  // Build M^T and Q^T directly
  for(n = 1; n <= k_; n++){
    ZMatrixMap QMapBlock = ZMatrixMap(Q_.data() + (n-1)*es_, nao_, nao_);

    for(l = 0; l <= k_; l++) {
      auto MMapBlock = MMap.block((n-1)*nao_,(l==0)? 0:(l-1)*nao_, nao_, nao_);

      // Known derivative term goes into Q
      if(l == 0) {
        QMapBlock.noalias() += ncplxi / h * I.poly_diff(n,l) * ZMatrixMap(Grow + tstp*es_, nao_, nao_).transpose();
      }
      else {  // Other derivative terms go into M
        MMapBlock.noalias() -= ncplxi / h * I.poly_diff(n,l) * IMap;
      }

      // Delta Energy term
      if(l == n) {
        MMapBlock.noalias() += mu * IMap - ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
      }

      // Integral terms
      if(l == 0) { // We know both G and Sigma so this goes into Q
          QMapBlock.noalias() += h*I.gregory_weights(n,l) * (ZMatrixMap(Grow + tstp*es_, nao_, nao_)
                                * ZMatrixMap(Srow[0] + (tstp-n)*es_, nao_, nao_)).transpose();
      }
      else { // Goes into M
        if(tstp-l >= tstp-n){ // We have Sig
          MMapBlock.noalias() -= h*I.gregory_weights(n,l) * ZMatrixMap(Srow[l] + (tstp-n)*es_, nao_, nao_).transpose();
        }
        else { // Don't have Sig
            MMapBlock.noalias() += h*I.gregory_weights(n,l) * ZMatrixMap(Srow[n] + (tstp-l)*es_, nao_, nao_).conjugate();
        }
      }
    }
  }

  // Solve XM=Q for X (M^T * X^T = Q^T)
  // We already constructed Q and M to be transposed
  Eigen::FullPivLU<ZMatrix> lu(MMap);
  HNESSI_ASSERT(lu_ok(lu), "dyson::dyson_timestep_ret", HNESSI_KV(lu.rank()));   // DY-T11 (tier 1)
  XMap = lu.solve(QMap);

  // Place results into directly stored slice, calculate error compared to previous iteration
  for(l=0; l<k_; l++) {
    ZMatrixMap Gl(Grow + (tstp-l-1)*es_, nao_, nao_);
    err += (Gl - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    Gl.noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }


  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  ZMatrixMap XMapSmall = ZMatrixMap(X_.data(), nao_, nao_);

  intstart = pnow_();
  // Do first part of integral (column 2)
  memset(Q_.data(), 0, (tstp+1)*es_*sizeof(cplx));
  for( n = k_+1; n <= tstp; n++) {
    ZMatrixMap QMapSmall = ZMatrixMap(Q_.data()+(tstp-n)*es_, nao_, nao_);
    for( l = 0; l <= k_; l++) {
      QMapSmall.noalias() += I.gregory_weights(n,l) * ZMatrixMap(Grow + (tstp-l)*es_, nao_, nao_)
                                          * ZMatrixMap(Srow[l] + (tstp-n)*es_, nao_, nao_);
    }
  }
  intend = pnow_();
  intcurrdur += intend-intstart;
  intdur += intend-intstart;

  // Timestepping yay
  const cplx cdiag = mu - I.bd_weights(0) * ncplxi / h;
  const double hw0 = h * I.omega(0);
  for(n = k_+1; n <= tstp; n++) {
    ZMatrixMap QMapSmall = ZMatrixMap(Q_.data() + (tstp-n)*es_, nao_, nao_);
    QMapSmall *= h;
    // Now we need to include the known derivative values into the RHS
    for(l = 1; l <= k_+1; l++) {
        QMapSmall.noalias() += I.bd_weights(l) * ncplxi / h * ZMatrixMap(Grow + (tstp-(n-l))*es_, nao_, nao_);
    }

    // M contains four parts: H, mu, unknown derivative term, and unknown integral term
    // (MMapSmall is M^T: the right-acting equation has G^R(t,t') [mu - h(t')], so its transpose holds -h^T; ISSUES Q3)
    MMapSmall.noalias() = -ZMatrixMap(H + (tstp-n)*es_, nao_, nao_).transpose();
    MMapSmall.noalias() -= hw0 * ZMatrixMap(Sigma.retptr_col(tstp-n,tstp-n), nao_, nao_).transpose();
    MMapSmall.noalias() += cdiag * IMap;

    // Solve XM=Q (M^T * X^T = Q^T) for X; P1: the LU of M^T(t') (archived t') is cached in slot t' (dyson.hpp)
    // Calculate error and put into directly stored slice
    lu_solve_cached_(tstp-n, true, M_.data(), QMapSmall.data(), true, X_.data(), where);
    ZMatrixMap Gn(Grow + (tstp-n)*es_, nao_, nao_);
    err += (Gn - XMapSmall).norm();
    Gn.noalias() = XMapSmall;

    // Now we have a new value of G which needs to be put into integrals (column 3; P1: per level in timing_lvl_)
    if(n == next_block_apply_at) {
      intstart = pnow_();
      for( int j = 0; j < nao_; j++) {
        for( int k = 0; k < nao_; k++) {
          int blkrows = Sigma.blklen(next_block);
          int blkcols = Sigma.ret().data().blocks()[next_block][k][j].cols();
          int epsrank = Sigma.ret().data().blocks()[next_block][k][j].epsrank();
          // DY-T9 (tier 1): the block fits the dyson scratch ((nt+1)/2*nao; epsrank <= min(rows, cols), HM-U9)
          HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), "dyson::dyson_timestep_ret", HNESSI_KV(epsrank),
                        HNESSI_KV(nao_), HNESSI_KV(epsnao_tmp_.size()));
          ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);

          tmp.noalias() = ZMatrixMap(Sigma.ret().data().blocks()[next_block][k][j].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Grow + Sigma.blkr1(next_block)*es_ + k, blkrows, nao_, Eigen::InnerStride<>(nao_));

          const double *Sv = Sigma.ret().data().blocks()[next_block][k][j].Sdata();   // P1: S applied to the rows of tmp in place
          for(int r = 0; r < epsrank; r++) tmp.row(r) *= Sv[r];

          Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkc1(next_block)*es_ + j, blkcols, nao_, Eigen::InnerStride<>(nao_)).noalias() += ZMatrixMap(Sigma.ret().data().blocks()[next_block][k][j].Vdata(), blkcols, epsrank).conjugate() * tmp;
        }
      }
      intend = pnow_();
      intblkdur += intend-intstart;
      intdur +=    intend-intstart;
      add_lvl_(tstp, 0, Sigma.blklevel(next_block), intstart, intend);

      next_block--;
      if ( next_block != -1 ) {
        next_block_apply_at = tstp-Sigma.blkr1(next_block);
      }
    }

    // Lastly we have to complete the integral for the next timestep, this amounts to 
    // adding in the directly stored elements.
    // First argument is the case where the svd block boundary lies above the integral region
    // second argument is the case where the curr_timestep() tensor is the cutoff
    if( n != tstp) {

      intstart = pnow_();
      const auto dircorrstart = intstart;   // column 5: direct + correction regions together (BUGS B28)
      const int c = tstp-n-1;
      ZMatrixMap Qc(Q_.data() + c * es_, nao_, nao_);
      int lstart = std::max(tstp-Sigma.r2_dir(c), k_+1);
      if(tstp-n < tstp-lstart+1) {
        // P1: column c of the directly stored triangle is contiguous in its row index (doc 11 §3.3, retptr_col):
        // Sigma^R(ll, c) = Scol + (ll - (tstp-n))*es for tstp-n <= ll <= r2_dir(c)
        cplx *Scol = Sigma.retptr_col(tstp-n, c);
        HNESSI_ASSERT(Sigma.retptr_col(tstp-lstart, c) == Scol + (n-lstart)*es_, where, HNESSI_KV(tstp), HNESSI_KV(n));
        for(int ll = tstp-n; ll < tstp-lstart+1; ll++) {
          double weight = ll > tstp-n+k_-1 ? 1. : I.gregory_weights(n+1, tstp-ll);
          Qc.noalias() += weight * ZMatrixMap(Grow + ll*es_, nao_, nao_) * ZMatrixMap(Scol + (ll-(tstp-n))*es_, nao_, nao_);
        }
      }
      intend = pnow_();
      intdirdur += intend-intstart;
      intdur += intend-intstart;

      // There is one caveat, which is if there are less than k+1 stored in a column, we have to
      // adjust some of the values from the svd blocks, which at the moment have coefficient 1
      int lstartcorr = std::max(n+1-k_, k_+1);
      int lendcorr = tstp-Sigma.r2_dir(c)-1;
      intstart = pnow_();
      for( l = lstartcorr; l <= lendcorr; l++) {
        Qc.noalias() += (I.gregory_weights(n+1, l) - 1.)* ZMatrixMap(Grow + (tstp-l)*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(tstp-l,c), nao_, nao_);
      }
      intend = pnow_();
      intcorrdur += intend-intstart;
      intdur += intend-intstart;
      intdircorrdur += intend-dircorrstart;
    }
  }

  retend = pnow_();
  retdur = retend-retstart;

  if(profile_) timing(tstp, 0) = retdur.count();
  if(profile_) timing(tstp, 1) = intdur.count();
  if(profile_) timing(tstp, 2) = intcurrdur.count();
  if(profile_) timing(tstp, 3) = intblkdur.count();
  if(profile_) timing(tstp, 4) = intdirdur.count();
  if(profile_) timing(tstp, 5) = intdircorrdur.count();
  if(profile_) timing(tstp, 6) = intcorrdur.count();

  stamp_(G, tstp, st4_ret);   // ST-4
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_les_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  call_scope cs(*this);
  const char *where = "dyson::dyson_timestep_les_nobc";
  if(cs.outer()) {
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, false);
    require_stamp_(where, G, tstp, st4_ret | st4_tv, "call dyson_timestep_ret and dyson_timestep_tv for this tstp and G first (or use dyson_timestep_nobc)");   // ST-4 (DY-T7)
  }

  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  clock_tp_ lesstart, lesend, int1start, int1end, int2start, int2end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0);

  lesstart = pnow_();

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
  int2start = pnow_();
  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = pnow_();
  int2dur = int2end-int2start;

  int1start = pnow_();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = pnow_();
  int1dur = int1end-int1start;

  // Initial Condition. Do not use tv, remember integral is transposed
  // integral
  // B14: this boundary equation is the Hermitian conjugate of Eq. 8 at t = 0 multiplied by -1 (LHS +i a_0/h, the
  // other RHS terms carry the matching signs), so the mixed term -i int G^rceil(0,tau) Sigma^lceil(tau,T) enters
  // with a minus sign: Y0 L = ... - [les_it_int(0,T,G,Sigma)]^T.
  memset(Q_.data(),0,sizeof(cplx)*es_);
  les_it_int(0, tstp, G, Sigma, Q_.data());
  ZMatrixMap(Q_.data(), nao_, nao_) *= -1.;

  // derivative
  for(int l = 1; l <= k_; l++) {
    ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(0,tstp-l), nao_, nao_).transpose();
  }
  G.get_les(0,tstp-(k_+1), X_.data());
  ZMatrixMap(Q_.data(), nao_, nao_) -= cplxi/h * I.bd_weights(k_+1) * ZMatrixMap(X_.data(), nao_, nao_).transpose();
  
  // M 
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

        HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), where, HNESSI_KV(epsrank), HNESSI_KV(epsnao_tmp_.size())); ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);   // DY-T9; P1: preallocated (was a ZMatrix per (b,i,k))

        tmp.noalias() = ZMatrixMap(G.les().data().blocks()[b][i][k].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Sigma.curr_timestep_ret_ptr(tstp, lstart) + k, blkrows, nao_, Eigen::InnerStride<>(nao_)).conjugate();

        const double *Sv = G.les().data().blocks()[b][i][k].Sdata(); for(int r = 0; r < epsrank; r++) tmp.row(r) *= -h * Sv[r];   // P1: in place

        ZMatrixMap(Q_.data(), nao_, nao_).col(i).noalias() += tmp.transpose() * ZMatrixMap(G.les().data().blocks()[b][i][k].Vdata(), 1, epsrank).adjoint();   // (conj(V row 0) tmp)^T
      }
    }
  }
  Eigen::FullPivLU<ZMatrix> luIC(ZMatrixMap(M_.data(), nao_, nao_));
  HNESSI_ASSERT(lu_ok(luIC), "dyson::dyson_timestep_les_nobc", HNESSI_KV(luIC.rank()));   // DY-T11 (tier 1)
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
  HNESSI_ASSERT(lu_ok(lu), "dyson::dyson_timestep_les_nobc", HNESSI_KV(lu.rank()));   // DY-T11 (tier 1)
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));

  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  err += les_march_(tstp, G, mu, H, Sigma, I, h, where);
  lesend = pnow_();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_les_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  call_scope cs(*this);
  const char *where = "dyson::dyson_timestep_les_2leg";
  if(cs.outer()) {
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, false);
    require_stamp_(where, G, tstp, st4_ret, "call dyson_timestep_ret for this tstp and G first (or use dyson_timestep_2leg)");   // ST-4 (DY-T7)
  }

  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  clock_tp_ lesstart, lesend, int1start, int1end, int2start, int2end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0);

  lesstart = pnow_();

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
  int2start = pnow_();
//  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = pnow_();
  int2dur = int2end-int2start;

  int1start = pnow_();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = pnow_();
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

        HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), where, HNESSI_KV(epsrank), HNESSI_KV(epsnao_tmp_.size())); ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);   // DY-T9; P1: preallocated (was a ZMatrix per (b,i,k))

        tmp.noalias() = ZMatrixMap(G.les().data().blocks()[b][i][k].Udata(), blkrows, epsrank).transpose() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Sigma.curr_timestep_ret_ptr(tstp, lstart) + k, blkrows, nao_, Eigen::InnerStride<>(nao_)).conjugate();

        const double *Sv = G.les().data().blocks()[b][i][k].Sdata(); for(int r = 0; r < epsrank; r++) tmp.row(r) *= -h * Sv[r];   // P1: in place

        ZMatrixMap(Q_.data(), nao_, nao_).col(i).noalias() += tmp.transpose() * ZMatrixMap(G.les().data().blocks()[b][i][k].Vdata(), 1, epsrank).adjoint();   // (conj(V row 0) tmp)^T
      }
    }
  }
  Eigen::FullPivLU<ZMatrix> luIC(ZMatrixMap(M_.data(), nao_, nao_));
  HNESSI_ASSERT(lu_ok(luIC), "dyson::dyson_timestep_les_2leg", HNESSI_KV(luIC.rank()));   // DY-T11 (tier 1)
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
  HNESSI_ASSERT(lu_ok(lu), "dyson::dyson_timestep_les_2leg", HNESSI_KV(lu.rank()));   // DY-T11 (tier 1)
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));

  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  err += les_march_(tstp, G, mu, H, Sigma, I, h, where);
  lesend = pnow_();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_les(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  call_scope cs(*this);
  const char *where = "dyson::dyson_timestep_les";
  if(cs.outer()) {
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, false);
    require_stamp_(where, G, tstp, st4_ret | st4_tv, "call dyson_timestep_ret and dyson_timestep_tv for this tstp and G first (or use dyson_timestep)");   // ST-4 (DY-T7)
  }

  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  clock_tp_ lesstart, lesend, int1start, int1end, int2start, int2end;
  std::chrono::duration<double, std::micro> lesdur(0), int1dur(0), int2dur(0);

  lesstart = pnow_();

  double err = 0;
  cplx cplxi = cplx(0,1);

  // Matrix Maps
  ZMatrixMap MMap = ZMatrixMap(M_.data(), k_*nao_, k_*nao_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  ZMatrixMap SMap = ZMatrixMap(bound_.data(), nao_, nao_); // Used for boundary cases where we need to extract Sigma from SVD


  // Integrals go into Q via increment.  Q must be 0.
  memset(Q_.data(),0,sizeof(cplx)*(tstp+1)*es_);

  // INTEGRAL 1 AND 2 GO - Integrals are transposed
  int2start = pnow_();
  les_it_int(tstp, Sigma, G, Q_.data());
  int2end = pnow_();
  int2dur = int2end-int2start;

  int1start = pnow_();
  les_lesadv_int_0_tstp(tstp, Sigma, G, h, I);
  int1end = pnow_();
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
  HNESSI_ASSERT(lu_ok(lu), "dyson::dyson_timestep_les", HNESSI_KV(lu.rank()));   // DY-T11 (tier 1)
  ZMatrixMap(X_.data() + es_, k_*nao_, nao_).noalias() = lu.solve(ZMatrixMap(Q_.data() + es_, k_*nao_, nao_));
  for(int i = 1; i <= k_; i++) {
    ZMatrixMap(X_.data() + i*es_, nao_, nao_) = ZMatrixMap(X_.data() + i*es_, nao_, nao_).transpose().eval();
  }

  err += les_march_(tstp, G, mu, H, Sigma, I, h, where);
  lesend = pnow_();
  lesdur = lesend-lesstart;

  if(profile_) timing(tstp, 10) = lesdur.count();
  if(profile_) timing(tstp, 11) = int2dur.count();
  if(profile_) timing(tstp, 12) = int1dur.count();

  return cs.outer() ? ensure_finite_(err, where) : err;
}

// P1: rows m = k+1..T of the les components, shared by dyson_timestep_les, _les_nobc and _les_2leg (the three copies
// were identical). On entry Q_ holds the transposed it/lesadv integrals of rows 0..T and X_ rows 0..k of the solution
// (transposed back); the march applies the Sigma^R blocks itself (first those with blkc2 <= k).
double dyson::les_march_(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                         Integration::Integrator &I, double h, const char *where) {
  clock_tp_ int3start, int3end;
  std::chrono::duration<double, std::micro> int3dur(0), int3dur_curr(0), int3dur_corr(0), int3dur_block(0);
  zero_lvl_(tstp, 1);
  lu_params_(mu, h, Sigma, H);   // P1: clear the step-matrix cache if mu, h, Sigma or the h array changed

  int next_block_apply_at = Sigma.blkc2(0);
  int next_block = 0;
  double err = 0;
  const cplx cplxi = cplx(0,1);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  // one TSVD block of Sigma^R applied to the solved rows: Q(rows of b) += h U S V^dagger X(cols of b)
  auto apply_block = [&](int b) {
    int3start = pnow_();
    for(int i = 0; i < nao_; i++) {
      for(int k = 0; k < nao_; k++) {
        int blkrows = Sigma.blklen(b);
        int blkcols = Sigma.ret().data().blocks()[b][i][k].cols();
        int epsrank = Sigma.ret().data().blocks()[b][i][k].epsrank();
        // DY-T9 (tier 1): the block fits the dyson scratch (P1: preallocated instead of a ZMatrix per (i,k))
        HNESSI_ASSERT(epsrank*nao_ <= (int)epsnao_tmp_.size(), where, HNESSI_KV(epsrank), HNESSI_KV(nao_),
                      HNESSI_KV(epsnao_tmp_.size()));
        ZMatrixMap tmp(epsnao_tmp_.data(), epsrank, nao_);

        tmp.noalias() = ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Vdata(), blkcols, epsrank).adjoint() * Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(X_.data() + Sigma.blkc1(b)*es_ + k, blkcols, nao_, Eigen::InnerStride<>(nao_));

        const double *Sv = Sigma.ret().data().blocks()[b][i][k].Sdata();   // P1: h S applied to the rows of tmp in place
        for(int r = 0; r < epsrank; r++) tmp.row(r) *= h * Sv[r];

        Eigen::Map<Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::InnerStride<> >(Q_.data() + Sigma.blkr1(b)*es_ + i, blkrows, nao_, Eigen::InnerStride<>(nao_)).noalias() += ZMatrixMap(Sigma.ret().data().blocks()[b][i][k].Udata(), blkrows, epsrank) * tmp;
      }
    }
    int3end = pnow_();
    int3dur += int3end-int3start;
    int3dur_block += int3end-int3start;
    add_lvl_(tstp, 1, Sigma.blklevel(b), int3start, int3end);
  };

  // Do first part of retles integral
  for(int b = 0; b < Sigma.built_blocks(); b++) {
    if(Sigma.blkc2(b) <= k_) {
      if(Sigma.blkr2(b) > k_) apply_block(b);
      next_block++;
      // DY-T10: past the last block there is nothing left to apply (m-1 >= k never equals -1)
      next_block_apply_at = next_block < Sigma.nbox() ? Sigma.blkc2(next_block) : -1;
    }
    else {
      break;
    }
  }

  // P1: rows m >= tstpmk of Sigma^R are in the curr_timestep buffer, contiguous in the column (doc 11 §3.3)
  const int smk = Sigma.tstpmk();
  const cplx cdiag = cplxi/h*I.bd_weights(0) + mu;
  const double hw0 = h*I.omega(0);

  // Timestepping
  ZMatrixMap MMapSmall = ZMatrixMap(M_.data(), nao_, nao_);
  for(int m = k_+1; m <= tstp; m++) {
    ZMatrixMap Qm(Q_.data() + m*es_, nao_, nao_);

    // Check if next box can be applied
    if(next_block_apply_at == m-1) {
      apply_block(next_block);
      next_block++;
      if(next_block < Sigma.nbox()) next_block_apply_at = Sigma.blkc2(next_block);
    }

    // finish direct
    int3start = pnow_();
    if(m < smk) {
      for(int tbar = Sigma.c1_dir(m); tbar < m; tbar++) {
        Qm.noalias() += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data()+tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_col(m,tbar), nao_, nao_).transpose();
      }
    }
    else {
      cplx *Srow = Sigma.curr_timestep_ret_ptr(m, 0);
      for(int tbar = 0; tbar < m; tbar++) {
        Qm.noalias() += h * I.gregory_weights(m,tbar) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Srow + tbar*es_, nao_, nao_).transpose();
      }
    }
    int3end = pnow_();
    int3dur += int3end-int3start;
    int3dur_curr += int3end-int3start;

    // do edge corrections on left
    int3start = pnow_();
    if(m < smk && m >= Sigma.blkr1(0)) {
      for(int tbar = 0; tbar <= std::min(k_, Sigma.c1_dir(m)-1); tbar++) {
        Qm.noalias() += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.ret_left_edge().data() + (m*(k_+1) + tbar)*nao_*nao_, nao_, nao_).transpose();
      }
      // do edge corrections on right
      // k+1 case is for when m is small.  Right and left correction regions overlap.  Taken care of by left
      for(int tbar = std::max(m-k_,k_+1); tbar < Sigma.c1_dir(m); tbar++) {
        Qm.noalias() += h * (I.gregory_weights(m,tbar)-1) * ZMatrixMap(X_.data() + tbar*es_, nao_, nao_) * ZMatrixMap(Sigma.retptr_corr(m,tbar), nao_, nao_).transpose();
      }
    }
    int3end = pnow_();
    int3dur += int3end-int3start;
    int3dur_corr += int3end-int3start;


    // integrals are transposed (P1: in place, was a temporary)
    Qm.transposeInPlace();

    // BULK OF ROW
    if(m != tstp || rho_version_ == 0) { // (m == tstp: HORIZONTAL FOR RHO, the same equation)
      // Set up M
      cplx *sigptrmm = m >= smk ? Sigma.curr_timestep_ret_ptr(m,m) : Sigma.retptr_col(m,m);
      MMapSmall.noalias() = -ZMatrixMap(H+m*es_, nao_, nao_) + cdiag*IMap - hw0*ZMatrixMap(sigptrmm, nao_, nao_);

      // Derivatives into Q
      for(int l = 1; l <= k_+1; l++) {
        Qm.noalias() -= cplxi/h*I.bd_weights(l) * ZMatrixMap(X_.data() + (m-l)*es_, nao_, nao_).transpose();
      }
    }
    else { // rho_version_ == 1: DIAGONAL FOR RHO
      ZMatrixMap GLtt(G.curr_timestep_les_ptr(tstp, tstp), nao_, nao_);
      // finish integral
      Qm.noalias() += h * I.omega(0) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(tstp, tstp), nao_, nao_) * GLtt;
      // hamiltonian
      Qm.noalias() += (ZMatrixMap(H+tstp*es_, nao_, nao_) - mu*IMap) * GLtt;
      // diagonal
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() = -cplxi * (Qm + Qm.adjoint());
      // Derivatives into Q
      for(int l = 1; l <= k_; l++) {
        ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(l) * ZMatrixMap(G.curr_timestep_les_ptr(tstp-l, tstp-l), nao_, nao_);
      }
      G.get_les(tstp-k_-1, tstp-k_-1, M_.data());
      ZMatrixMap(Q_.data(), nao_, nao_).noalias() -= 1./h*I.bd_weights(k_+1) * ZMatrixMap(M_.data(), nao_, nao_);
      MMapSmall.noalias() = 1./h*I.bd_weights(0) * IMap;
      Qm = ZMatrixMap(Q_.data(), nao_, nao_);
    }

    // Solve MX=Q; P1: the LU of M(m), m < T, is cached in slot nt+m (dyson.hpp)
    lu_solve_cached_(nt_ + m, m < tstp, M_.data(), Qm.data(), false, X_.data() + m*es_, where);   // row T: never cached
  }

  // Write elements into G
  for(int l = 0; l <= tstp; l++) {
    ZMatrixMap Gl(G.curr_timestep_les_ptr(l,tstp), nao_, nao_);
    err += (Gl - ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose()).norm();
    Gl.noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_).transpose();
  }

  if(profile_) timing(tstp, 21) = int3dur.count();
  if(profile_) timing(tstp, 22) = int3dur_block.count();
  if(profile_) timing(tstp, 23) = int3dur_curr.count();
  if(profile_) timing(tstp, 24) = int3dur_corr.count();
  return err;
}

double dyson::dyson_timestep_tv(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h){
  call_scope cs(*this);
  const char *where = "dyson::dyson_timestep_tv";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, tstp, tstp, true);

  assert(tstp > I.get_k());
  assert(Sigma.nt() > tstp);

  assert(G.nt() > tstp);
  assert(G.sig() == Sigma.sig());
  assert(G.size1() == Sigma.size1());

  assert(G.r() == Sigma.r());

  assert(G.size1() == nao_);
  assert(I.get_k() == k_);

  clock_tp_ tvstart, tvend, int1start, int1end, int2start, int2end;
  std::chrono::duration<double, std::micro> tvdur(0), int1dur(0), int2dur(0);

  tvstart = pnow_();

  double err = 0;

  cplx cplxi = cplx(0.,1.);
  auto IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  auto XMap = ZMatrixMap(Q_.data(), nao_, nao_);
  auto MMap = ZMatrixMap(M_.data(), nao_, nao_);

  // Do integrals and put them into G.tv(tstp,:)
  int1start = pnow_();
  tv_ret_int(tstp, Sigma, G, I, h);
  int1end = pnow_();
  int1dur = int1end-int1start;

  int2start = pnow_();
  tv_it_conv(tstp, Sigma, G, G.tvptr(tstp,0));
  int2end = pnow_();
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
  HNESSI_ASSERT(lu_ok(lu), "dyson::dyson_timestep_tv", HNESSI_KV(lu.rank()));   // DY-T11 (tier 1)

  // Solve MX=Q
  for(int m=0; m<r_; m++) {
    XMap = lu.solve(ZMatrixMap(G.tvptr(tstp, m), nao_, nao_));
    err += (ZMatrixMap(NTauTmp_.data()+m*es_, nao_, nao_) - XMap).norm();
    ZMatrixMap(G.tvptr(tstp,m), nao_, nao_).noalias() = XMap;
  }

  tvend = pnow_();
  tvdur = tvend-tvstart;


  if(profile_) timing(tstp, 7) = tvdur.count();
  if(profile_) timing(tstp, 8) = int1dur.count();
  if(profile_) timing(tstp, 9) = int2dur.count();

  stamp_(G, tstp, st4_tv);   // ST-4
  return cs.outer() ? ensure_finite_(err, where) : err;
}

} // namespace
