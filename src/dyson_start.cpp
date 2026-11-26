#include "h_nessi/dyson.hpp"
#include <cmath>
#include <iomanip>

using namespace std::chrono;

namespace h_nessi {

namespace {
// DY-S10 (tier 2): the bootstrap block systems must be non-singular (once per solve, O(n) after the O(n^3) LU)
void ensure_lu(const Eigen::FullPivLU<ZMatrix> &lu, const char *where) {
  HNESSI_ENSURE(lu.isInvertible(), where, HNESSI_KV(lu.rank()), HNESSI_KV(lu.rows()),
                detail::note{"the bootstrap linear system is singular (or not finite)"});
}

}  // namespace

double dyson::dyson_start_tv(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_tv";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, true);

  // We also enforce TTI for 0 <= t <= k
  // This is guaranteed using checks in the dyson_start() routine as well
  int m, l, n, i;
  double err = 0;

  int es = G.size1() * G.size2();
  int r = G.r();

  cplx cplxi = cplx(0.,1.);
  ZMatrixMap QMap = ZMatrixMap(Q_.data(), nao_*k_, nao_);
  ZMatrixMap MMap = ZMatrixMap(M_.data(), nao_*k_, nao_*k_);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);

  // Boundary Conditions
  ZMatrixMap(NTauTmp_.data(), r_, es_) = DMatrixConstMap(dlr_.it2itr(), r_, r_).transpose() * DMatrixMap(G.matptr(0), r_, es_);
  for(m=0; m<r_; m++) {
    auto tvmap = ZMatrixMap(G.tvptr(0,m), nao_, nao_);

    auto matmap = ZMatrixMap(NTauTmp_.data()+m*es_, nao_, nao_);
    err += (tvmap - (double)G.sig()*cplxi*matmap).norm();

    tvmap.noalias() = (double)G.sig()*cplxi*matmap;
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
    ensure_lu(lu, where);   // DY-S10 (tier 2)
    ZMatrixMap(X_.data(), k_*nao_, nao_) = lu.solve(QMap);

    for(l=0; l<k_; l++){
      err += (ZColVectorMap(G.tvptr(l+1,m), es_) - ZColVectorMap(X_.data() + l*es_, es_)).norm();
      ZMatrixMap(G.tvptr(l+1,m), nao_, nao_).noalias() = ZMatrixMap(X_.data() + l*es_, nao_, nao_);
    }
  }

  stamp_(G, boot_stamp_, st4_tv);   // ST-4
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ret(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_ret";
  if(cs.outer()) {
    enter_start_(where, G, Sigma, I, h, mu, H, false, false);
    require_stamp_(where, G, boot_stamp_, st4_tv, "call dyson_start_tv for this G first (or use dyson_start)");   // ST-4 (DY-S6)
  }

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
  
 
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_ret_ntti";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, false);

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
        else     MMap.block((n-m-1)*nao_, (l-m-1)*nao_, nao_, nao_) -= -h*I.poly_integ(m,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_).adjoint(); // Sigma^R(n,l) = -Sigma^R(l,n)^dagger
      }

    }

    // Solve MX=Q for X
    Eigen::FullPivLU<ZMatrix> lu(ZMatrixMap(M_.data(), (k_-m)*nao_, (k_-m)*nao_));
    ensure_lu(lu, where);   // DY-S10 (tier 2)
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
      // M multiplies X = G^R(m,0) from the right (right-acting equation at t' = 0): G^R(m,0) [mu - h(0)] (was h(0)^dagger,
      // equal for the Hermitian h that D-11 guarantees; ISSUES Q3)
      MMap0 = -ncplxi/h * I.poly_diff(k_,k_) * IMap - ZMatrixMap(H, nao_, nao_) + mu*IMap - h * I.poly_integ(k_-m,k_,k_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(0,0), nao_, nao_);
      QMap0.setZero();
      for(int i = 0; i < k_-m; i++)  QMap0 -= ncplxi/h * I.poly_diff(k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(k_-i,m), nao_, nao_).adjoint();
      for(int i = k_-m; i < k_; i++) QMap0 += ncplxi/h * I.poly_diff(k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(m,k_-i), nao_, nao_);
      for(int i = 0; i < k_-m; i++)  QMap0 -= h * I.poly_integ(k_-m,k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(k_-i,m), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(k_-i,0), nao_, nao_);
      for(int i = k_-m; i < k_; i++) QMap0 += h * I.poly_integ(k_-m,k_,i) * ZMatrixMap(G.curr_timestep_ret_ptr(m,k_-i), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(k_-i,0), nao_, nao_);

      // Solve XM=Q for X, i.e. M^T X^T = Q^T
      Eigen::FullPivLU<ZMatrix> lu(ZMatrixMap(M_.data(), nao_, nao_).transpose());
      ensure_lu(lu, where);   // DY-S10 (tier 2)
      ZMatrixMap(X_.data(), nao_, nao_) = lu.solve(ZMatrixMap(Q_.data(), nao_, nao_).transpose()).transpose();
      err += (ZColVectorMap(GR0.data() + (m-1)*es_, es_) - ZColVectorMap(X_.data(), es_)).norm();
      ZMatrixMap(G.curr_timestep_ret_ptr(m,0), nao_, nao_).noalias() = ZMatrixMap(X_.data(), nao_, nao_);
    }
  }

  stamp_(G, boot_stamp_, st4_ret_ntti);   // ST-4
  return cs.outer() ? ensure_finite_(err, where) : err;
}


double dyson::dyson_start_les(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_les";
  if(cs.outer()) {
    enter_start_(where, G, Sigma, I, h, mu, H, true, false);
    require_stamp_(where, G, boot_stamp_, st4_tv, "call dyson_start_tv for this G first (or use dyson_start)");   // ST-4 (DY-S6)
  }

  // We enforce TTI for 0 <= t <= k
  // This is guaranteed using checks in the dyson_start() routine
  double err = 0;
  for(int tp = 0; tp <= k_; tp++) {
    G.get_tv_tau(tp, 0, dlr_, M_.data());
    ZMatrix X = -ZMatrixMap(M_.data(), nao_, nao_).adjoint();
    if(tp == 0) X = boundary_antihermitian_(X);
    for( int l = 0; l <= k_-tp; l++) {
      err += (ZMatrixMap(G.curr_timestep_les_ptr(l,l+tp), nao_, nao_) - X).norm();
      ZMatrixMap(G.curr_timestep_les_ptr(l,l+tp), nao_, nao_).noalias() = X;
    }
  }

  return cs.outer() ? ensure_finite_(err, where) : err;
}


// WP7b side finding 4 (WP9a): the boundary value G^<(0,0) = -[G^rceil(0,0)]^dagger is anti-Hermitian only to the
// accuracy of the off-node DLR evaluation of G^rceil at tau = 0 (~1e-13); project it onto its anti-Hermitian part.
// (X - X^dagger)/2 is exactly anti-Hermitian in floating point (a - b == -(b - a), real diagonal exactly 0).
ZMatrix dyson::boundary_antihermitian_(const ZMatrix &X) {
  return 0.5 * (X - X.adjoint());
}

// Rows m = 1..k of G^< (and, with RHO_DIAGONAL, the diagonal) for dyson_start_les_ntti, _ntti_nobc and _2leg, after
// row m = 0 has been set by the caller. The three differ only in the mixed les_it_int terms (absent for _2leg). (Until
// WP9 the ntti variant used h^* in the row matrix and h^dagger in the diagonal equation where the equations have h^T and
// h; equal for the Hermitian h that D-11 guarantees, ISSUES Q3.) err accumulates the change of G^<(m,n), 1 <= m <= n <= k.
void dyson::start_les_rows_(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I,
                            double h, bool mixed, double &err, const char *where) {
  const cplx cplxi = cplx(0,1);
  ZMatrixMap IMap = ZMatrixMap(iden_.data(), nao_, nao_);
  const int kn = k_*nao_;

  // the transposed right-acting row system for X_l = [G^<(m,l)]^T, l = 1..k (doc 22 §4.6 step 3)
  auto build_row = [&](int m, ZMatrix &MIC, ZMatrix &QIC) {
    MIC = ZMatrix::Zero(kn, kn);
    QIC = ZMatrix::Zero(kn, nao_);
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
      if(mixed) les_it_int(m, n, G, Sigma, M_.data());
      QBlock += ZMatrixMap(M_.data(), nao_, nao_);
    }
    for(int n = 1; n <= k_; n++) {
      for(int l = 1; l <= k_; l++) {
        auto MBlock = MIC.block((n-1)*nao_, (l-1)*nao_, nao_, nao_);
        MBlock += -cplxi/h * I.poly_diff(n,l) * IMap;
        if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;   // [G^<(m,l) (h(l)-mu)]^T
        if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
        else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
      }
    }
  };

  if(rho_version_ == 0) {   // RHO_HORIZONTAL: the diagonal G^<(m,m) is the n = m entry of row m
    ZMatrix MIC, QIC, XIC;
    for(int m = 1; m <= k_; m++) {
      build_row(m, MIC, QIC);
      Eigen::FullPivLU<ZMatrix> lu2(MIC);
      ensure_lu(lu2, where);   // DY-S10 (tier 2)
      XIC = lu2.solve(QIC);
      for(int i = m; i <= k_; i++) {
        err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
        ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
      }
    }
    return;
  }

  // RHO_DIAGONAL (B34, WP7/WP7b). The diagonal G^<(m,m) is not an unknown of row m: as in the time stepper it obeys the
  // diagonal (density-matrix) equation d/dt G^<(t,t) = B - B^dagger. Row m is solved with G^<(m,m) as known data
  // (column l = m moved to the right-hand side, the collocation equation at n = m dropped), and the diagonal is solved
  // from B evaluated with the rows and the diagonal values themselves. The rows depend on the diagonal and the diagonal
  // on the rows, so the two are iterated here to their common fixed point (to roundoff): for fixed Sigma one call is
  // then idempotent, and a user loop over dyson_start_* is needed only for self-consistency in Sigma.

  // input values, for the returned change of G
  std::vector<ZMatrix> Gin;
  for(int m = 1; m <= k_; m++)
    for(int i = m; i <= k_; i++) Gin.push_back(ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_));

  // rows: the matrices do not depend on the diagonal. Factorise once with column/equation m removed (identity block),
  // keep the removed column C_m and the right-hand side Q_m.
  std::vector<Eigen::FullPivLU<ZMatrix>> lu_row(k_+1);
  std::vector<ZMatrix> Q_row(k_+1), C_row(k_+1);
  for(int m = 1; m <= k_; m++) {
    ZMatrix MIC;
    build_row(m, MIC, Q_row[m]);
    const int c = (m-1)*nao_;
    C_row[m] = MIC.block(0, c, kn, nao_);
    MIC.block(0, c, kn, nao_).setZero();
    MIC.block(c, 0, nao_, kn).setZero();
    MIC.block(c, c, nao_, nao_).setIdentity();
    lu_row[m].compute(MIC);
    ensure_lu(lu_row[m], where);   // DY-S10 (tier 2)
  }

  // diagonal: (1/h) sum_j D_ij G^<(j,j) = B_i - B_i^dagger - (1/h) D_i0 G^<(0,0), i = 1..k
  ZMatrix MD = ZMatrix::Zero(kn, kn);
  for(int i = 1; i <= k_; i++)
    for(int j = 1; j <= k_; j++) MD.block((i-1)*nao_, (j-1)*nao_, nao_, nao_) = 1./h * I.poly_diff(i,j) * IMap;
  Eigen::FullPivLU<ZMatrix> lu3(MD);
  ensure_lu(lu3, where);   // DY-S10 (tier 2)
  std::vector<ZMatrix> mix(k_+1, ZMatrix::Zero(nao_, nao_));   // -i int G^| Sigma^| (i,i): independent of G^<
  if(mixed) {
    for(int i = 1; i <= k_; i++) {
      ZMatrixMap(M_.data(), nao_, nao_) = ZMatrix::Zero(nao_, nao_);
      les_it_int(i, i, G, Sigma, M_.data());
      mix[i] = ZMatrixMap(M_.data(), nao_, nao_).transpose();
    }
  }

  ZMatrix QIC(kn, nao_), XIC(kn, nao_);
  double change = 0, scale = 1, change_min = INFINITY;
  int pass = 0;
  bool converged = false, diverged = false;
  while(!converged && !diverged && pass < start_diag_max_pass) {
    pass++;
    // rows m = 1..k with the current diagonal G^<(m,m) known; only G^<(m,i), i > m, are written
    for(int m = 1; m <= k_; m++) {
      const int c = (m-1)*nao_;
      const ZMatrix XmT = ZMatrixMap(G.curr_timestep_les_ptr(m,m), nao_, nao_).transpose();
      QIC = Q_row[m];
      for(int n = 1; n <= k_; n++) {
        if(n != m) QIC.block((n-1)*nao_, 0, nao_, nao_) -= C_row[m].block((n-1)*nao_, 0, nao_, nao_) * XmT;
      }
      QIC.block(c, 0, nao_, nao_) = XmT;
      XIC = lu_row[m].solve(QIC);
      for(int i = m+1; i <= k_; i++) {
        ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
      }
    }
    // ISSUES Q8 (D-8, intended): the right-hand side is explicit: B uses the rows G^<(i,l) of this pass and the
    // diagonal G^<(i,i) of the previous pass (the l = i terms); the loop below iterates the pair to its fixed point.
    // diagonal. B_i = i[ G^<(i,i) eps(i) + int G^R Sigma^< + int G^< Sigma^A - i int G^| Sigma^| ] (doc 22 §4.6
    // step 4); the l = i term of int G^< Sigma^A uses G^<(i,i) itself.
    // I         =  S^R G^< + S^< G^A
    // I^\dagger = -G^< S^A - G^R S^<
    for(int i = 1; i <= k_; i++) {
      ZMatrix B = ZMatrix::Zero(nao_, nao_);
      B += cplxi * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * (ZMatrixMap(H+i*es_, nao_, nao_) - mu*IMap);
      for(int l = 0; l <= i; l++) {
        B += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_les_ptr(l,i), nao_, nao_);
      }
      for(int l = i+1; l <= k_; l++) {
        B += cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_ret_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_les_ptr(i,l), nao_, nao_).adjoint();
      }
      for(int l = 0; l < i; l++) {
        B -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(l,i), nao_, nao_).adjoint() * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,l), nao_, nao_).adjoint();
      }
      B += cplxi * I.poly_integ(0,i,i) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(i,i), nao_, nao_).adjoint();
      for(int l = i+1; l <= k_; l++) {
        B -= cplxi * I.poly_integ(0,i,l) * h * ZMatrixMap(G.curr_timestep_les_ptr(i,l), nao_, nao_) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,i), nao_, nao_);
      }
      B += cplxi * mix[i];
      QIC.block((i-1)*nao_, 0, nao_, nao_) = B - B.adjoint() - 1./h * I.poly_diff(i,0) * ZMatrixMap(G.curr_timestep_les_ptr(0,0), nao_, nao_);
    }
    XIC = lu3.solve(QIC);
    change = 0;
    scale = 1;
    for(int i = 1; i <= k_; i++) {
      auto Gii = ZMatrixMap(G.curr_timestep_les_ptr(i,i), nao_, nao_);
      auto Xi = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_);
      change = std::max(change, (Xi - Gii).norm());
      scale = std::max(scale, Xi.norm());
      Gii = Xi;
    }
    converged = change <= start_diag_rel_tol * scale;
    change_min = std::min(change_min, change);
    diverged = !std::isfinite(change) || (pass > 10 && change > 100 * change_min);   // growing again: not contracting
  }
  // a non-contracting fixed point (the explicit diagonal update, like the RHO_DIAGONAL time step, can diverge for large
  // h*spread(eps)): do not return a silently unconverged bootstrap
  HNESSI_ENSURE(converged, where, HNESSI_KV(pass), HNESSI_KV(diverged), HNESSI_KV(change), HNESSI_KV(change_min),
                HNESSI_KV(scale), HNESSI_KV(k_), HNESSI_KV(h),
                detail::note{"the RHO_DIAGONAL bootstrap fixed point (rows <-> diagonal, t = 0..k) did not converge "
                             "(diverging or too slow). It stops contracting at about h*spread(h(t)) ~ 0.6-1 (smaller for "
                             "larger k; spread = largest minus smallest eigenvalue of h(t)), at roughly half the h at "
                             "which the RHO_DIAGONAL time step corrector stops converging (WP9a measurement); reduce h or "
                             "use RHO_HORIZONTAL"});

  size_t q = 0;
  for(int m = 1; m <= k_; m++)
    for(int i = m; i <= k_; i++) err += (ZMatrixMap(G.curr_timestep_les_ptr(m,i), nao_, nao_) - Gin[q++]).norm();
}

double dyson::dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_les_ntti";
  if(cs.outer()) {
    enter_start_(where, G, Sigma, I, h, mu, H, true, false);
    require_stamp_(where, G, boot_stamp_, st4_tv | st4_ret_ntti, "call dyson_start_tv and dyson_start_ret_ntti for this G first (or use dyson_start_ntti)");   // ST-4 (DY-S6)
  }

  double err = 0;
  for(int tp = 0; tp <= k_; tp++) {
    G.get_tv_tau(tp, 0, dlr_, M_.data());
    ZMatrix X = -ZMatrixMap(M_.data(), nao_, nao_).adjoint();
    if(tp == 0) X = boundary_antihermitian_(X);
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,tp), nao_, nao_) - X).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,tp), nao_, nao_).noalias() = X;
  }

  // rows m = 1..k and, with RHO_DIAGONAL, the diagonal (converged to the fixed point of the pair, B34)
  start_les_rows_(G, mu, H, Sigma, I, h, true, err, where);

  return cs.outer() ? ensure_finite_(err, where) : err;
}


double dyson::dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_les_ntti_nobc";
  if(cs.outer()) {
    enter_start_(where, G, Sigma, I, h, mu, H, true, false);
    require_stamp_(where, G, boot_stamp_, st4_tv | st4_ret_ntti, "call dyson_start_tv and dyson_start_ret_ntti for this G first (or use dyson_start_ntti_nobc)");   // ST-4 (DY-S6)
  }
  check_les00_(where, G);   // DY-S7: the caller set G^<(0,0)

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
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
      if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
    }
  }
  Eigen::FullPivLU<ZMatrix> lu(MIC);
  ensure_lu(lu, where);   // DY-S10 (tier 2)
  XIC = lu.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
  }

  // rows m = 1..k and, with RHO_DIAGONAL, the diagonal (converged to the fixed point of the pair, B34)
  start_les_rows_(G, mu, H, Sigma, I, h, true, err, where);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(*this);
  const char *where = "dyson::dyson_start_les_2leg";
  if(cs.outer()) {
    enter_start_(where, G, Sigma, I, h, mu, H, true, false);
    require_stamp_(where, G, boot_stamp_, st4_ret_ntti, "call dyson_start_ret_ntti for this G first (or use dyson_start_2leg)");   // ST-4 (DY-S6)
  }
  check_les00_(where, G);   // DY-S7: the caller set G^<(0,0)

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
      if(n==l) MBlock -= ZMatrixMap(H+l*es_, nao_, nao_).transpose() - mu*IMap;
      if(n>=l) MBlock -= (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(n,l), nao_, nao_).adjoint()).transpose();
      else if(n<l) MBlock += (h * I.poly_integ(0,n,l) * ZMatrixMap(Sigma.curr_timestep_ret_ptr(l,n), nao_, nao_)).transpose();
    }
  }
  Eigen::FullPivLU<ZMatrix> lu(MIC);
  ensure_lu(lu, where);   // DY-S10 (tier 2)
  XIC = lu.solve(QIC);
  for(int i = 1; i <= k_; i++) {
    err += (ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) - ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose()).norm();
    ZMatrixMap(G.curr_timestep_les_ptr(0,i), nao_, nao_) = ZMatrixMap(XIC.data() + (i-1)*es_, nao_, nao_).transpose();
  }

  // rows m = 1..k and, with RHO_DIAGONAL, the diagonal (converged to the fixed point of the pair, B34)
  start_les_rows_(G, mu, H, Sigma, I, h, false, err, where);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

} // namespace





















