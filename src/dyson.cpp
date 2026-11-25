#include "h_nessi/dyson.hpp"
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace std::chrono;

namespace h_nessi {

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


// DY-C1, DY-C2 (tier 2)
void dyson::check_dyson_ctor_(const char *where, int nt, int nao, int k, dlr_info &dlr) {
  HNESSI_REQUIRE(1 <= k && k <= KMAX, where, HNESSI_KV(k), HNESSI_KV(KMAX));
  HNESSI_REQUIRE(nt >= k+1, where, HNESSI_KV(nt), HNESSI_KV(k));
  HNESSI_REQUIRE(nao >= 1, where, HNESSI_KV(nao));
  HNESSI_REQUIRE(dlr.size1() == nao && dlr.size2() == nao, where, HNESSI_KV(dlr.size1()), HNESSI_KV(nao),
                 detail::note{"the dlr_info must be built for the same number of orbitals"});
}

dyson::dyson(int nt, int nao, int k, dlr_info &dlr, bool rho_version, bool profile) :
dlr_(dlr)
 {
  check_dyson_ctor_("dyson::dyson", nt, nao, k, dlr);
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
  DG0_.resize(r_ * es_);
  DG0ConvMat_.resize(r_ * r_ * es_);
  epsnao_tmp_.resize((nt_+1)/2 * nao_);
  epsnao_tmp_2_.resize((nt_+1)/2 * nao_);

  Q_.resize((nt_+1) * es_);
  X_.resize((nt_+1) * es_);
  M_.resize((k+1) * k * es_);
  iden_.resize(es_);
  bound_.resize(es_);

  ZMatrixMap(iden_.data(), nao_, nao_).noalias() = ZMatrix::Identity(nao_, nao_);

  profile_ = profile;
  timing = DMatrix::Zero(profile_ ? nt : 0, 25);
  herm_diag_.assign(nt_+1, 0.0);
}

std::vector<double> dyson::memory_usage_bytes(void) {

    // every heap buffer of the class, counted from its actual size
    double D = (double)sizeof(double);
    double Z = (double)sizeof(cplx);

    double tautmp = 0;
    double g0 = 0;
    double g0conv = 0;
    double epsnaotmp = 0;
    double qxm = 0;
    double tim = 0;

    tautmp = Z * (NTauTmp_.size() + NTauTmp2_.size()) + D * (DNTauTmp_.size() + DNTauTmp2_.size());
    g0 = D * DG0_.size();
    g0conv = D * DG0ConvMat_.size();
    epsnaotmp = Z * (epsnao_tmp_.size() + epsnao_tmp_2_.size());
    qxm = Z * (Q_.size() + X_.size() + M_.size() + iden_.size() + bound_.size());
    tim = D * timing.rows() * timing.cols() + D * herm_diag_.size();   // timing + Hermiticity diagnostic (D-11)

    return {tautmp, g0, g0conv, epsnaotmp, qxm, tim, tautmp + g0 + g0conv + epsnaotmp + qxm + tim};
}

void dyson::print_memory_usage(void) {

    std::vector<double> mem = memory_usage_bytes();
    for(double &m : mem) m /= 1e9;

    const int total_width = 10;
    const int precision = 2;

    std::cout << "==========================================" << std::endl;
    std::cout << "==        Memory usage for dyson        ==" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "==                                      ==" << std::endl;
    printf(      "==  DLR buffers        :%*.*f MB   ==", total_width, precision, mem[0]*1000); std::cout << std::endl;
    printf(      "==  G0 (dyson_mat)     :%*.*f MB   ==", total_width, precision, mem[1]*1000); std::cout << std::endl;
    printf(      "==  G0 conv. matrix    :%*.*f MB   ==", total_width, precision, mem[2]*1000); std::cout << std::endl;
    printf(      "==  Mat Mul buffers    :%*.*f MB   ==", total_width, precision, mem[3]*1000); std::cout << std::endl;
    printf(      "==  Solver buffers     :%*.*f MB   ==", total_width, precision, mem[4]*1000); std::cout << std::endl;
    printf(      "==  Timing/diagnostics :%*.*f MB   ==", total_width, precision, mem[5]*1000); std::cout << std::endl;
    std::cout << "==                                      ==" << std::endl;
    printf(      "==  Total              :%*.*f MB   ==", total_width, precision, mem[6]*1000); std::cout << std::endl;
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
  // DY-C3 (tier 2, without the Q5 xi/r/beta comparison)
  check_dyson_ctor_("dyson::dyson(h5e::File&,label,dlr)", nt_, nao_, k_, dlr);
  HNESSI_REQUIRE(es_ == nao_*nao_, "dyson::dyson(h5e::File&,label,dlr)", HNESSI_KV(es_), HNESSI_KV(nao_));

  r_ = dlr.r();
  ntau_ = dlr.r();
  beta_ = dlr.beta();
  xi_ = dlr.xi();

  rho_version_ = in.getDataSet(label + std::string("/rho_version")).read<bool>();

  NTauTmp_.resize(r_ * es_);
  NTauTmp2_.resize(r_ * es_);
  DNTauTmp_.resize(r_ * es_);
  DNTauTmp2_.resize(r_ * es_);
  DG0_.resize(r_ * es_);
  DG0ConvMat_.resize(r_ * r_ * es_);
  epsnao_tmp_.resize((nt_+1)/2 * nao_);
  epsnao_tmp_2_.resize((nt_+1)/2 * nao_);

  Q_.resize((nt_+1) * es_);
  X_.resize((nt_+1) * es_);
  M_.resize((k_+1) * k_ * es_);
  iden_.resize(es_);
  bound_.resize(es_);

  ZMatrixMap(iden_.data(), nao_, nao_).noalias() = ZMatrix::Identity(nao_, nao_);

  profile_ = in.getDataSet(label + std::string("/profile")).read<bool>();
  timing = profile_ ? in.getDataSet(label + std::string("/timing")).read<DMatrix>() : DMatrix::Zero(0,25);
  HNESSI_REQUIRE(!profile_ || (timing.rows() == nt_ && timing.cols() == 25), "dyson::dyson(h5e::File&,label,dlr)",
                 HNESSI_KV(timing.rows()), HNESSI_KV(timing.cols()), HNESSI_KV(nt_));
  herm_diag_.assign(nt_+1, 0.0);
  // ST-3: the saved G0 (DG0_, DG0ConvMat_) is not checkpointed, so g0_valid_ stays false
  g0_valid_ = false;
}

// =====================================================================================================================
// Entry checks (WP6, doc 61 §2.11-2.14). They run once, in the outermost public call (call_scope); the dispatchers
// below call public components, which then skip them.
// =====================================================================================================================

// SOLV(G,Sigma,I) + DY-E8 (doc 61 §1.4): geometry of G and Sigma, pairing with this dyson (and so with its dlr_info)
void dyson::check_solver_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma,
                               Integration::Integrator &I, double h, double mu) {
  HNESSI_REQUIRE(G.size1() == nao_ && G.size2() == nao_, where, HNESSI_KV(G.size1()), HNESSI_KV(G.size2()),
                 HNESSI_KV(nao_));
  HNESSI_REQUIRE(G.sig() == xi_ && G.r() == r_, where, HNESSI_KV(G.sig()), HNESSI_KV(xi_), HNESSI_KV(G.r()),
                 HNESSI_KV(r_), detail::note{"G must use the statistics and DLR rank of the dlr_info of this dyson"});
  HNESSI_REQUIRE(G.k() == k_ && I.k() == k_, where, HNESSI_KV(G.k()), HNESSI_KV(I.k()), HNESSI_KV(k_));
  HNESSI_REQUIRE(G.nt() <= nt_, where, HNESSI_KV(G.nt()), HNESSI_KV(nt_),
                 detail::note{"the dyson object was constructed for fewer time steps"});
  HNESSI_REQUIRE(Sigma.nt() == G.nt() && Sigma.nlvl() == G.nlvl() && Sigma.k() == G.k() &&
                 Sigma.size1() == G.size1() && Sigma.size2() == G.size2() && Sigma.r() == G.r() &&
                 Sigma.sig() == G.sig(), where,
                 HNESSI_KV(Sigma.nt()), HNESSI_KV(G.nt()), HNESSI_KV(Sigma.nlvl()), HNESSI_KV(G.nlvl()),
                 HNESSI_KV(Sigma.k()), HNESSI_KV(G.k()), HNESSI_KV(Sigma.size1()), HNESSI_KV(G.size1()),
                 HNESSI_KV(Sigma.r()), HNESSI_KV(G.r()), HNESSI_KV(Sigma.sig()), HNESSI_KV(G.sig()),
                 detail::note{"G and Sigma must have the same geometry"});
  HNESSI_REQUIRE(std::isfinite(h) && h > 0 && std::isfinite(mu), where, HNESSI_KV(h), HNESSI_KV(mu));
}

// DY-T1, DY-T2: G and Sigma advanced by update_blocks exactly once for this step
void dyson::check_step_(const char *where, int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  HNESSI_REQUIRE(k_ < tstp && tstp < G.nt(), where, HNESSI_KV(tstp), HNESSI_KV(k_), HNESSI_KV(G.nt()),
                 detail::note{"time steps after the bootstrap are k < tstp < nt"});
  HNESSI_REQUIRE_STATE(tstp == G.tstpmk()+k_, where, HNESSI_KV(tstp), HNESSI_KV(G.tstpmk()), HNESSI_KV(k_),
                       detail::note{"call G.update_blocks(I) exactly once per time step before solving step tstp"});
  HNESSI_REQUIRE_STATE(tstp == Sigma.tstpmk()+k_, where, HNESSI_KV(tstp), HNESSI_KV(Sigma.tstpmk()), HNESSI_KV(k_),
                       detail::note{"call Sigma.update_blocks(I) exactly once per time step"});
}

// DY-S1: the bootstrap works on the windows 0..k of G and Sigma
void dyson::check_boot_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma) {
  HNESSI_REQUIRE_STATE(G.tstpmk() == 0 && Sigma.tstpmk() == 0, where, HNESSI_KV(G.tstpmk()),
                       HNESSI_KV(Sigma.tstpmk()),
                       detail::note{"the bootstrap needs the windows of G and Sigma at rows 0..k (no update_blocks yet)"});
}

// DY-E9: a function h(t) holds the slices -1..tlast and is nao x nao
void dyson::check_function_(const char *where, function &H, int tlast) {
  HNESSI_REQUIRE(H.size1() == nao_ && H.size2() == nao_, where, HNESSI_KV(H.size1()), HNESSI_KV(H.size2()),
                 HNESSI_KV(nao_));
  HNESSI_REQUIRE(H.nt() >= tlast+1, where, HNESSI_KV(H.nt()), HNESSI_KV(tlast),
                 detail::note{"h(t) must be defined for every slice the call reads"});
}

// DY-P1: a NaN/Inf anywhere in the new G propagates into the returned error
double dyson::ensure_finite_(double err, const char *where) {
  HNESSI_ENSURE(std::isfinite(err), where, HNESSI_KV(err),
                detail::note{"the solution is not finite (singular system, or NaN/Inf in h, Sigma or G)"});
  return err;
}

// D-11: slices t0..t1 of h (slice t at H0 + t*es_, t = -1 allowed). Each slice is checked first (read only) and
// throws above herm_err_tol_ before anything is modified; then, if enabled, it is replaced by (h+h^dagger)/2. Entries
// that are already Hermitian are not written (no race when threads share an h that is already Hermitian), and
// (a+conj(conj(a)))/2 == a exactly, so a slice symmetrised once is never changed again.
double dyson::hermitize_(cplx *H0, int t0, int t1, const char *where) {
  double maxrel = 0;
  int tworst = t0;
  for(int t = t0; t <= t1; t++) {
    cplx *hp = H0 + (std::ptrdiff_t)t * es_;
    double dn = 0, hn = 0;
    for(int i = 0; i < nao_; i++) {
      cplx d = hp[i*nao_+i];
      hn += std::norm(d);
      dn += 4. * d.imag() * d.imag();
      for(int j = i+1; j < nao_; j++) {
        cplx a = hp[i*nao_+j], b = hp[j*nao_+i];
        hn += std::norm(a) + std::norm(b);
        dn += 2. * std::norm(a - std::conj(b));
      }
    }
    double rel = dn == 0 ? 0. : std::sqrt(dn / hn);
    if(t >= -1 && t+1 < (int)herm_diag_.size()) herm_diag_[t+1] = std::max(herm_diag_[t+1], rel);
    if(rel > maxrel) { maxrel = rel; tworst = t; }
    HNESSI_REQUIRE(rel <= herm_err_tol_, where, HNESSI_KV(rel), HNESSI_KV(t), HNESSI_KV(herm_err_tol_),
                   detail::note{"h(t) is not Hermitian: rel = ||h(t)-h(t)^dagger||_F/||h(t)||_F (see "
                                "dyson::set_hermiticity_tolerances)"});
    if(herm_symmetrize_ && dn != 0) {
      for(int i = 0; i < nao_; i++) {
        cplx &d = hp[i*nao_+i];
        if(d.imag() != 0) d = cplx(d.real(), 0.);
        for(int j = i+1; j < nao_; j++) {
          cplx &a = hp[i*nao_+j], &b = hp[j*nao_+i];
          if(a != std::conj(b)) {
            cplx m = 0.5 * (a + std::conj(b));
            a = m;
            b = std::conj(m);
          }
        }
      }
    }
  }
  herm_last_ = maxrel;
  if(maxrel > herm_warn_tol_ && !herm_warned_) {
    herm_warned_ = true;
    std::ostringstream msg;
    msg << "h_nessi::" << where << ": h(t) has a relative anti-Hermitian part " << maxrel << " at t=" << tworst
        << " (> herm_warn_tol=" << herm_warn_tol_ << ")"
        << (herm_symmetrize_ ? "; h(t) is replaced in place by (h+h^dagger)/2" : "; h(t) is used as is (check only)")
        << ". Warned once per dyson object; see dyson::antihermiticity_per_slice().";
    detail::warn(msg.str());
  }
  return maxrel;
}

// D-11 for a real h(-1) (dyson_mat DMatrix overload): symmetric part (h+h^T)/2
double dyson::hermitize_real_(DMatrix &H, const char *where) {
  HNESSI_REQUIRE(H.rows() == nao_ && H.cols() == nao_, where, HNESSI_KV(H.rows()), HNESSI_KV(H.cols()),
                 HNESSI_KV(nao_));
  double dn = 0, hn = H.squaredNorm();
  for(int i = 0; i < nao_; i++)
    for(int j = i+1; j < nao_; j++) dn += 2. * (H(i,j)-H(j,i)) * (H(i,j)-H(j,i));
  double rel = dn == 0 ? 0. : std::sqrt(dn / hn);
  if(!herm_diag_.empty()) herm_diag_[0] = std::max(herm_diag_[0], rel);
  herm_last_ = rel;
  HNESSI_REQUIRE(rel <= herm_err_tol_, where, HNESSI_KV(rel), HNESSI_KV(herm_err_tol_),
                 detail::note{"h(-1) is not symmetric: rel = ||h-h^T||_F/||h||_F"});
  if(rel > herm_warn_tol_ && !herm_warned_) {
    herm_warned_ = true;
    std::ostringstream msg;
    msg << "h_nessi::" << where << ": h(-1) has a relative antisymmetric part " << rel << " (> herm_warn_tol="
        << herm_warn_tol_ << ")" << (herm_symmetrize_ ? "; h is replaced in place by (h+h^T)/2" : "")
        << ". Warned once per dyson object.";
    detail::warn(msg.str());
  }
  if(herm_symmetrize_ && dn != 0) {
    for(int i = 0; i < nao_; i++)
      for(int j = i+1; j < nao_; j++) { double m = 0.5*(H(i,j)+H(j,i)); H(i,j) = m; H(j,i) = m; }
  }
  return rel;
}

// DY-M4 (D1, decided: throw): G^M is stored real, so the Hermitian part of h(-1) must be real
void dyson::require_real_(const cplx *hp, const char *where) {
  double in = 0, hn = 0;
  for(int i = 0; i < nao_; i++)
    for(int j = 0; j < nao_; j++) {
      double im = 0.5 * (hp[i*nao_+j].imag() - hp[j*nao_+i].imag());  // Im of (h+h^dagger)/2
      in += im * im;
      hn += std::norm(hp[i*nao_+j]);
    }
  double rel = in == 0 ? 0. : std::sqrt(in / hn);
  HNESSI_REQUIRE(rel <= 1e-12, where, HNESSI_KV(rel),
                 detail::note{"h(-1) must be real (rel = ||Im h||_F/||h||_F): G^M is stored as a real function, a "
                              "complex h(-1) would be truncated (ISSUES D1)"});
}

void dyson::set_hermiticity_tolerances(double warn_tol, double err_tol) {
  HNESSI_REQUIRE(warn_tol >= 0 && err_tol >= 0 && warn_tol <= err_tol, "dyson::set_hermiticity_tolerances",
                 HNESSI_KV(warn_tol), HNESSI_KV(err_tol));
  herm_warn_tol_ = warn_tol;
  herm_err_tol_ = err_tol;
}

// all checks of a time-step entry point; H points to slice 0, slices hlo..hhi are read
void dyson::enter_timestep_(const char *where, int tstp, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma,
                            Integration::Integrator &I, double h, double mu, cplx *H, int hlo, int hhi, bool uses_tv) {
  check_solver_args_(where, G, Sigma, I, h, mu);
  check_step_(where, tstp, G, Sigma);
  if(uses_tv) G.require_gmconv_current(where);   // ST-2 (DY-T5)
  if(hlo <= hhi) hermitize_(H, hlo, hhi, where);  // D-11 (DY-T4)
}

// all checks of a bootstrap entry point (slices 0..k of h are read when reads_h)
void dyson::enter_start_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma, Integration::Integrator &I,
                         double h, double mu, cplx *H, bool reads_h, bool uses_tv) {
  check_solver_args_(where, G, Sigma, I, h, mu);
  check_boot_(where, G, Sigma);
  if(uses_tv) G.require_gmconv_current(where);   // ST-2 (DY-S5)
  if(reads_h) hermitize_(H, 0, k_, where);       // D-11 (DY-S3)
}

// =====================================================================================================================
// Time-step dispatchers
// =====================================================================================================================

double dyson::dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep";
  if(cs.outer()) {
    check_function_(where, H, tstp);
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H.ptr(0), 0, tstp, true);
  }
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

std::vector<double> dyson::dyson_timestep_errinfo(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep_errinfo";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, true);
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  std::vector<double> ret(4);

  ret[1] = dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  ret[2] = dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  ret[3] = dyson_timestep_les(tstp,G,mu,H,Sigma,I,h);

  ret[0] = ret[1] + ret[2] + ret[3];
  if(cs.outer()) ensure_finite_(ret[0], where);
  return ret;
}

double dyson::dyson_timestep(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, true);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les(tstp,G,mu,H,Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep_nobc";
  if(cs.outer()) {
    check_function_(where, H, tstp);
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H.ptr(0), 0, tstp, true);
  }
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_ret(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_nobc(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_nobc(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep_nobc";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, true);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_tv(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les_nobc(tstp,G,mu,H,Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep_2leg";
  if(cs.outer()) {
    check_function_(where, H, tstp);
    enter_timestep_(where, tstp, G, Sigma, I, h, mu, H.ptr(0), 0, tstp, false);
  }
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_ret(tstp,G,mu,H.ptr(0),Sigma,I,h);
  err += dyson_timestep_les_2leg(tstp,G,mu,H.ptr(0),Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_timestep_2leg(int tstp, herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h){
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_timestep_2leg";
  if(cs.outer()) enter_timestep_(where, tstp, G, Sigma, I, h, mu, H, 0, tstp, false);
  double err=0.0;
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  err += dyson_timestep_ret(tstp,G,mu,H,Sigma,I,h);
  err += dyson_timestep_les_2leg(tstp,G,mu,H,Sigma,I,h);

  return cs.outer() ? ensure_finite_(err, where) : err;
}

// =====================================================================================================================
// Bootstrap sub-steps (function& overloads) and dispatchers
// =====================================================================================================================

double dyson::dyson_start_tv(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_tv";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, true); }
  double err = dyson_start_tv(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ret(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ret";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), false, false); }
  double err = dyson_start_ret(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ret_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ret_ntti";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  double err = dyson_start_ret_ntti(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_les(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_les";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  double err = dyson_start_les(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_les_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_les_ntti";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  double err = dyson_start_les_ntti(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_les_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_les_ntti_nobc";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  double err = dyson_start_les_ntti_nobc(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_les_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_les_2leg";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  double err = dyson_start_les_2leg(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}



double dyson::dyson_start(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, true);
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  // We enforce Time-Translational Invariance for the first k timesteps

  // check TTI of H
  for(int t = 1; t <= k_; t++) {
    double diff = (ZMatrixMap(H, nao_, nao_) - ZMatrixMap(H + t*es_, nao_, nao_)).norm();
    HNESSI_REQUIRE(diff <= 1e-13, "dyson::dyson_start", HNESSI_KV(t), HNESSI_KV(diff),
                   detail::note{"H is not TTI for 0 <= t <= k: ||H(t)-H(0)||_F must be <= 1e-13"});
  }

  // check TTI of Sigma Les
  for(int tp = 0; tp <= k_; tp++) {
    for(int t = 0; t <= tp; t++) {
      double diff = (ZMatrixMap(Sigma.curr_timestep_les_ptr(0,tp-t), nao_, nao_) - ZMatrixMap(Sigma.curr_timestep_les_ptr(t,tp), nao_, nao_)).norm();
      HNESSI_REQUIRE(diff <= 1e-13, "dyson::dyson_start", HNESSI_KV(t), HNESSI_KV(tp), HNESSI_KV(diff),
                     detail::note{"Sigma^< is not TTI for 0 <= t <= k"});
    }
  }

  // check TTI of Sigma Ret
  for(int t = 0; t <= k_; t++) {
    for(int tp = 0; tp <= t; tp++) {
      double diff = (ZMatrixMap(Sigma.curr_timestep_ret_ptr(t-tp,0), nao_, nao_) - ZMatrixMap(Sigma.curr_timestep_ret_ptr(t,tp), nao_, nao_)).norm();
      HNESSI_REQUIRE(diff <= 1e-13, "dyson::dyson_start", HNESSI_KV(t), HNESSI_KV(tp), HNESSI_KV(diff),
                     detail::note{"Sigma^R is not TTI for 0 <= t <= k"});
    }
  }


  double err = 0;
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret(G, mu, H, Sigma, I, h);
  err += dyson_start_les(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, true); }
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  double err = dyson_start(G, mu, H.ptr(0), Sigma, I, h);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ntti(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ntti";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, true);
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ntti(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ntti";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, true); }
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  double err = dyson_start_ntti(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ntti_nobc";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, true);
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_tv(G, mu, H, Sigma, I, h);
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_ntti_nobc(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_ntti_nobc(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_ntti_nobc";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, true); }
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  double err = dyson_start_ntti_nobc(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_2leg(herm_matrix_hodlr &G, double mu, cplx *H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_2leg";
  if(cs.outer()) enter_start_(where, G, Sigma, I, h, mu, H, true, false);
  G.can_extrap() = false;
  Sigma.can_extrap() = false;

  double err = 0;
  err += dyson_start_ret_ntti(G, mu, H, Sigma, I, h, imp_tp0);
  err += dyson_start_les_2leg(G, mu, H, Sigma, I, h);
  G.set_tstpmk(0);
  
  return cs.outer() ? ensure_finite_(err, where) : err;
}

double dyson::dyson_start_2leg(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, Integration::Integrator &I, double h, bool imp_tp0) {
  call_scope cs(api_depth_);
  const char *where = "dyson::dyson_start_2leg";
  if(cs.outer()) { check_function_(where, H, k_); enter_start_(where, G, Sigma, I, h, mu, H.ptr(0), true, false); }
  G.can_extrap() = false;
  Sigma.can_extrap() = false;
  double err = dyson_start_2leg(G, mu, H.ptr(0), Sigma, I, h, imp_tp0);
  return cs.outer() ? ensure_finite_(err, where) : err;
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

} // namespace
