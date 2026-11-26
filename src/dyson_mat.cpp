#include "h_nessi/dyson.hpp"
#include <iomanip>
#include <cmath>

using namespace std::chrono;

namespace h_nessi {

// DY-M1, DY-M2, DY-M6, ST-3 (tier 2)
void dyson::check_mat_args_(const char *where, herm_matrix_hodlr &G, herm_matrix_hodlr &Sigma, double mu, double alpha,
                            bool fixHam) {
    HNESSI_REQUIRE(G.size1() == nao_ && G.size2() == nao_, where, HNESSI_KV(G.size1()), HNESSI_KV(G.size2()),
                   HNESSI_KV(nao_));
    HNESSI_REQUIRE(G.sig() == xi_ && G.r() == r_, where, HNESSI_KV(G.sig()), HNESSI_KV(xi_), HNESSI_KV(G.r()),
                   HNESSI_KV(r_), detail::note{"G must use the statistics and DLR rank of the dlr_info of this dyson"});
    HNESSI_REQUIRE(Sigma.size1() == nao_ && Sigma.size2() == nao_ && Sigma.sig() == xi_ && Sigma.r() == r_, where,
                   HNESSI_KV(Sigma.size1()), HNESSI_KV(Sigma.size2()), HNESSI_KV(Sigma.sig()), HNESSI_KV(Sigma.r()),
                   HNESSI_KV(nao_), HNESSI_KV(xi_), HNESSI_KV(r_));
    HNESSI_REQUIRE(0 < alpha && alpha <= 1 && std::isfinite(mu), where, HNESSI_KV(alpha), HNESSI_KV(mu));
    HNESSI_REQUIRE_STATE(!fixHam || g0_valid_, where, HNESSI_KV(fixHam), HNESSI_KV(g0_valid_),
                         detail::note{"fixHam=true reuses the G0 of an earlier fixHam=false call on this dyson object; "
                                      "there is none (fresh object, or restored from a checkpoint)"});
}

// L9 (tier 2): libdlr's dyson_it ignores the LAPACK info, a singular system gives Inf/NaN in G^M
void dyson::ensure_mat_finite_(const char *where) {
    bool finite = DMatrixMap(DNTauTmp_.data(), 1, r_*es_).allFinite();
    HNESSI_ENSURE(finite, where, HNESSI_KV(finite),
                  detail::note{"the Matsubara Dyson solve (libdlr dyson_it) returned a non-finite G^M; G is unchanged"});
}

double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, function &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    call_scope cs(*this);
    const char *where = "dyson::dyson_mat";
    if(cs.outer()) {
      check_mat_args_(where, G, Sigma, mu, alpha, fixHam);
      if(!fixHam) {
        check_function_(where, H, -1);
        hermitize_(H.ptr(0), -1, -1, where);   // D-11 (DY-M3), in place on slice -1
        require_real_(H.ptr(-1), where);       // DY-M4 (D1)
      }
    }
    // G0 and its convolution matrix live in dyson-owned buffers (DG0_, DG0ConvMat_), so that fixHam=true can reuse
    // them and neither Sigma nor G.GMConvTensMatrix_ is used as scratch
    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H.get_map(-1));
      DMatrixMap(DG0_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DG0_.data(),DG0ConvMat_.data());
      g0_valid_ = true;   // ST-3
    }

    DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DG0_.data(), DG0ConvMat_.data(), DNTauTmp_.data(), DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();
    ensure_mat_finite_(where);   // L9 / DY-M8: before G is modified

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();

    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return cs.outer() ? ensure_finite_(ret, where) : ret;
}


double dyson::dyson_mat(herm_matrix_hodlr &G, double mu, DMatrix &H, herm_matrix_hodlr &Sigma, bool fixHam, double alpha){
    call_scope cs(*this);
    const char *where = "dyson::dyson_mat";
    if(cs.outer()) {
      check_mat_args_(where, G, Sigma, mu, alpha, fixHam);
      if(!fixHam) hermitize_real_(H, where);   // D-11 (DY-M3, DY-M5), in place
    }
    // G0 and its convolution matrix live in dyson-owned buffers (DG0_, DG0ConvMat_), so that fixHam=true can reuse
    // them and neither Sigma nor G.GMConvTensMatrix_ is used as scratch
    if(!fixHam) {
      dyson::green_from_H_mat(DNTauTmp2_.data(),mu,H);
      DMatrixMap(DG0_.data(), es_, ntau_).noalias() = DMatrixMap(DNTauTmp2_.data(), ntau_, es_).transpose();
      c_dlr_convmat(&r_,&nao_,dlr_.it2cf(),dlr_.it2cfp(),dlr_.phi(),DG0_.data(),DG0ConvMat_.data());
      g0_valid_ = true;   // ST-3
    }

    DMatrixMap(DNTauTmp_.data(), es_, ntau_).noalias() = DMatrixMap(Sigma.mat(), ntau_, es_).transpose();

    c_dyson_it(&r_, &nao_, dlr_.it2cf(), dlr_.it2cfp(), dlr_.phi(), DG0_.data(), DG0ConvMat_.data(), DNTauTmp_.data(), DNTauTmp2_.data());
    DMatrixMap(DNTauTmp_.data(), ntau_, es_).noalias() = DMatrixMap(DNTauTmp2_.data(), es_, ntau_).transpose();
    ensure_mat_finite_(where);   // L9 / DY-M8: before G is modified

    double ret = (DMatrixMap(DNTauTmp_.data(), 1, r_*es_) - DMatrixMap(G.mat(), 1, r_*es_)).norm();
    DMatrixMap(G.mat(), 1, r_*es_) = (1-alpha) * DMatrixMap(G.mat(), 1, r_*es_) + alpha * DMatrixMap(DNTauTmp_.data(), 1, r_*es_);

    return cs.outer() ? ensure_finite_(ret, where) : ret;
}

} // namespace
