#include <vector>
#include <iostream>
#include <string>

#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/checks.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace h_nessi {

// dlr pairing (tier 2, HM-I1..HM-I6): the dlr_info evaluates with its own r and orbital size, and G is indexed with
// r_, size1_, size2_; a mismatch reads or writes out of bounds.
void herm_matrix_hodlr::check_dlr_(const char *where, const dlr_info &dlr) const {
  HNESSI_REQUIRE(dlr.r() == r_ && dlr.size1() == size1_ && dlr.size2() == size2_, where, HNESSI_KV(dlr.r()),
                 HNESSI_KV(r_), HNESSI_KV(dlr.size1()), HNESSI_KV(size1_),
                 detail::note{"the dlr_info must have the DLR rank and orbital size of this object"});
}

// HM-I1 (tier 2): 0 <= tau <= beta. For tau < 0 the DLR expansion gives G(tau+beta) without the xi factor (doc 03
// §4.3); a few ulps above beta are accepted (tau computed as m*beta/n).
void herm_matrix_hodlr::check_tau_(const char *where, double tau, const dlr_info &dlr) const {
  HNESSI_REQUIRE(tau >= 0 && tau <= dlr.beta() * (1 + 1e-14), where, HNESSI_KV(tau), HNESSI_KV(dlr.beta()),
                 detail::note{"the imaginary time must lie in [0, beta]"});
}

// Eigen output of the evaluators: not resized, so it must hold at least n entries
#define HNESSI_REQUIRE_OUT_SIZE_(where, M, n)                                                                       \
  HNESSI_REQUIRE((M).size() >= (Eigen::Index)(n), where, HNESSI_KV((M).size()), HNESSI_KV(n),                        \
                 detail::note{"the output matrix is not resized and must be allocated by the caller"})

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, DMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_mat_tau", tau, dlr);
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::get_mat_tau", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, double *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_mat_tau", tau, dlr);
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_mat_tau", tau, dlr);
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::get_mat_tau", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  dlr.eval_point(tau, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau(double tau, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_mat_tau", tau, dlr);
  dlr.eval_point(tau, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_tv_tau", tau, dlr);
  check_row_("herm_matrix_hodlr::get_tv_tau", tstp);
  HNESSI_REQUIRE(M.rows() == size1_ && M.cols() == size2_, "herm_matrix_hodlr::get_tv_tau", HNESSI_KV(M.rows()),
                 HNESSI_KV(M.cols()), HNESSI_KV(size1_));
  dlr.eval_point(tau, tvptr(tstp, 0), M.data());
}

void herm_matrix_hodlr::get_tv_tau(int tstp, double tau, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau", dlr);
  check_tau_("herm_matrix_hodlr::get_tv_tau", tau, dlr);
  check_row_("herm_matrix_hodlr::get_tv_tau", tstp);
  dlr.eval_point(tau, tvptr(tstp, 0), M);
}





void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, DMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_mat_tau_array", taus(i), dlr);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_mat_tau_array", M, taus.size()*size1_*size2_);
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, double *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_mat_tau_array", taus(i), dlr);
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_mat_tau_array", taus(i), dlr);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_mat_tau_array", M, taus.size()*size1_*size2_);
  dlr.eval_point(taus, matptr(0), M.data());
}

void herm_matrix_hodlr::get_mat_tau_array(DColVector taus, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_mat_tau_array", dlr);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_mat_tau_array", taus(i), dlr);
  dlr.eval_point(taus, matptr(0), M);
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, ZMatrix &M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau_array", dlr);
  check_row_("herm_matrix_hodlr::get_tv_tau_array", tstp);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_tv_tau_array", taus(i), dlr);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_tv_tau_array", M, taus.size()*size1_*size2_);
  dlr.eval_point(taus, tvptr(tstp,0), M.data());
}

void herm_matrix_hodlr::get_tv_tau_array(int tstp, DColVector taus, dlr_info &dlr, cplx *M){
  check_dlr_("herm_matrix_hodlr::get_tv_tau_array", dlr);
  check_row_("herm_matrix_hodlr::get_tv_tau_array", tstp);
  for(Eigen::Index i = 0; i < taus.size(); i++) check_tau_("herm_matrix_hodlr::get_tv_tau_array", taus(i), dlr);
  dlr.eval_point(taus, tvptr(tstp,0), M);
}



// return (r,size1,size2) array where data[n,i,j] = G^M[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, double *M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    DMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, std::complex<double> *M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
    ZMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, DMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_mat_reversed", M, r_*size1_*size2_);   // HM-I4
    DMatrixMap(M.data(), r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * DMatrixMap(mat_.data(), r_, size1_*size2_);
}
// (i,j) overloads (HM-I4): one orbital element, M holds r entries
#define HNESSI_REQUIRE_IJ_(where, i, j)                                                                             \
  HNESSI_REQUIRE(0 <= (i) && (i) < size1_ && 0 <= (j) && (j) < size2_, where, HNESSI_KV(i), HNESSI_KV(j),          \
                 HNESSI_KV(size1_), HNESSI_KV(size2_))

void herm_matrix_hodlr::get_mat_reversed(dlr_info &dlr, DMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_mat_reversed", dlr);
  HNESSI_REQUIRE_IJ_("herm_matrix_hodlr::get_mat_reversed", i, j);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_mat_reversed", M, r_);
    int ij_flat = i * size2_ + j;
    
    Eigen::Map<const DMatrix, 0, Eigen::InnerStride<>> mat_vec(
        mat_.data() + ij_flat,
        r_,
        1,
        Eigen::InnerStride<>(size1_ * size2_)
    );
    
    DMatrixMap(M.data(), r_, 1) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * mat_vec;
}


// return (r,size1,size2) array where data[n,i,j] = G^tv[beta-tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, cplx *M) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
  check_row_("herm_matrix_hodlr::get_tv_reversed", tstp);   // HM-I5
  ZMatrixMap(M, r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr(tstp, 0), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
  check_row_("herm_matrix_hodlr::get_tv_reversed", tstp);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_tv_reversed", M, r_*size1_*size2_);
  ZMatrixMap(M.data(), r_, size1_*size2_) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * ZMatrixMap(tvptr(tstp, 0), r_, size1_*size2_);
}
void herm_matrix_hodlr::get_tv_reversed(int tstp, dlr_info &dlr, ZMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_tv_reversed", dlr);
  check_row_("herm_matrix_hodlr::get_tv_reversed", tstp);
  HNESSI_REQUIRE_IJ_("herm_matrix_hodlr::get_tv_reversed", i, j);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_tv_reversed", M, r_);
    int ij_flat = i * size2_ + j;
    
    Eigen::Map<const ZMatrix, 0, Eigen::InnerStride<>> tv_vec(
        tvptr(tstp, 0) + ij_flat,
        r_,
        1,
        Eigen::InnerStride<>(size1_ * size2_)
    );
    
    ZMatrixMap(M.data(), r_, 1) = DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * tv_vec;
}


// return (r,size1,size2) array where data[n,i,j] = G^vt[tau[n],i,j]
// tau[n] is the n^{th} dlr node on the imaginary time axis
// G^vt_{ij}(tau_n, t) = -xi [G^tv_{ji}(t, beta-tau_n)]^*  (doc 03 §4.10), read from tv directly (no transposed copy
// since WP6c, ISSUES H27): data(n,i,j) = -xi conj(sum_a it2itr_{an} tv(t,a,j,i)), one Eigen Tensor contraction.
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, cplx *M) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
  check_row_("herm_matrix_hodlr::get_vt", tstp);   // HM-I6
  using IndexPair1 = Eigen::array<Eigen::IndexPair<int>, 1>;
  Eigen::TensorMap<const Eigen::Tensor<double, 2, Eigen::RowMajor>> itr(dlr.it2itr(), r_, r_);          // (a,n)
  Eigen::TensorMap<const Eigen::Tensor<cplx, 3, Eigen::RowMajor>> tv(tvptr(tstp, 0), r_, size1_, size2_);  // (a,j,i)
  Eigen::TensorMap<Eigen::Tensor<cplx, 3, Eigen::RowMajor>> vt(M, r_, size2_, size1_);                    // (n,i,j)
  // itr(a,n) tv(a,j,i) -> (n,j,i) -> (n,i,j)
  vt = itr.cast<cplx>().contract(tv, IndexPair1{{{0,0}}}).shuffle(Eigen::array<int,3>{{0,2,1}}).conjugate() *
       cplx(-sig_);
}
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, ZMatrix &M) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
  check_row_("herm_matrix_hodlr::get_vt", tstp);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_vt", M, r_*size1_*size2_);
  get_vt(tstp, dlr, M.data());
}
// single element (i,j): data(n) = -xi conj(sum_a it2itr_{an} tv(t,a,j,i)), a strided read of tv (no transpose needed)
void herm_matrix_hodlr::get_vt(int tstp, dlr_info &dlr, ZMatrix &M, int i, int j) {
  check_dlr_("herm_matrix_hodlr::get_vt", dlr);
  check_row_("herm_matrix_hodlr::get_vt", tstp);
  HNESSI_REQUIRE_IJ_("herm_matrix_hodlr::get_vt", i, j);
  HNESSI_REQUIRE_OUT_SIZE_("herm_matrix_hodlr::get_vt", M, r_);
  Eigen::Map<const ZMatrix, 0, Eigen::InnerStride<>> tv_ji(tvptr(tstp, 0) + j * size2_ + i, r_, 1,
                                                           Eigen::InnerStride<>(size1_ * size2_));
  ZMatrixMap(M.data(), r_, 1) = -sig_ * (DMatrixConstMap(dlr.it2itr(), r_, r_).transpose() * tv_ji).conjugate();
}

} // namespace
