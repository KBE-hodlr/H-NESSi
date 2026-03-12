#include "distributed_single_timeslice.hpp"

namespace hodlr {
/* #######################################################################################
#
#   CONSTRUCTION/DESTRUCTION
#
########################################################################################*/

// max_block_size is usually (2*nt+ntau)*size1*size2, edge case is when nt is small
distributed_single_timeslice::distributed_single_timeslice(int nk, int nt, int ntau, int size1, int size2, bool mpi, int order_k, int sig) : data_(nk, std::max(2*nt+ntau, (order_k+2)*(order_k+1) + (order_k+1)*ntau)*size1*size2, mpi) {
  nk_ = nk;
  nt_ = nt;
  ntau_ = ntau;
  size1_ = size1;
  size2_ = size2;
  current_timestep_ = -1;
  sig_ = sig;
  order_k_ = order_k;
  
  local_k_indices_ = std::vector<int>(data_.numblock_rank());
  for(int i = 0; i < local_k_indices_.size(); i++) {
    local_k_indices_[i] = data_.firstblock_rank() + i;
  }
  set_tstp(-1);
}

void distributed_single_timeslice::mpi_bcast_all() {
  if(current_timestep_ == -1) data_.mpi_bcast_all_double();
  else data_.mpi_bcast_all_complex();
}


void distributed_single_timeslice::set_tstp(int tstp) {
  assert(tstp >= -1 and tstp < nt_);
  int size = 0;

  if(tstp == -1) size = ntau_ * size1_ * size2_;
  else if(tstp <= order_k_) size = (2 * (order_k_+1) * (order_k_+2) / 2 + (order_k_+1)*ntau_) * size1_ * size2_;
  else                 size = (2 * (tstp+1) + ntau_) * size1_ * size2_;

  data_.reset_blocksize(size);
  current_timestep_ = tstp;
}


// ######################################################################
// ######################################################################
// ##                       Set from pointers                          ##
// ######################################################################
// ######################################################################

void distributed_single_timeslice::set_mat(int k_index, double *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());
  assert(current_timestep_ == -1);

  std::memcpy(matptr(k_index, 0), data, ntau_*size1_*size2_ * sizeof(double));
}

void distributed_single_timeslice::set_ret(int k_index, cplx *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  int size = (current_timestep_+1)*size1_*size2_;

  std::memcpy(retptr(k_index, current_timestep_, 0), data, size * sizeof(std::complex<double>));
}

void distributed_single_timeslice::set_ret_boot(int k_index, herm_matrix_hodlr &G) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  for(int t = 0; t <= order_k_; t++) {
    std::memcpy(retptr(k_index, t, 0), G.curr_timestep_ret_ptr(t,0), (t+1)*(size1_*size2_) * sizeof(std::complex<double>));
  }
}

void distributed_single_timeslice::set_les(int k_index, cplx *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  int size = (current_timestep_+1)*size1_*size2_;

  std::memcpy(lesptr(k_index, 0, current_timestep_), data, size * sizeof(std::complex<double>));
}

void distributed_single_timeslice::set_les_boot(int k_index, herm_matrix_hodlr &G) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  for(int t = 0; t <= order_k_; t++) {
    std::memcpy(lesptr(k_index, 0, t), G.curr_timestep_les_ptr(0,t), (t+1)*(size1_*size2_) * sizeof(std::complex<double>));
  }
}

void distributed_single_timeslice::set_tv(int k_index, cplx *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  int size = current_timestep_ > order_k_ ? ntau_ * size1_ * size2_ : (order_k_+1) * ntau_ * size1_ * size2_;
  int t_min = current_timestep_ <= order_k_ ? 0 : current_timestep_;

  std::memcpy(tvptr(k_index, t_min, 0), data, size * sizeof(std::complex<double>));
}


// ######################################################################
// ######################################################################
// ##                    Set from herm_matrix                          ##
// ######################################################################
// ######################################################################

void distributed_single_timeslice::set_mat(int k_index, herm_matrix_hodlr &G) {
  set_mat(k_index, G.matptr(0));
}

void distributed_single_timeslice::set_ret(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));
  
  if(current_timestep_ > order_k_) set_ret(k_index, G.curr_timestep_ret_ptr(current_timestep_,0));
  else if(current_timestep_ <= order_k_) set_ret_boot(k_index, G);
}

void distributed_single_timeslice::set_les(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));

  if(current_timestep_ > order_k_) set_les(k_index, G.curr_timestep_les_ptr(0,current_timestep_));
  else if(current_timestep_ <= order_k_) set_les_boot(k_index, G);
}

void distributed_single_timeslice::set_tv(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));
  
  int t_min = current_timestep_ <= order_k_ ? 0 : current_timestep_;

  set_tv(k_index, G.tvptr(t_min,0));
}

void distributed_single_timeslice::set_all(int k_index, herm_matrix_hodlr &G) {
  set_ret(k_index, G);
  set_les(k_index, G);
  set_tv(k_index, G);
}

// ######################################################################
// ######################################################################
// ##                         Get Pointers                             ##
// ######################################################################
// ######################################################################

double *distributed_single_timeslice::matptr(int k_index, int tau) {
  double *double_data_pointer = reinterpret_cast<double*>(data_.data());
  return double_data_pointer + k_index * data_.blocksize() + tau * size1_ * size2_;
}

cplx *distributed_single_timeslice::retptr(int k_index, int t, int tp) {
  int time_index = current_timestep_ > order_k_ ? tp : t*(t+1)/2 + tp;
  return data_.data() + k_index * data_.blocksize() + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice::lesptr(int k_index, int t, int tp) {
  int ret_size = current_timestep_ > order_k_ ? (current_timestep_+1)*size1_*size2_ : ((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_index = current_timestep_ > order_k_ ? t : tp*(tp+1)/2 + t;
  return data_.data() + k_index * data_.blocksize() + ret_size + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice::tvptr(int k_index, int t, int tau) {
  int retles_size = current_timestep_ > order_k_ ? 2*(current_timestep_+1)*size1_*size2_ : 2*((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_offset = current_timestep_ > order_k_ ? 0 : t*ntau_;

  return data_.data() + k_index * data_.blocksize() + retles_size + time_offset*size1_*size2_ + tau * size1_ * size2_;
}

// ######################################################################
// ######################################################################
// ##                         Get Scalars                              ##
// ######################################################################
// ######################################################################

double distributed_single_timeslice::get_mat(int k_index, int tau, int i, int j) {
  assert(current_timestep_ == -1);
  assert(tau < ntau_);
  assert(tau >= 0);

  return matptr(k_index, tau)[i * size1_ + j];  
}

std::complex<double> distributed_single_timeslice::get_ret(int k_index, int t, int tp, int i, int j) {
  assert(t >= tp);
  assert(t == current_timestep_ or (current_timestep_ <= order_k_ and t <= order_k_));
  assert(t >= 0);
  assert(tp >= 0);

  return retptr(k_index, t, tp)[i * size1_ + j];  
}

std::complex<double> distributed_single_timeslice::get_les(int k_index, int t, int tp, int i, int j) {
  assert(tp >= t);
  assert(tp == current_timestep_ or (current_timestep_ <= order_k_ and tp <= order_k_));
  assert(t >= 0);
  assert(tp >= 0);

  return lesptr(k_index, t, tp)[i * size1_ + j];  
}

std::complex<double> distributed_single_timeslice::get_tv(int k_index, int t, int tau, int i, int j) {
  assert(t == current_timestep_ or (current_timestep_ <= order_k_ and t <= order_k_));
  assert(t >= 0);
  assert(tau >= 0);
  assert(tau < ntau_);

  return tvptr(k_index, t, tau)[i * size1_ + j];  
}

// ######################################################################
// ######################################################################
// ##                         Get Maps                                 ##
// ######################################################################
// ######################################################################

DMatrixMap distributed_single_timeslice::get_mat_map(int k_index, int tau) {
  assert(current_timestep_ == -1);
  assert(tau < ntau_);
  assert(tau >= 0);

  return DMatrixMap(matptr(k_index, tau), size1_, size2_);  
}

ZMatrixMap distributed_single_timeslice::get_ret_map(int k_index, int t, int tp) {
  assert(t >= tp);
  assert(t == current_timestep_ or (current_timestep_ <= order_k_ and t <= order_k_));
  assert(t >= 0);
  assert(tp >= 0);
  
  return ZMatrixMap(retptr(k_index, t, tp), size1_, size2_);
}

ZMatrixMap distributed_single_timeslice::get_les_map(int k_index, int t, int tp) {
  assert(tp >= t);
  assert(tp == current_timestep_ or (current_timestep_ <= order_k_ and tp <= order_k_));
  assert(t >= 0);
  assert(tp >= 0);

  return ZMatrixMap(lesptr(k_index, t, tp), size1_, size2_);  
}

ZMatrixMap distributed_single_timeslice::get_tv_map(int k_index, int t, int tau) {
  assert(t == current_timestep_ or (current_timestep_ <= order_k_ and t <= order_k_));
  assert(t >= 0);
  assert(tau >= 0);
  assert(tau < ntau_);

  return ZMatrixMap(tvptr(k_index, t, tau), size1_, size2_);  
}

void distributed_single_timeslice::get_les(int k_index, int t, int tp, cplx* res) {
  assert(tp >= t);
  assert(tp == current_timestep_ or (current_timestep_ <= order_k_ and tp <= order_k_));
  assert(t >= 0);
  assert(tp >= 0);

  ZMatrixMap(res, size1_, size2_) = ZMatrixMap(lesptr(k_index, t, tp), size1_, size2_);
}

// ######################################################################
// ######################################################################
// ##                         Matsubara Stuff                          ##
// ######################################################################
// ######################################################################

void distributed_single_timeslice::get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, cplx *M, int k_point) {
  int es = size1_*size2_;
  int one = 1;

  double *Gij_it = new double[ntau_ * es];
  double *Gijc_it = new double[ntau_ * es];
  double *res = new double[es];

  DMatrixMap(Gij_it, es, ntau_).noalias() = DMatrixMap(matptr(k_point, 0), ntau_, es).transpose();

  double tau01 = tau/beta;
  double taurel;

  c_dlr_it2cf(&ntau_, &size1_, it2cf, it2cfp, Gij_it, Gijc_it);
  c_abs2rel(&one, &tau01, &taurel);
  c_dlr_it_eval(&ntau_, &size1_, dlrrf, Gijc_it, &taurel, res);

  ZMatrixMap(M, size1_, size2_).noalias() = DMatrixMap(res, size1_, size2_);

  delete[] Gij_it;
  delete[] Gijc_it;
  delete[] res;
}

void distributed_single_timeslice::density_matrix(int tstp, double *it2cf, int *it2cfp, double *dlrrf, cplx *res, int k_point) {
  get_mat_tau(1.,1.,it2cf,it2cfp,dlrrf,res,k_point);
  ZMatrixMap(res, size1_, size2_) *= -1.;
}

void distributed_single_timeslice::get_mat_reversed(double *res, double *it2itr, int k_point) {
  DMatrixMap(res, ntau_, size1_*size2_).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * DMatrixMap(matptr(k_point, 0), ntau_, size1_*size2_);
}

void distributed_single_timeslice::get_vt_transpose(int tstp, cplx *res, double *it2itr, int k_point) {
  ZMatrixMap(res, ntau_, size1_*size2_).noalias() = -sig_ * (DMatrixMap(it2itr, ntau_, ntau_).transpose() * ZMatrixMap(tvptr(k_point, tstp, 0), ntau_, size1_*size2_)).conjugate();
}

}
