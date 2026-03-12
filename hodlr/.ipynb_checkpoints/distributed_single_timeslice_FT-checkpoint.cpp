#include "distributed_single_timeslice_FT.hpp"

namespace hodlr {
/* #######################################################################################
#
#   CONSTRUCTION/DESTRUCTION
#
########################################################################################*/

// max_block_size is usually (2*nt+ntau)*size1*size2, edge case is when nt is small
distributed_single_timeslice_FT::distributed_single_timeslice_FT(int nk, int nt, int ntau, int size1, int size2, bool mpi, int order_k, int sig, int ndim) : 
data_(nk, std::max(2*nt+ntau, (order_k+2)*(order_k+1) + (order_k+1)*ntau)*size1*size2, mpi),
data2_(nk, std::max(2*nt+ntau, (order_k+2)*(order_k+1) + (order_k+1)*ntau)*size1*size2, mpi) {
  nk_ = nk;
  nt_ = nt;
  ntau_ = ntau;
  size1_ = size1;
  size2_ = size2;
  current_timestep_ = -1;
  sig_ = sig;
  order_k_ = order_k;
  ndim_ = ndim;

  taustart_ = 0;
  tauend_ = 0;
  tstart_ = 0;
  tend_ = 0;
  tdis_list_ = std::vector<int>(data_.ntasks(),0);
  tlen_list_ = std::vector<int>(data_.ntasks(),0);
  taudis_list_ = std::vector<int>(data_.ntasks(),0);
  taulen_list_ = std::vector<int>(data_.ntasks(),0);
  
  local_k_indices_ = std::vector<int>(data_.numblock_rank());
  for(int i = 0; i < local_k_indices_.size(); i++) {
    local_k_indices_[i] = data_.firstblock_rank() + i;
  }

  int nomp;
  #pragma omp parallel
  {
    nomp = omp_get_num_threads();
  }
  fftw_plan_with_nthreads(nomp);
  fftw_plan_with_nthreads(1);

  int howmany = 1;
  int istride = 1;
  int ostride = 1;
  int idist = 1;
  int odist = 1;
  if(ndim_ == 1) {
    // These will immediately be replaced in set_tstp, so make everything small
    int rank = 1;
    int n[] = {1};
    p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_r2k_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_r2k_tv = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    p_k2r_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    p_k2r_tv = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else if(ndim_ == 2) {
    // These will immediately be replaced in set_tstp, so make everything small
    int rank = 2;
    int n[] = {1, 1};
    p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_r2k_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_r2k_tv = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
    p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    p_k2r_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    p_k2r_tv = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  set_tstp(-1);
}

void distributed_single_timeslice_FT::FT_k2r() {
  if(current_timestep_ > order_k_) {
    fftw_execute(p_k2r_ret);
    fftw_execute(p_k2r_les);
    fftw_execute(p_k2r_tv);
//    ZMatrixMap(data2_.vec().data() + tstart_*size1_*size2_*nk_, (tend_-tstart_+1) * size1_ * size2_ * nk_, 1) /= nk_;
//    ZMatrixMap(data2_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_, (tend_-tstart_+1) * size1_ * size2_ * nk_, 1) /= nk_;
//    ZMatrixMap(data2_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_, (tauend_-taustart_+1) * size1_ * size2_ * nk_, 1) /= nk_;
  }
  else {
    fftw_execute(p_k2r_ret);
//    ZMatrixMap(data2_.vec().data(), data_.blocksize() * nk_, 1) /= nk_;
  }
}

void distributed_single_timeslice_FT::FT_r2k() {
  fftw_execute(p_r2k_ret);
  if(current_timestep_ > order_k_) {
    fftw_execute(p_r2k_les);
    fftw_execute(p_r2k_tv);
  }
}

//void distributed_single_timeslice_FT::Trans_Sout() {
//  ZMatrixMap(data_.data(), nk_, data_.blocksize()) = ZMatrixMap(data2_.data(), data_.blocksize(), nk_).transpose();
//}

void distributed_single_timeslice_FT::mpi_bcast_all() {
  data_.mpi_bcast_all_complex();
}

void distributed_single_timeslice_FT::mpi_sum() {
  if(data_.ntasks() > 1) {
    MPI_Allgatherv(MPI_IN_PLACE, tlen_list_[data_.tid()], MPI_CXX_DOUBLE_COMPLEX, retptr2_zab_S(current_timestep_,0,0,0), tlen_list_.data(), tdis_list_.data(), MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, tlen_list_[data_.tid()], MPI_CXX_DOUBLE_COMPLEX, lesptr2_zab_S(0,current_timestep_,0,0), tlen_list_.data(), tdis_list_.data(), MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, taulen_list_[data_.tid()], MPI_CXX_DOUBLE_COMPLEX, tvptr2_zab_S(current_timestep_,0,0,0), taulen_list_.data(), taudis_list_.data(), MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
  }
}

void distributed_single_timeslice_FT::mpi_sum_it() {
  if(data_.ntasks() > 1) {
    MPI_Allgatherv(MPI_IN_PLACE, taulen_list_[data_.tid()], MPI_CXX_DOUBLE_COMPLEX, tvptr2_zab_S(current_timestep_,0,0,0), taulen_list_.data(), taudis_list_.data(), MPI_CXX_DOUBLE_COMPLEX, MPI_COMM_WORLD);
  }
}


void distributed_single_timeslice_FT::set_tstp(int tstp) {
  assert(tstp >= -1 and tstp < nt_);
  int size = 0;

  if(tstp == -1) size = ntau_ * size1_ * size2_;
  else if(tstp <= order_k_) size = (2 * (order_k_+1) * (order_k_+2) / 2 + (order_k_+1)*ntau_) * size1_ * size2_;
  else                 size = (2 * (tstp+1) + ntau_) * size1_ * size2_;

  data_.reset_blocksize(size);
  data2_.reset_blocksize(size);
  current_timestep_ = tstp;

  fftw_destroy_plan(p_r2k_ret);
  fftw_destroy_plan(p_k2r_ret);

  // nothing special, everybody do every z-point
  if(current_timestep_ <= order_k_) {
    int howmany = data_.blocksize();
    int istride = data_.blocksize();
    int ostride = 1;
    int idist = 1;
    int odist = nk_;
    if(ndim_ == 1) {
      int rank = 1;
      int n[] = {nk_};
      p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    else if(ndim_ == 2) {
      int rank = 2;
      int L = (int) std::sqrt(nk_);
      int n[] = {L, L};
      p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data()), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data()), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
  else {
    fftw_destroy_plan(p_r2k_les);
    fftw_destroy_plan(p_k2r_les);
    fftw_destroy_plan(p_r2k_tv);
    fftw_destroy_plan(p_k2r_tv);

    // round robin schedule real-time points
    int smallest_allocation = (current_timestep_+1) / data_.ntasks();
    int extra_allocation = (current_timestep_+1) % data_.ntasks();
    int my_allocation = smallest_allocation + (data_.tid() < extra_allocation ? 1 : 0);
    tstart_ = 0;
    for(int i = 0; i < data_.tid(); i++) tstart_ += smallest_allocation + (i < extra_allocation ? 1 : 0);
    tend_ = tstart_ + my_allocation-1;
    int howmany = my_allocation * size1_ * size2_;

    // round robin schedule imaginary-time points
    int smallest_allocation_tau = ntau_ / data_.ntasks();
    int extra_allocation_tau = ntau_ % data_.ntasks();
    int my_allocation_tau = smallest_allocation_tau + (data_.tid() < extra_allocation_tau ? 1 : 0);
    taustart_ = 0;
    for(int i = 0; i < data_.tid(); i++) taustart_ += smallest_allocation_tau + (i < extra_allocation_tau ? 1 : 0);
    tauend_ = taustart_ + my_allocation_tau-1;
    int howmany_tau = my_allocation_tau * size1_ * size2_;

    tlen_list_[0] = smallest_allocation + (0 < extra_allocation ? 1 : 0);
    taulen_list_[0] = smallest_allocation_tau + (0 < extra_allocation_tau ? 1 : 0);
    for(int i = 1; i < data_.ntasks(); i++) {
      tdis_list_[i] = tdis_list_[i-1] + smallest_allocation + ((i-1) < extra_allocation ? 1 : 0);
      taudis_list_[i] = taudis_list_[i-1] + smallest_allocation_tau + ((i-1) < extra_allocation_tau ? 1 : 0);
      tlen_list_[i] = smallest_allocation + (i < extra_allocation ? 1 : 0);
      taulen_list_[i] = smallest_allocation_tau + (i < extra_allocation_tau ? 1 : 0);
    }
    for(int i = 0; i < data_.ntasks(); i++) {
      tdis_list_[i] *=   size1_*size2_*nk_; 
      taudis_list_[i] *= size1_*size2_*nk_; 
      tlen_list_[i] *=   size1_*size2_*nk_; 
      taulen_list_[i] *= size1_*size2_*nk_; 
    }

    int nomp;
    #pragma omp parallel
    {
      nomp = omp_get_num_threads();
    }
    fftw_plan_with_nthreads(nomp);
    fftw_plan_with_nthreads(1);


    // always the same no matter ndim_ or scheduling
    int istride = data_.blocksize();
    int ostride = 1;
    int idist = 1;
    int odist = nk_;

    if(ndim_ == 1) {
      int rank = 1;
      int n[] = {nk_};

      p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + tstart_*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + tstart_*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_r2k_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_r2k_tv = fftw_plan_many_dft(rank, n, howmany_tau, reinterpret_cast<fftw_complex*>(data_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + tstart_*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + tstart_*nk_*size1_*size2_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
      p_k2r_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (current_timestep_+1+tstart_)*nk_*size1_*size2_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
      p_k2r_tv = fftw_plan_many_dft(rank, n, howmany_tau, reinterpret_cast<fftw_complex*>(data_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (2*(current_timestep_+1)+taustart_)*nk_*size1_*size2_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    else if(ndim_ == 2) {
      int rank = 2;
      int L = (int) std::sqrt(nk_);
      int n[] = {L, L};

      p_r2k_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + tstart_*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + tstart_*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_r2k_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_r2k_tv = fftw_plan_many_dft(rank, n, howmany_tau, reinterpret_cast<fftw_complex*>(data_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_), NULL, ostride, odist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
      p_k2r_ret = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + tstart_*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + tstart_*size1_*size2_*nk_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
      p_k2r_les = fftw_plan_many_dft(rank, n, howmany, reinterpret_cast<fftw_complex*>(data_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (current_timestep_+1+tstart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
      p_k2r_tv = fftw_plan_many_dft(rank, n, howmany_tau, reinterpret_cast<fftw_complex*>(data_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_), NULL, istride, idist, reinterpret_cast<fftw_complex*>(data2_.vec().data() + (2*(current_timestep_+1)+taustart_)*size1_*size2_*nk_), NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
  }
}


// ######################################################################
// ######################################################################
// ##                       Set from pointers                          ##
// ######################################################################
// ######################################################################

void distributed_single_timeslice_FT::set_mat(int k_index, double *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());
  assert(current_timestep_ == -1);

  ZMatrixMap(matptr(k_index, 0), ntau_*size1_*size2_, 1) = DMatrixMap(data, ntau_*size1_*size2_, 1);
}

void distributed_single_timeslice_FT::set_ret(int k_index, cplx *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  int size = (current_timestep_+1)*size1_*size2_;

  std::memcpy(retptr(k_index, current_timestep_, 0), data, size * sizeof(std::complex<double>));
}

void distributed_single_timeslice_FT::set_ret_boot(int k_index, herm_matrix_hodlr &G) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  for(int t = 0; t <= order_k_; t++) {
    std::memcpy(retptr(k_index, t, 0), G.curr_timestep_ret_ptr(t,0), (t+1)*(size1_*size2_) * sizeof(std::complex<double>));
  }
}

void distributed_single_timeslice_FT::set_les(int k_index, cplx *data) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  int size = (current_timestep_+1)*size1_*size2_;

  std::memcpy(lesptr(k_index, 0, current_timestep_), data, size * sizeof(std::complex<double>));
}

void distributed_single_timeslice_FT::set_les_boot(int k_index, herm_matrix_hodlr &G) {
  assert(std::find(local_k_indices_.begin(), local_k_indices_.end(), k_index) != local_k_indices_.end());

  for(int t = 0; t <= order_k_; t++) {
    std::memcpy(lesptr(k_index, 0, t), G.curr_timestep_les_ptr(0,t), (t+1)*(size1_*size2_) * sizeof(std::complex<double>));
  }
}

void distributed_single_timeslice_FT::set_tv(int k_index, cplx *data) {
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

void distributed_single_timeslice_FT::set_mat(int k_index, herm_matrix_hodlr &G) {
  set_mat(k_index, G.matptr(0));
}

void distributed_single_timeslice_FT::set_ret(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));
  
  if(current_timestep_ > order_k_) set_ret(k_index, G.curr_timestep_ret_ptr(current_timestep_,0));
  else if(current_timestep_ <= order_k_) set_ret_boot(k_index, G);
}

void distributed_single_timeslice_FT::set_les(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));

  if(current_timestep_ > order_k_) set_les(k_index, G.curr_timestep_les_ptr(0,current_timestep_));
  else if(current_timestep_ <= order_k_) set_les_boot(k_index, G);
}

void distributed_single_timeslice_FT::set_tv(int k_index, herm_matrix_hodlr &G) {
  assert(current_timestep_ == G.tstpmk() + G.k() or (current_timestep_ <= order_k_ and current_timestep_ >= 0 and G.tstpmk() == 0));
  
  int t_min = current_timestep_ <= order_k_ ? 0 : current_timestep_;

  set_tv(k_index, G.tvptr(t_min,0));
}

void distributed_single_timeslice_FT::set_all(int k_index, herm_matrix_hodlr &G) {
  set_ret(k_index, G);
  set_les(k_index, G);
  set_tv(k_index, G);
}

// ######################################################################
// ######################################################################
// ##                     Get Pointers (Szab)                          ##
// ######################################################################
// ######################################################################

cplx *distributed_single_timeslice_FT::matptr(int k_index, int tau) {
  return data_.data() + k_index * data_.blocksize() + tau * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::retptr(int k_index, int t, int tp) {
  int time_index = current_timestep_ > order_k_ ? tp : t*(t+1)/2 + tp;
  return data_.data() + k_index * data_.blocksize() + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::lesptr(int k_index, int t, int tp) {
  int ret_size = current_timestep_ > order_k_ ? (current_timestep_+1)*size1_*size2_ : ((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_index = current_timestep_ > order_k_ ? t : tp*(tp+1)/2 + t;
  return data_.data() + k_index * data_.blocksize() + ret_size + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::tvptr(int k_index, int t, int tau) {
  int retles_size = current_timestep_ > order_k_ ? 2*(current_timestep_+1)*size1_*size2_ : 2*((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_offset = current_timestep_ > order_k_ ? 0 : t*ntau_;

  return data_.data() + k_index * data_.blocksize() + retles_size + time_offset*size1_*size2_ + tau * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::matptr2(int k_index, int tau) {
  return data2_.data() + k_index * data_.blocksize() + tau * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::retptr2(int k_index, int t, int tp) {
  int time_index = current_timestep_ > order_k_ ? tp : t*(t+1)/2 + tp;
  return data2_.data() + k_index * data_.blocksize() + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::lesptr2(int k_index, int t, int tp) {
  int ret_size = current_timestep_ > order_k_ ? (current_timestep_+1)*size1_*size2_ : ((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_index = current_timestep_ > order_k_ ? t : tp*(tp+1)/2 + t;
  return data2_.data() + k_index * data_.blocksize() + ret_size + time_index * size1_ * size2_;
}

cplx *distributed_single_timeslice_FT::tvptr2(int k_index, int t, int tau) {
  int retles_size = current_timestep_ > order_k_ ? 2*(current_timestep_+1)*size1_*size2_ : 2*((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_;
  int time_offset = current_timestep_ > order_k_ ? 0 : t*ntau_;

  return data2_.data() + k_index * data_.blocksize() + retles_size + time_offset*size1_*size2_ + tau * size1_ * size2_;
}


// ######################################################################
// ######################################################################
// ##                     Get Pointers (zabS)                          ##
// ######################################################################
// ######################################################################


cplx *distributed_single_timeslice_FT::matptr_zab_S(int tau, int a, int b) {
  return data_.data() + tau * nk_ * size1_ * size2_ + a * nk_ * size1_ + b * nk_;
}

cplx *distributed_single_timeslice_FT::retptr_zab_S(int t, int tp, int a, int b) {
  int time_index = current_timestep_ > order_k_ ? tp : t*(t+1)/2 + tp;
  return data_.data() + time_index * nk_ * size1_ * size2_ + a*nk_*size1_+b*nk_;
}

cplx *distributed_single_timeslice_FT::lesptr_zab_S(int t, int tp, int a, int b) {
  int ret_size = current_timestep_ > order_k_ ? (current_timestep_+1)*size1_*size2_*nk_ : ((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_*nk_;
  int time_index = current_timestep_ > order_k_ ? t : tp*(tp+1)/2 + t;
  return data_.data() + ret_size + time_index * size1_ * size2_*nk_ + a*nk_*size1_ + b*nk_;
}

cplx *distributed_single_timeslice_FT::tvptr_zab_S(int t, int tau, int a, int b) {
  int retles_size = current_timestep_ > order_k_ ? 2*(current_timestep_+1)*size1_*size2_*nk_ : 2*((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_ * nk_;
  int time_offset = current_timestep_ > order_k_ ? 0 : t*ntau_;

  return data_.data() + retles_size + time_offset*size1_*size2_*nk_ + tau * size1_ * size2_*nk_ + a*size1_*nk_ + b*nk_;
}

cplx *distributed_single_timeslice_FT::matptr2_zab_S(int tau, int a, int b) {
  return data2_.data() + tau * nk_ * size1_ * size2_ + a * nk_ * size1_ + b * nk_;
}

cplx *distributed_single_timeslice_FT::retptr2_zab_S(int t, int tp, int a, int b) {
  int time_index = current_timestep_ > order_k_ ? tp : t*(t+1)/2 + tp;
  return data2_.data() + time_index * nk_ * size1_ * size2_ + a*nk_*size1_+b*nk_;
}

cplx *distributed_single_timeslice_FT::lesptr2_zab_S(int t, int tp, int a, int b) {
  int ret_size = current_timestep_ > order_k_ ? (current_timestep_+1)*size1_*size2_*nk_ : ((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_*nk_;
  int time_index = current_timestep_ > order_k_ ? t : tp*(tp+1)/2 + t;
  return data2_.data() + ret_size + time_index * size1_ * size2_*nk_ + a*nk_*size1_ + b*nk_;
}

cplx *distributed_single_timeslice_FT::tvptr2_zab_S(int t, int tau, int a, int b) {
  int retles_size = current_timestep_ > order_k_ ? 2*(current_timestep_+1)*size1_*size2_*nk_ : 2*((order_k_+1) * (order_k_+2) / 2) * size1_ * size2_ * nk_;
  int time_offset = current_timestep_ > order_k_ ? 0 : t*ntau_;

  return data2_.data() + retles_size + time_offset*size1_*size2_*nk_ + tau * size1_ * size2_*nk_ + a*size1_*nk_ + b*nk_;
}

// ######################################################################
// ######################################################################
// ##                         Matsubara Stuff                          ##
// ######################################################################
// ######################################################################

void distributed_single_timeslice_FT::get_mat_tau(double tau,double beta,double *it2cf,int *it2cfp, double *dlrrf, cplx *M, int k_point) {
  int es = size1_*size2_;
  int one = 1;

  double *Gij_it = new double[ntau_ * es];
  double *Gijc_it = new double[ntau_ * es];
  double *res = new double[es];

  DMatrixMap(Gij_it, es, ntau_).noalias() = ZMatrixMap(matptr(k_point, 0), ntau_, es).transpose().real();

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

void distributed_single_timeslice_FT::density_matrix(int tstp, double *it2cf, int *it2cfp, double *dlrrf, cplx *res, int k_point) {
  get_mat_tau(1.,1.,it2cf,it2cfp,dlrrf,res,k_point);
  ZMatrixMap(res, size1_, size2_) *= -1.;
}

void distributed_single_timeslice_FT::get_mat_reversed(double *res, double *it2itr, int k_point) {
  DMatrixMap(res, ntau_, size1_*size2_).noalias() = DMatrixMap(it2itr, ntau_, ntau_).transpose() * ZMatrixMap(matptr(k_point, 0), ntau_, size1_*size2_).real();
}

void distributed_single_timeslice_FT::get_vt_transpose(int tstp, cplx *res, double *it2itr, int k_point) {
  ZMatrixMap(res, ntau_, size1_*size2_).noalias() = -sig_ * (DMatrixMap(it2itr, ntau_, ntau_).transpose() * ZMatrixMap(tvptr(k_point, tstp, 0), ntau_, size1_*size2_)).conjugate();
}

}
