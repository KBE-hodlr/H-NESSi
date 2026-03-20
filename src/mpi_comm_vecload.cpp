#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/utils.hpp"

namespace h_nessi {

ZMatrixMap mpi_comm::map_ret(int k_global, int t_global) {
  int t_local = t_global - my_first_t;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(t_allk_buff.data() + t_allk_buff_index(t_local, 0, k_local, k_global), nao_, nao_);
}

ZMatrixMap mpi_comm::map_les(int k_global, int t_global){
  int t_local = t_global - my_first_t;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(t_allk_buff.data() + t_allk_buff_index(t_local, nao_*nao_, k_local, k_global), nao_, nao_);
}
ZMatrixMap mpi_comm::map_tv(int k_global, int tau_global){
  int tau_local = tau_global - my_first_tau;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(tau_allk_buff.data() + tau_allk_buff_index(tau_local, 0, k_local, k_global), nao_, nao_);
}
ZMatrixMap mpi_comm::map_tv_rev(int k_global, int tau_global){
  int tau_local = tau_global - my_first_tau;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(tau_allk_buff.data() + tau_allk_buff_index(tau_local, nao_*nao_, k_local, k_global), nao_, nao_);
}
ZMatrixMap mpi_comm::map_mat(int k_global, int tau_global) {
  int tau_local = tau_global - my_first_tau;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(tau_allk_buff.data() + tau_allk_buff_index(tau_local, 0, k_local, k_global), nao_, nao_);
}
ZMatrixMap mpi_comm::map_mat_rev(int k_global, int tau_global) {
  int tau_local = tau_global - my_first_tau;
  int mpi_index = k_rank_map[k_global];
  int k_local = k_global-init_k_per_rank[mpi_index];
  return ZMatrixMap(tau_allk_buff.data() + tau_allk_buff_index(tau_local, nao_*nao_, k_local, k_global), nao_, nao_);
}



void mpi_comm::mpi_get_and_comm_spawn
(int ti,
 std::vector<herm_matrix_hodlr> &hmh_vec,
 dlr_info &dlr
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> init_tp_per_rank(mpi_size);
    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    my_first_t = init_tp_per_rank[mpi_rank];
    my_Nt = Ntp_per_rank[mpi_rank];

    #pragma omp parallel for schedule(static)
    for(int tpi=0; tpi<tp_size; tpi++){
      for (int krank = 0; krank<my_Nk; krank++){
        // first retarded compi=0
        int index = alltp_buff_index(my_Nk,tpi,krank,0);
        std::copy_n(hmh_vec[krank].curr_timestep_ret_ptr(ti,tpi), nao_*nao_, t_alltp_buff.begin() + index);
        // then lesser compi=nao^2
        index = alltp_buff_index(my_Nk,tpi,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].curr_timestep_les_ptr(tpi,ti), nao_*nao_, t_alltp_buff.begin() + index);
      }
    }

    for(int mpii=0; mpii<mpi_size; mpii++){
      t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]    *my_Nk;
      t_recv_counts[mpii] = max_component_size_*Ntp_per_rank[mpi_rank]*Nk_per_rank[mpii];

      t_send_displs[mpii] = 0;
      t_recv_displs[mpii] = 0;
      for(int i = 0; i<mpii; i++){
        t_send_displs[mpii] = t_send_displs[mpii] + t_send_counts[i];
        t_recv_displs[mpii] = t_recv_displs[mpii] + t_recv_counts[i];
      }
    }

    MPI_Alltoallv(t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }

  if(ti >= 0) { // tau array is filled using tv
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      hmh_vec[krank].get_tv_reversed(ti, dlr, hmh_vec[krank].tvptr_trans(ti,0));
      for(int taui=0; taui<r_; taui++){
        // only do tv, so compi = 0
        int index = alltp_buff_index(my_Nk,taui,krank,0);
        std::copy_n(hmh_vec[krank].tvptr(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        // now do tv_rev, so compi = nao^2
        index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].tvptr_trans(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
      }
    }
  }
  else { // tau array filled using mat
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      hmh_vec[krank].get_mat_reversed(dlr, hmh_vec[krank].tvptr_trans(0,0));
      for(int taui=0; taui<r_; taui++){
        // only do mat, so compi = 0
        int index = alltp_buff_index(my_Nk,taui,krank,0);
        std::copy_n(hmh_vec[krank].matptr(taui), nao_*nao_, tau_alltp_buff.begin() + index);
        // now do mat_rev, so compi = nao^2
        index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].tvptr_trans(0,taui), nao_*nao_, tau_alltp_buff.begin() + index);
      }
    }
  }

  for(int mpii=0; mpii<mpi_size; mpii++){
    tau_send_counts[mpii] = max_component_size_*Ntau_per_rank[mpii]*Nk_per_rank[mpi_rank];
    tau_recv_counts[mpii] = max_component_size_*Ntau_per_rank[mpi_rank]*Nk_per_rank[mpii];

    tau_send_displs[mpii] = 0;
    tau_recv_displs[mpii] = 0;
    for(int i = 0; i<mpii; i++){
      tau_send_displs[mpii] = tau_send_displs[mpii] + tau_send_counts[i];
      tau_recv_displs[mpii] = tau_recv_displs[mpii] + tau_recv_counts[i];
    }
  }

  MPI_Alltoallv(tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

}

void mpi_comm::mpi_get_and_comm_spawn
(int ti,
 std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec,
 dlr_info &dlr
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> init_tp_per_rank(mpi_size);
    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    my_first_t = init_tp_per_rank[mpi_rank];
    my_Nt = Ntp_per_rank[mpi_rank];

    #pragma omp parallel for schedule(static)
    for(int tpi=0; tpi<tp_size; tpi++){
      for (int krank = 0; krank<my_Nk; krank++){
        // first retarded compi=0
        int index = alltp_buff_index(my_Nk,tpi,krank,0);
        std::copy_n(hmh_vec[krank].get().curr_timestep_ret_ptr(ti,tpi), nao_*nao_, t_alltp_buff.begin() + index);
        // then lesser compi=nao^2
        index = alltp_buff_index(my_Nk,tpi,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].get().curr_timestep_les_ptr(tpi,ti), nao_*nao_, t_alltp_buff.begin() + index);
      }
    }

    for(int mpii=0; mpii<mpi_size; mpii++){
      t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]    *my_Nk;
      t_recv_counts[mpii] = max_component_size_*Ntp_per_rank[mpi_rank]*Nk_per_rank[mpii];

      t_send_displs[mpii] = 0;
      t_recv_displs[mpii] = 0;
      for(int i = 0; i<mpii; i++){
        t_send_displs[mpii] = t_send_displs[mpii] + t_send_counts[i];
        t_recv_displs[mpii] = t_recv_displs[mpii] + t_recv_counts[i];
      }
    }

    MPI_Alltoallv(t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }


  if(ti >= 0) { // tau array is filled using tv
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      hmh_vec[krank].get().get_tv_reversed(ti, dlr, hmh_vec[krank].get().tvptr_trans(ti,0));
      for(int taui=0; taui<r_; taui++){
        // only do tv, so compi = 0
        int index = alltp_buff_index(my_Nk,taui,krank,0);
        std::copy_n(hmh_vec[krank].get().tvptr(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        // now do tv_rev, so compi = nao^2
        index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].get().tvptr_trans(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
      }
    }
  }
  else { // tau array filled using mat
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      hmh_vec[krank].get().get_mat_reversed(dlr, hmh_vec[krank].get().tvptr_trans(0,0));
      for(int taui=0; taui<r_; taui++){
        // only do mat, so compi = 0
        int index = alltp_buff_index(my_Nk,taui,krank,0);
        std::copy_n(hmh_vec[krank].get().matptr(taui), nao_*nao_, tau_alltp_buff.begin() + index);
        // now do mat_rev, so compi = nao^2
        index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
        std::copy_n(hmh_vec[krank].get().tvptr_trans(0,taui), nao_*nao_, tau_alltp_buff.begin() + index);
      }
    }
  }

  for(int mpii=0; mpii<mpi_size; mpii++){
    tau_send_counts[mpii] = max_component_size_*Ntau_per_rank[mpii]*Nk_per_rank[mpi_rank];
    tau_recv_counts[mpii] = max_component_size_*Ntau_per_rank[mpi_rank]*Nk_per_rank[mpii];

    tau_send_displs[mpii] = 0;
    tau_recv_displs[mpii] = 0;
    for(int i = 0; i<mpii; i++){
      tau_send_displs[mpii] = tau_send_displs[mpii] + tau_send_counts[i];
      tau_recv_displs[mpii] = tau_recv_displs[mpii] + tau_recv_counts[i];
    }
  }

  MPI_Alltoallv(tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

}

void mpi_comm::mpi_get_and_comm_nospawn
(int ti,
 std::vector<herm_matrix_hodlr> &hmh_vec,
 dlr_info &dlr
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> init_tp_per_rank(mpi_size);
    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    my_first_t = init_tp_per_rank[mpi_rank];
    my_Nt = Ntp_per_rank[mpi_rank];

    std::array<int,3> indxs = get_my_index(thread_id,nthreads,tp_size,0);
    int th_my_init_tp = indxs[0];
    int th_my_end_tp = indxs[1];
    int th_my_Ntp = indxs[2];

    if(th_my_Ntp != 0){
      for(int tpi=th_my_init_tp; tpi<=th_my_end_tp; tpi++){
        for (int krank = 0; krank<my_Nk; krank++){
          // first retarded compi=0
          int index = alltp_buff_index(my_Nk,tpi,krank,0);
          std::copy_n(hmh_vec[krank].curr_timestep_ret_ptr(ti,tpi), nao_*nao_, t_alltp_buff.begin() + index);
          // then lesser compi=nao^2
          index = alltp_buff_index(my_Nk,tpi,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].curr_timestep_les_ptr(tpi,ti), nao_*nao_, t_alltp_buff.begin() + index);
        }
      }
    }

    #pragma omp barrier

    if(thread_id==0){
      for(int mpii=0; mpii<mpi_size; mpii++){
        t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]    *my_Nk;
        t_recv_counts[mpii] = max_component_size_*Ntp_per_rank[mpi_rank]*Nk_per_rank[mpii];
  
        t_send_displs[mpii] = 0;
        t_recv_displs[mpii] = 0;
        for(int i = 0; i<mpii; i++){
          t_send_displs[mpii] = t_send_displs[mpii] + t_send_counts[i];
          t_recv_displs[mpii] = t_recv_displs[mpii] + t_recv_counts[i];
        }
      }
  
      MPI_Alltoallv(t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    MPI_COMM_WORLD);
    }

    #pragma omp barrier
  }


  // here we split up over local k instead of r. so we can easily calculate reverse
  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);
  int th_my_init_k = indxs[0];
  int th_my_end_k = indxs[1];
  int th_my_Nk = indxs[2];

  if(th_my_Nk != 0){
    if(ti >= 0) { // tau array is filled using tv
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        hmh_vec[krank].get_tv_reversed(ti, dlr, hmh_vec[krank].tvptr_trans(ti,0));
        for(int taui=0; taui<r_; taui++){
          // only do tv, so compi = 0
          int index = alltp_buff_index(my_Nk,taui,krank,0);
          std::copy_n(hmh_vec[krank].tvptr(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
          // now do tv_rev, so compi = nao^2
          index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].tvptr_trans(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        }
      }
    }
    else { // tau array is filled using mat
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        hmh_vec[krank].get_mat_reversed(dlr, hmh_vec[krank].tvptr_trans(0,0));
        for(int taui=0; taui<r_; taui++){
          // only do mat, so compi = 0
          int index = alltp_buff_index(my_Nk,taui,krank,0);
          std::copy_n(hmh_vec[krank].matptr(taui), nao_*nao_, tau_alltp_buff.begin() + index);
          // now do mat_rev, so compi = nao^2
          index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].tvptr_trans(0,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        }
      }
    }
  }

  #pragma omp barrier

  if(thread_id==0){
    for(int mpii=0; mpii<mpi_size; mpii++){
      tau_send_counts[mpii] = max_component_size_*Ntau_per_rank[mpii]*Nk_per_rank[mpi_rank];
      tau_recv_counts[mpii] = max_component_size_*Ntau_per_rank[mpi_rank]*Nk_per_rank[mpii];
  
      tau_send_displs[mpii] = 0;
      tau_recv_displs[mpii] = 0;
      for(int i = 0; i<mpii; i++){
        tau_send_displs[mpii] = tau_send_displs[mpii] + tau_send_counts[i];
        tau_recv_displs[mpii] = tau_recv_displs[mpii] + tau_recv_counts[i];
      }
    }
  
    MPI_Alltoallv(tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }

  #pragma omp barrier
}

void mpi_comm::mpi_get_and_comm_nospawn
(int ti,
 std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec,
 dlr_info &dlr
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> init_tp_per_rank(mpi_size);
    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    my_first_t = init_tp_per_rank[mpi_rank];
    my_Nt = Ntp_per_rank[mpi_rank];

    std::array<int,3> indxs = get_my_index(thread_id,nthreads,tp_size,0);
    int th_my_init_tp = indxs[0];
    int th_my_end_tp = indxs[1];
    int th_my_Ntp = indxs[2];

    if(th_my_Ntp != 0){
      for(int tpi=th_my_init_tp; tpi<=th_my_end_tp; tpi++){
        for (int krank = 0; krank<my_Nk; krank++){
          // first retarded compi=0
          int index = alltp_buff_index(my_Nk,tpi,krank,0);
          std::copy_n(hmh_vec[krank].get().curr_timestep_ret_ptr(ti,tpi), nao_*nao_, t_alltp_buff.begin() + index);
          // then lesser compi=nao^2
          index = alltp_buff_index(my_Nk,tpi,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].get().curr_timestep_les_ptr(tpi,ti), nao_*nao_, t_alltp_buff.begin() + index);
        }
      }
    }

    #pragma omp barrier

    if(thread_id==0){
      for(int mpii=0; mpii<mpi_size; mpii++){
        t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]    *my_Nk;
        t_recv_counts[mpii] = max_component_size_*Ntp_per_rank[mpi_rank]*Nk_per_rank[mpii];
  
        t_send_displs[mpii] = 0;
        t_recv_displs[mpii] = 0;
        for(int i = 0; i<mpii; i++){
          t_send_displs[mpii] = t_send_displs[mpii] + t_send_counts[i];
          t_recv_displs[mpii] = t_recv_displs[mpii] + t_recv_counts[i];
        }
      }
  
      MPI_Alltoallv(t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    MPI_COMM_WORLD);
    }

    #pragma omp barrier
  }


  // here we split up over local k instead of r. so we can easily calculate reverse
  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);
  int th_my_init_k = indxs[0];
  int th_my_end_k = indxs[1];
  int th_my_Nk = indxs[2];

  if(th_my_Nk != 0){
    if(ti >= 0) { // tau array is filled using tv
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        hmh_vec[krank].get().get_tv_reversed(ti, dlr, hmh_vec[krank].get().tvptr_trans(ti,0));
        for(int taui=0; taui<r_; taui++){
          // only do tv, so compi = 0
          int index = alltp_buff_index(my_Nk,taui,krank,0);
          std::copy_n(hmh_vec[krank].get().tvptr(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
          // now do tv_rev, so compi = nao^2
          index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].get().tvptr_trans(ti,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        }
      }
    }
    else { // tau array is filled using mat
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        hmh_vec[krank].get().get_mat_reversed(dlr, hmh_vec[krank].get().tvptr_trans(0,0));
        for(int taui=0; taui<r_; taui++){
          // only do mat, so compi = 0
          int index = alltp_buff_index(my_Nk,taui,krank,0);
          std::copy_n(hmh_vec[krank].get().matptr(taui), nao_*nao_, tau_alltp_buff.begin() + index);
          // now do mat_rev, so compi = nao^2
          index = alltp_buff_index(my_Nk,taui,krank,nao_*nao_);
          std::copy_n(hmh_vec[krank].get().tvptr_trans(0,taui), nao_*nao_, tau_alltp_buff.begin() + index);
        }
      }
    }
  }

  #pragma omp barrier

  if(thread_id==0){
    for(int mpii=0; mpii<mpi_size; mpii++){
      tau_send_counts[mpii] = max_component_size_*Ntau_per_rank[mpii]*Nk_per_rank[mpi_rank];
      tau_recv_counts[mpii] = max_component_size_*Ntau_per_rank[mpi_rank]*Nk_per_rank[mpii];
  
      tau_send_displs[mpii] = 0;
      tau_recv_displs[mpii] = 0;
      for(int i = 0; i<mpii; i++){
        tau_send_displs[mpii] = tau_send_displs[mpii] + tau_send_counts[i];
        tau_recv_displs[mpii] = tau_recv_displs[mpii] + tau_recv_counts[i];
      }
    }
  
    MPI_Alltoallv(tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }

  #pragma omp barrier
}








void mpi_comm::mpi_comm_and_set_spawn
(int ti,
 std::vector<herm_matrix_hodlr> &hmh_vec
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }

    MPI_Alltoallv(t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(tp_size * nao_*nao_);
      //first we do ret, which is compi=0
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
          int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].curr_timestep_ret_ptr(ti,0));
      //next we do les, which is compi=nao^2
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
          int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, nao_*nao_);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].curr_timestep_les_ptr(0,ti));
    }
  }


  MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

  if(ti >= 0) { // fill with tv
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
          int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), r_ * nao_*nao_, hmh_vec[krank].tvptr(ti,0));
    }
  }
  else { // fill with mat
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
          int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
        }
      }
      // cant use copy_n because we are copying into a real array
      for(int i = 0; i < r_ * nao_*nao_; i++) hmh_vec[krank].matptr(0)[i] = Stmp[i].real();
    }
  }
};

void mpi_comm::mpi_comm_and_set_spawn
(int ti,
 std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }

    MPI_Alltoallv(t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(tp_size * nao_*nao_);
      //first we do ret, which is compi=0
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
          int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].get().curr_timestep_ret_ptr(ti,0));
      //next we do les, which is compi=nao^2=nao_*nao_
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
          int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, nao_*nao_);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].get().curr_timestep_les_ptr(0,ti));
    }
  }


  MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

  if(ti >= 0) { // fill with tv
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
          int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
        }
      }
      std::copy_n(Stmp.begin(), r_ * nao_*nao_, hmh_vec[krank].get().tvptr(ti,0));
    }
  }
  else { // fill with mat
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
          int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
          // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
          std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
        }
      }
      // cant use copy_n because we are copying into a real array
      for(int i = 0; i < r_ * nao_*nao_; i++) hmh_vec[krank].get().matptr(0)[i] = Stmp[i].real();
    }
  }
};


void mpi_comm::mpi_comm_and_set_nospawn
(int ti,
 std::vector<herm_matrix_hodlr> &hmh_vec
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }

    #pragma omp barrier

    if(thread_id==0){
      MPI_Alltoallv(t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    MPI_COMM_WORLD);
    }

    #pragma omp barrier

    std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);

    int th_my_init_k = indxs[0];
    int th_my_end_k = indxs[1];
    int th_my_Nk = indxs[2];


    if(th_my_Nk != 0){
      std::vector<std::complex<double>> Stmp(tp_size * nao_*nao_);
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        //first we do ret, which is compi=0
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
            int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].curr_timestep_ret_ptr(ti,0));
        //next we do les, which is compi=nao^2
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
            int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, nao_*nao_);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].curr_timestep_les_ptr(0,ti));
      }
    }
  }

  #pragma omp barrier

  if(thread_id==0){
    MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }

  #pragma omp barrier

  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);

  int th_my_init_k = indxs[0];
  int th_my_end_k = indxs[1];
  int th_my_Nk = indxs[2];


  if(th_my_Nk != 0) {
    if(ti >= 0) { // fill with tv
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
            int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), r_ * nao_*nao_, hmh_vec[krank].tvptr(ti,0));
      }
    }
    else { // fill with mat
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
            int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
          }
        }
        // cant use copy_n because we are copying into a real array
        for(int i = 0; i < r_ * nao_*nao_; i++) hmh_vec[krank].matptr(0)[i] = Stmp[i].real();
      }
    }
  }
};

void mpi_comm::mpi_comm_and_set_nospawn
(int ti,
 std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }

    #pragma omp barrier

    if(thread_id==0){
      MPI_Alltoallv(t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                    MPI_COMM_WORLD);
    }

    #pragma omp barrier

    std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);

    int th_my_init_k = indxs[0];
    int th_my_end_k = indxs[1];
    int th_my_Nk = indxs[2];


    if(th_my_Nk != 0){
      std::vector<std::complex<double>> Stmp(tp_size * nao_*nao_);
      for(int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        //first we do ret, which is compi=0
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
            int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].get().curr_timestep_ret_ptr(ti,0));
        //next we do les, which is compi=nao^2
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){
            int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, nao_*nao_);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(t_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tp_per_rank[mpi_id]+tpi)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), tp_size * nao_*nao_, hmh_vec[krank].get().curr_timestep_les_ptr(0,ti));
      }
    }
  }

  #pragma omp barrier

  if(thread_id==0){
    MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);
  }

  #pragma omp barrier

  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Nk,0);

  int th_my_init_k = indxs[0];
  int th_my_end_k = indxs[1];
  int th_my_Nk = indxs[2];


  if(th_my_Nk != 0) {
    if(ti >= 0) { // fill with tv
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
            int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
          }
        }
        std::copy_n(Stmp.begin(), r_ * nao_*nao_, hmh_vec[krank].get().tvptr(ti,0));
      }
    }
    else { // fill with mat
      std::vector<std::complex<double>> Stmp(r_ * nao_*nao_);
      for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){
            int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, 0);
            // index has max_component_size built in, the init_tp_per_rank[mpi_id] does not
            std::copy_n(tau_alltp_buff.begin() + index, nao_*nao_, Stmp.begin() + (init_tau_per_rank[mpi_id]+taui)*nao_*nao_);
          }
        }
        // cant use copy_n because we are copying into a real array
        for(int i = 0; i < r_ * nao_*nao_; i++) hmh_vec[krank].get().matptr(0)[i] = Stmp[i].real();
      }
    }
  }
};

} // namespace 
