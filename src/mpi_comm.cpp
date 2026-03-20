#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/utils.hpp"

namespace h_nessi {

mpi_comm::mpi_comm(){};

mpi_comm::mpi_comm
(int Nk, 
 int Nt,
 int r,
 int nao,
 int ncomponents
):Nk_(Nk), Nt_(Nt), r_(r), max_component_size_(ncomponents*nao*nao), nao_(nao)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  tau_send_displs = std::vector<int>(mpi_size);
  tau_send_counts = std::vector<int>(mpi_size);
  tau_recv_displs = std::vector<int>(mpi_size);
  tau_recv_counts = std::vector<int>(mpi_size);

  t_send_displs = std::vector<int>(mpi_size);
  t_send_counts = std::vector<int>(mpi_size);
  t_recv_displs = std::vector<int>(mpi_size);
  t_recv_counts = std::vector<int>(mpi_size);
  
  k_rank_map = std::vector<int>(Nk_);

  init_k_per_rank = std::vector<int>(mpi_size);
  std::vector<int> end_k_per_rank(mpi_size);
  Nk_per_rank = std::vector<int>(mpi_size);

  for(int mpii=0; mpii<mpi_size; mpii++){
    std::array<int,3> k_mpi_indxs = get_my_index(mpii,mpi_size,Nk_,0);
    init_k_per_rank[mpii] = k_mpi_indxs[0];
    end_k_per_rank[mpii] = k_mpi_indxs[1];
    Nk_per_rank[mpii] = k_mpi_indxs[2];
  }

  for(int mpii=0; mpii<mpi_size; mpii++){
    if(Nk_per_rank[mpii]!=0){
      for(int ki=0; ki<Nk_; ki++){
        if((init_k_per_rank[mpii] <= ki) && (ki <= end_k_per_rank[mpii]) ){
          k_rank_map[ki] = mpii;
        }
      }
    }
  }

  kindex_rank = std::vector<int>(Nk_per_rank[mpi_rank]);

  int iq=0;
  for(int k=0;k<Nk;k++){
    if(k_rank_map[k]==mpi_rank){
      kindex_rank[iq]=k;        
      iq++;
    }
  }

  init_tau_per_rank = std::vector<int>(mpi_size);
  std::vector<int> end_tau_per_rank(mpi_size);
  Ntau_per_rank = std::vector<int>(mpi_size);

  for(int mpii=0; mpii<mpi_size; mpii++){
    std::array<int,3> tau_mpi_indxs = get_my_tauindex(mpii,mpi_size,r_);
    init_tau_per_rank[mpii] = tau_mpi_indxs[0];
    end_tau_per_rank[mpii] = tau_mpi_indxs[1];
    Ntau_per_rank[mpii] = tau_mpi_indxs[2];
  }
  my_Ntau = Ntau_per_rank[mpi_rank];
  my_first_tau = init_tau_per_rank[mpi_rank];
  my_Nk = Nk_per_rank[mpi_rank];
  my_global_k_list = std::vector<int>(my_Nk);
  for(int k = 0; k < my_Nk; k++) {
    my_global_k_list[k] = init_k_per_rank[mpi_rank] + k;
  }

  int max_Ntp = 0;
  for(int ti=0; ti<Nt_; ti++){
    std::array<int,3> tps_mpi_indxs = get_my_index(mpi_rank,mpi_size,ti+1,0);
    if(tps_mpi_indxs[2] > max_Ntp){
      max_Ntp = tps_mpi_indxs[2];
    }
  }

  t_alltp_buff = std::vector<std::complex<double>>(Nk_per_rank[mpi_rank]*Nt_*max_component_size_);
  t_allk_buff = std::vector<std::complex<double>>(Nk_*max_Ntp*max_component_size_);

  int max_Ntau = Ntau_per_rank[mpi_rank];

  tau_alltp_buff = std::vector<std::complex<double>>(Nk_per_rank[mpi_rank]*r_*max_component_size_);
  tau_allk_buff = std::vector<std::complex<double>>(Nk_*max_Ntau*max_component_size_);

  double mib = memory_usage();
  double mib_total = 0.0;
  MPI_Reduce(&mib, &mib_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(mpi_rank==0){
    std::cout << "----------------------------------------\n";
    std::cout << "mpi_comm instance memory usage per rank: " << mib << " MiB\n";
    std::cout << "mpi_comm instance memory usage across all ranks: " << mib_total << " MiB, " << mib_total/(1024.0) <<" GiB, " << mib_total/(1024.0*1024.0) <<" TiB\n";
    std::cout << "----------------------------------------\n";
  }
}

double mpi_comm::memory_usage() const
{
  std::size_t bytes = 0;

  // complex buffers
  bytes += t_alltp_buff.capacity() * sizeof(std::complex<double>);
  bytes += t_allk_buff.capacity() * sizeof(std::complex<double>);
  bytes += tau_alltp_buff.capacity() * sizeof(std::complex<double>);
  bytes += tau_allk_buff.capacity() * sizeof(std::complex<double>);

  // int vectors
  bytes += tau_send_displs.capacity() * sizeof(int);
  bytes += tau_send_counts.capacity() * sizeof(int);
  bytes += tau_recv_displs.capacity() * sizeof(int);
  bytes += tau_recv_counts.capacity() * sizeof(int);

  bytes += t_send_displs.capacity() * sizeof(int);
  bytes += t_send_counts.capacity() * sizeof(int);
  bytes += t_recv_displs.capacity() * sizeof(int);
  bytes += t_recv_counts.capacity() * sizeof(int);

  bytes += kindex_rank.capacity() * sizeof(int);
  bytes += k_rank_map.capacity() * sizeof(int);

  // bytes += init_k_per_rank.capacity() * sizeof(int);
  // bytes += end_k_per_rank.capacity() * sizeof(int);
  bytes += Nk_per_rank.capacity() * sizeof(int);

  bytes += init_tau_per_rank.capacity() * sizeof(int);
  // bytes += end_tau_per_rank.capacity() * sizeof(int);
  bytes += Ntau_per_rank.capacity() * sizeof(int);

  // convert to MiB
  double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  return mib;
}

int mpi_comm::alltp_buff_index(int loc_Nk, int tpi, int local_ki, int compi)
{
  return tpi*loc_Nk*max_component_size_ + local_ki*max_component_size_ + compi;
};

int mpi_comm::t_allk_buff_index(int tpi, int compi, int local_ki, int global_ki)
{
  return t_recv_displs[k_rank_map[global_ki]] + tpi*max_component_size_*Nk_per_rank[k_rank_map[global_ki]] + local_ki*max_component_size_ + compi;
};

int mpi_comm::tau_allk_buff_index(int taui, int compi, int local_ki, int global_ki)
{
  return tau_recv_displs[k_rank_map[global_ki]] + taui*max_component_size_*Nk_per_rank[k_rank_map[global_ki]] + local_ki*max_component_size_ + compi;
};


void mpi_comm::mpi_get_and_comm_spawn
(int ti,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;
    int getsLG_size = getsLG.size();

    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      Ntp_per_rank[mpii] = indxs[2];
    }
    int my_Ntp = Ntp_per_rank[mpi_rank];
    

    #pragma omp parallel for schedule(static)
    for(int tpi=0; tpi<tp_size; tpi++){
      for (int krank = 0; krank<my_Nk; krank++){
        for (int compi = 0; compi < getsLG_size; compi++){     
            int index = alltp_buff_index(my_Nk,tpi,krank,compi);
            t_alltp_buff[index] = getsLG[compi](krank, ti, tpi);
        }
      }
    }

    for(int mpii=0; mpii<mpi_size; mpii++){
      t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]*my_Nk;
      t_recv_counts[mpii] = max_component_size_*my_Ntp*Nk_per_rank[mpii];

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

  int getstv_size = getstv.size();

  #pragma omp parallel for schedule(static)
  for(int taui=0; taui<r_; taui++){
    for (int krank = 0; krank<my_Nk; krank++){
      for (int compi = 0; compi < getstv_size; compi++){     
          int index = alltp_buff_index(my_Nk,taui,krank,compi);
          tau_alltp_buff[index] = getstv[compi](krank, ti, taui);
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
 std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;
    int getsLG_size = getsLG.size();

    std::vector<int> Ntp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      Ntp_per_rank[mpii] = indxs[2];
    }
    int my_Ntp = Ntp_per_rank[mpi_rank];

    std::array<int,3> indxs = get_my_index(thread_id,nthreads,tp_size,0);
    int th_my_init_tp = indxs[0];
    int th_my_end_tp = indxs[1];
    int th_my_Ntp = indxs[2];

    if(th_my_Ntp != 0){ 
      for(int tpi=th_my_init_tp; tpi<=th_my_end_tp; tpi++){
        for (int krank = 0; krank<my_Nk; krank++){
          for (int compi = 0; compi < getsLG_size; compi++){     
              int index = alltp_buff_index(my_Nk,tpi,krank,compi);
              t_alltp_buff[index] = getsLG[compi](krank, ti, tpi);
          }
        }
      }
    }

    #pragma omp barrier

    if(thread_id==0){
      for(int mpii=0; mpii<mpi_size; mpii++){
        t_send_counts[mpii] = max_component_size_*Ntp_per_rank[mpii]*my_Nk;
        t_recv_counts[mpii] = max_component_size_*my_Ntp*Nk_per_rank[mpii];

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

  int getstv_size = getstv.size();

  std::array<int,3> indxs = get_my_index(thread_id,nthreads,r_,0);
  int th_my_init_tau = indxs[0];
  int th_my_end_tau = indxs[1];
  int th_my_Ntau = indxs[2];

  if(th_my_Ntau != 0){
    for(int taui=th_my_init_tau; taui<=th_my_end_tau; taui++){
      for (int krank = 0; krank<my_Nk; krank++){
        for (int compi = 0; compi < getstv_size; compi++){     
            int index = alltp_buff_index(my_Nk,taui,krank,compi);
            tau_alltp_buff[index] = getstv[compi](krank, ti, taui);
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
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(ti>=0){
    int tp_size = ti+1;
    int setsLR_size = setsLR.size();

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    int my_Ntp = Ntp_per_rank[mpi_rank];

    MPI_Alltoallv(t_allk_buff.data(), t_recv_counts.data(), t_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  t_alltp_buff.data(), t_send_counts.data(), t_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_COMM_WORLD);

    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      std::vector<std::complex<double>> Stmp(tp_size);
      for (int compi = 0; compi<setsLR_size; compi++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){

            int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, compi);
            Stmp[init_tp_per_rank[mpi_id]+tpi] = t_alltp_buff[index];

          }
        }
        setsLR[compi](krank, ti, Stmp);
      }
    }
  }

  int setstv_size = setstv.size();

  MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

  #pragma omp parallel for schedule(static)
  for (int krank = 0; krank<my_Nk; krank++){
    std::vector<std::complex<double>> Stmp(r_);
    for (int compi = 0; compi<setstv_size; compi++){
      for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
        for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){

          int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, compi);
          Stmp[init_tau_per_rank[mpi_id]+taui] = tau_alltp_buff[index];
        }
      }
      setstv[compi](krank, ti, Stmp);
    }
  }
};

void mpi_comm::mpi_comm_and_set_nospawn
(int ti,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();

  if(ti>=0){
    int tp_size = ti+1;
    int setsLR_size = setsLR.size();

    std::vector<int> Ntp_per_rank(mpi_size);
    std::vector<int> init_tp_per_rank(mpi_size);
    for(int mpii=0; mpii<mpi_size; mpii++){
      std::array<int,3> indxs = get_my_index(mpii,mpi_size,ti+1,0);
      init_tp_per_rank[mpii] = indxs[0];
      Ntp_per_rank[mpii] = indxs[2];
    }
    int my_Ntp = Ntp_per_rank[mpi_rank];

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
      for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
        std::vector<std::complex<double>> Stmp(tp_size);
        for (int compi = 0; compi<setsLR_size; compi++){
          for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
            for(int tpi=0; tpi<Ntp_per_rank[mpi_id]; tpi++){

              int index = alltp_buff_index(my_Nk, init_tp_per_rank[mpi_id]+tpi, krank, compi);
              Stmp[init_tp_per_rank[mpi_id]+tpi] = t_alltp_buff[index];

            }
          }
          setsLR[compi](krank, ti, Stmp);
        }
      }
    }
  }

  int setstv_size = setstv.size();

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

  if(th_my_Nk != 0){
    for (int krank = th_my_init_k; krank<=th_my_end_k; krank++){
      std::vector<std::complex<double>> Stmp(r_);
      for (int compi = 0; compi<setstv_size; compi++){
        for(int mpi_id = 0; mpi_id<mpi_size; mpi_id++){
          for(int taui=0; taui<Ntau_per_rank[mpi_id]; taui++){

            int index = alltp_buff_index(my_Nk, init_tau_per_rank[mpi_id]+taui, krank, compi);
            Stmp[init_tau_per_rank[mpi_id]+taui] = tau_alltp_buff[index];
          }
        }
        setstv[compi](krank, ti, Stmp);
      }
    }
  }
};

} // namespace
