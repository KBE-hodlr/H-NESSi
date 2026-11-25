#include "h_nessi/mpi_comm_utils.hpp"
#include "h_nessi/checks.hpp"

#include <algorithm>
#include <cstring>

std::array<int, 3> get_my_index(int thread_id, int thread_num, int size, int t0i){
  // MU-1 (tier 1)
  HNESSI_ASSERT(thread_num >= 1 && 0 <= thread_id && thread_id < thread_num && size >= 0, "get_my_index",
                HNESSI_KV(thread_id), HNESSI_KV(thread_num), HNESSI_KV(size));

  int th_my_init_tp, th_my_end_tp, th_my_Ntp, th_Ntp_per_thread;

  if(size==0){
    th_my_init_tp=0;
    th_my_end_tp=0; 
    th_my_Ntp=0;
  } else {
    if(size>thread_num){
      th_Ntp_per_thread = size/thread_num;
      th_my_Ntp = th_Ntp_per_thread;
      if(size%thread_num !=0){
        if(thread_id<size%thread_num){
        th_my_Ntp = th_my_Ntp+1;
        th_my_init_tp = t0i + thread_id*th_my_Ntp;
        } else{
        th_my_init_tp = t0i + thread_id*th_my_Ntp+size%thread_num;
        }
      } else{
        th_my_init_tp = t0i + thread_id*th_my_Ntp;
      }
      th_my_end_tp = th_my_init_tp+th_my_Ntp-1;
    } else{
      if(thread_id<=size-1){
        th_Ntp_per_thread = 1;
        th_my_Ntp = th_Ntp_per_thread;
        th_my_init_tp = t0i + thread_id*th_Ntp_per_thread;
        th_my_end_tp = th_my_init_tp+th_my_Ntp-1;
      } else{
        th_my_Ntp = 0;
        th_my_init_tp = 0;
        th_my_end_tp = 0;
      }
    }
  }

  std::array<int, 3> arr = {th_my_init_tp,th_my_end_tp,th_my_Ntp};
  return arr;

}

std::array<int, 3> get_my_tauindex(int tid, int ntasks, int r){
  // MU-2 (tier 1)
  HNESSI_ASSERT(ntasks >= 1 && 0 <= tid && tid < ntasks && r >= 0, "get_my_tauindex", HNESSI_KV(tid), HNESSI_KV(ntasks),
                HNESSI_KV(r));

  int my_Ntau, Ntau_per_rank, my_init_tau, my_end_tau;

  if(r>ntasks){
    Ntau_per_rank = r/ntasks;
    my_Ntau = Ntau_per_rank;
    if(r%ntasks !=0){
      if(tid<r%ntasks){
      my_Ntau = my_Ntau+1;
      my_init_tau = tid*my_Ntau;
      } else{
      my_init_tau = tid*my_Ntau+r%ntasks;
      }
    } else{
      my_init_tau = tid*my_Ntau;
    }
    my_end_tau = my_init_tau+my_Ntau-1;
  } else{
    if(tid<r){
      Ntau_per_rank = 1;
      my_Ntau = Ntau_per_rank;
      my_init_tau = tid*Ntau_per_rank;
      my_end_tau = my_init_tau+my_Ntau-1;
    } else{
      my_Ntau = 0;
      my_init_tau = 0;
      my_end_tau = 0;
    }
  }

  std::array<int, 3> arr = {my_init_tau,my_end_tau,my_Ntau};
  return arr;

}
  
int iflatten2(int kxi, int kyi, int Nk){
  HNESSI_ASSERT(0 <= kxi && kxi < Nk && 0 <= kyi && kyi < Nk, "iflatten2", HNESSI_KV(kxi), HNESSI_KV(kyi), HNESSI_KV(Nk));  // MU-3
  int i = kxi*Nk+kyi;
  return i;
  }

int iflatten3(int kxi, int kyi, int taui, int Nk, int Ntau){
  HNESSI_ASSERT(0 <= kxi && kxi < Nk && 0 <= kyi && kyi < Nk && 0 <= taui && taui < Ntau, "iflatten3", HNESSI_KV(kxi),
                HNESSI_KV(kyi), HNESSI_KV(taui), HNESSI_KV(Nk), HNESSI_KV(Ntau));  // MU-3
  int i = kxi*(Nk*Ntau)+kyi*Ntau+taui;
  return i;
  }

int minus_xi(int x, int Nk){
  HNESSI_ASSERT(0 <= x && x < Nk, "minus_xi", HNESSI_KV(x), HNESSI_KV(Nk));  // MU-3
  int i;
  if(x == 0){
    i = x;
  } else{
    i = Nk-x;
  }
  return i;
}

int inverse_xi(int x, int Nk){
  HNESSI_ASSERT(0 <= x && x < Nk, "inverse_xi", HNESSI_KV(x), HNESSI_KV(Nk));  // MU-3
  int i = Nk-1-x;
  return i;
}


MemInfo getMemoryInfo() {
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  MemInfo mem;

  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal:") == 0)
      mem.total = std::stod(line.substr(9)) / 1024.0; // kB -> MiB
    else if (line.find("MemFree:") == 0)
      mem.free = std::stod(line.substr(8)) / 1024.0;
    else if (line.find("MemAvailable:") == 0)
      mem.available = std::stod(line.substr(13)) / 1024.0;

    if (mem.total && mem.free && mem.available)
      break;
  }
  // MU-4 (warning): without /proc/meminfo the caller would silently get zeros
  if (!(mem.total > 0))
    h_nessi::detail::warn("getMemoryInfo: /proc/meminfo could not be read; the returned memory sizes are zero");

  return mem;
}
