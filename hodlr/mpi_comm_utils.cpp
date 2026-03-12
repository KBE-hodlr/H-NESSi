#include "mpi_comm_utils.hpp"

#include <algorithm>
#include <cstring>

std::array<int, 3> get_my_index(int thread_id, int thread_num, int size, int t0i){

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
  int i = kxi*Nk+kyi;
  return i;
  }

int iflatten3(int kxi, int kyi, int taui, int Nk, int Ntau){
  int i = kxi*(Nk*Ntau)+kyi*Ntau+taui;
  return i;
  }

int minus_xi(int x, int Nk){
  int i;
  if(x == 0){
    i = x;
  } else{
    i = Nk-x;
  }
  return i;
}

int inverse_xi(int x, int Nk){
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

  return mem;
}
