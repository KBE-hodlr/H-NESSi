#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/checks.hpp"
#include "h_nessi/utils.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

namespace h_nessi {

namespace {
// WP6a side finding 4: an uncaught exception on one rank would leave the other ranks waiting in a collective. The
// first mpi_comm installs this std::terminate handler (once per process): while MPI is active it prints the exception
// with the rank and calls MPI_Abort; otherwise it chains to the previous handler.
std::terminate_handler g_prev_terminate = nullptr;
std::once_flag g_terminate_once;

[[noreturn]] void mpi_terminate_handler() {
  int init = 0, fin = 0;
  MPI_Initialized(&init);
  if(init) MPI_Finalized(&fin);
  if(init && !fin) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string what = "no active exception";
    if(std::exception_ptr p = std::current_exception()) {
      try {
        std::rethrow_exception(p);
      } catch(const std::exception &e) {
        what = e.what();
      } catch(...) {
        what = "non-standard exception";
      }
    }
    std::fprintf(stderr, "h_nessi: std::terminate on MPI rank %d: %s\n  calling MPI_Abort(MPI_COMM_WORLD, 1)\n", rank,
                 what.c_str());
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if(g_prev_terminate) g_prev_terminate();
  std::abort();
}
}  // namespace

mpi_comm::mpi_comm(){};

mpi_comm::mpi_comm
(int Nk, 
 int Nt,
 int r,
 int nao,
 int ncomponents
):Nk_(Nk), Nt_(Nt), r_(r), max_component_size_(ncomponents*nao*nao), nao_(nao)
{
  // MP-1, MP-2 (tier 2)
  HNESSI_REQUIRE(Nk >= 1 && Nt >= 1 && r >= 1 && nao >= 1 && ncomponents >= 1, "mpi_comm::mpi_comm", HNESSI_KV(Nk),
                 HNESSI_KV(Nt), HNESSI_KV(r), HNESSI_KV(nao), HNESSI_KV(ncomponents));
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  HNESSI_REQUIRE_STATE(mpi_initialized, "mpi_comm::mpi_comm", HNESSI_KV(mpi_initialized),
                       detail::note{"initialise MPI (MPI_Init_thread) before constructing an mpi_comm"});
  std::call_once(g_terminate_once, [] { g_prev_terminate = std::set_terminate(mpi_terminate_handler); });

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // H28: the _nospawn routines call MPI from thread 0 inside an OpenMP parallel region, which requires at least
  // MPI_THREAD_FUNNELED. Warn here (programs that use only the _spawn routines are fine with MPI_Init); every
  // _nospawn call then requires it (MP-3).
  int thread_level = MPI_THREAD_SINGLE;
  MPI_Query_thread(&thread_level);
  mpi_thread_level_ = thread_level;
  if(thread_level < MPI_THREAD_FUNNELED && mpi_rank==0){
    std::cerr << "h_nessi::mpi_comm warning: the MPI thread support level is below MPI_THREAD_FUNNELED. "
                 "The *_nospawn routines call MPI from inside an OpenMP parallel region and need at least "
                 "MPI_THREAD_FUNNELED: initialise MPI with MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided). "
                 "The *_spawn routines are not affected.\n";
  }

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
  imag_rel_k_.assign(my_Nk, 0.);
  rev_buff_.assign((std::size_t)my_Nk * r_ * nao_ * nao_, std::complex<double>(0.));

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
  bytes += rev_buff_.capacity() * sizeof(std::complex<double>);   // high-level get scratch (WP6c)

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

  bytes += init_k_per_rank.capacity() * sizeof(int);
  bytes += Nk_per_rank.capacity() * sizeof(int);
  bytes += my_global_k_list.capacity() * sizeof(int);

  bytes += init_tau_per_rank.capacity() * sizeof(int);
  bytes += Ntau_per_rank.capacity() * sizeof(int);

  // check state (WP6b; memory rule: every member is counted)
  bytes += imag_rel_k_.capacity() * sizeof(double);
  bytes += sizeof(last_get_ti_) + sizeof(last_get_nLG_) + sizeof(last_get_ntv_) + sizeof(last_get_highlevel_) +
           sizeof(mpi_thread_level_) + sizeof(imag_warned_);

  // convert to MiB
  double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  return mib;
}

// WP6a side finding 2: the component-count checks follow the checks.hpp policy (HNESSI_REQUIRE): they throw
// h_nessi::precondition_error outside a parallel region and print + abort inside a user region (_nospawn).
void mpi_comm::check_component_count_(std::size_t ncomp, const char *what, const char *func) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE(ncomp <= static_cast<std::size_t>(max_component_size_), where.c_str(), HNESSI_KV(what),
                 HNESSI_KV(ncomp), HNESSI_KV(max_component_size_),
                 detail::note{"the number of lambdas exceeds max_component_size_ = ncomponents*nao*nao (construct "
                              "mpi_comm with a larger ncomponents)"});
}

void mpi_comm::check_highlevel_components_(const char *func) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE(max_component_size_ >= 2*nao_*nao_, where.c_str(), HNESSI_KV(max_component_size_), HNESSI_KV(nao_),
                 detail::note{"the herm_matrix_hodlr API uses 2*nao*nao components per slot (construct mpi_comm with "
                              "ncomponents >= 2)"});
}

void mpi_comm::check_ti_(const char *func, int ti) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE(-1 <= ti && ti < Nt_, where.c_str(), HNESSI_KV(ti), HNESSI_KV(Nt_),
                 detail::note{"ti = -1 (Matsubara) or 0 <= ti < Nt"});
}

void mpi_comm::check_objects_(const char *func, int ti, std::size_t n,
                              const std::function<herm_matrix_hodlr &(int)> &at) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE(n == static_cast<std::size_t>(my_Nk), where.c_str(), HNESSI_KV(n), HNESSI_KV(my_Nk),
                 detail::note{"hmh_vec must hold one object per local k-point (my_Nk)"});
  for(int i = 0; i < my_Nk; i++) {
    herm_matrix_hodlr &g = at(i);
    HNESSI_REQUIRE(g.size1() == nao_ && g.size2() == nao_ && g.r() == r_ && g.nt() > ti, where.c_str(), HNESSI_KV(i),
                   HNESSI_KV(g.size1()), HNESSI_KV(nao_), HNESSI_KV(g.r()), HNESSI_KV(r_), HNESSI_KV(g.nt()),
                   HNESSI_KV(ti));
    HNESSI_REQUIRE(ti < 0 || (g.tstpmk() <= ti && ti <= g.tstp()), where.c_str(), HNESSI_KV(i), HNESSI_KV(ti),
                   HNESSI_KV(g.tstpmk()), HNESSI_KV(g.tstp()),
                   detail::note{"row ti must be in the current window of every object"});
  }
}

void mpi_comm::check_dlr_(const char *func, const dlr_info &dlr) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE(dlr.r() == r_ && dlr.size1() == nao_, where.c_str(), HNESSI_KV(dlr.r()), HNESSI_KV(r_),
                 HNESSI_KV(dlr.size1()), HNESSI_KV(nao_));
}

void mpi_comm::require_funneled_(const char *func) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE_STATE(mpi_thread_level_ >= MPI_THREAD_FUNNELED, where.c_str(), HNESSI_KV(mpi_thread_level_),
                       detail::note{"the _nospawn routines need MPI_THREAD_FUNNELED: initialise MPI with "
                                    "MPI_Init_thread(..., MPI_THREAD_FUNNELED, &provided)"});
}

void mpi_comm::record_get_(int ti, int nLG, int ntv, bool highlevel)
{
  last_get_ti_ = ti;
  last_get_nLG_ = nLG;
  last_get_ntv_ = ntv;
  last_get_highlevel_ = highlevel;
}

void mpi_comm::require_get_(const char *func, int ti, int nLR, int ntv, bool highlevel) const
{
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_REQUIRE_STATE(last_get_ti_ == ti && last_get_highlevel_ == highlevel &&
                       (ti < 0 || last_get_nLG_ == nLR) && last_get_ntv_ == ntv, where.c_str(), HNESSI_KV(ti),
                       HNESSI_KV(last_get_ti_), HNESSI_KV(highlevel), HNESSI_KV(last_get_highlevel_), HNESSI_KV(nLR),
                       HNESSI_KV(last_get_nLG_), HNESSI_KV(ntv), HNESSI_KV(last_get_ntv_),
                       detail::note{"a set must follow a get with the same ti, API (high-level or lambda) and "
                                    "component counts: it reuses that get's counts and displacements (last_get_ti_ = "
                                    "-2147483648: no get since the last set)"});
}

void mpi_comm::debug_collective_(const char *func, int ti, int n1, int n2) const
{
#ifndef NDEBUG
  int v[6] = {ti, n1, n2, -ti, -n1, -n2}, m[6];
  MPI_Allreduce(v, m, 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::string where = std::string("mpi_comm::") + func;
  HNESSI_ASSERT(m[0] == -m[3] && m[1] == -m[4] && m[2] == -m[5], where.c_str(), HNESSI_KV(ti), HNESSI_KV(n1),
                HNESSI_KV(n2), HNESSI_KV(m[0]), HNESSI_KV(-m[3]));
#else
  (void)func; (void)ti; (void)n1; (void)n2;
#endif
}

void mpi_comm::warn_imag_mat_(const char *func, double rel)
{
  if(rel > 1e-10 && !imag_warned_) {
    imag_warned_ = true;
    detail::warn(std::string("h_nessi::mpi_comm::") + func + ": Sigma^M has a relative imaginary part " +
                 std::to_string(rel) + " (> 1e-10) that is dropped: G^M and Sigma^M are stored real (ISSUES D1). "
                 "Warned once per mpi_comm object.");
  }
}

void mpi_comm::set_my_t_block_(int ti, int mpi_rank, int mpi_size)
{
  std::array<int,3> indxs = get_my_index(mpi_rank,mpi_size,ti+1,0);
  my_first_t = indxs[0];
  my_Nt = indxs[2];
}

int mpi_comm::alltp_buff_index(int loc_Nk, int tpi, int local_ki, int compi)
{
  // MP-16 (tier 1)
  HNESSI_ASSERT(0 <= tpi && 0 <= local_ki && local_ki < loc_Nk && 0 <= compi && compi < max_component_size_ &&
                (std::size_t)(tpi*loc_Nk*max_component_size_ + local_ki*max_component_size_ + compi) <
                    std::max(t_alltp_buff.size(), tau_alltp_buff.size()),
                "mpi_comm::alltp_buff_index", HNESSI_KV(tpi), HNESSI_KV(local_ki), HNESSI_KV(loc_Nk), HNESSI_KV(compi));
  return tpi*loc_Nk*max_component_size_ + local_ki*max_component_size_ + compi;
};

int mpi_comm::t_allk_buff_index(int tpi, int compi, int local_ki, int global_ki)
{
  // MP-17 (tier 1): valid after a get with ti >= 0
  HNESSI_ASSERT(0 <= global_ki && global_ki < Nk_ && 0 <= tpi && tpi < my_Nt && 0 <= compi &&
                compi < max_component_size_ && local_ki == global_ki - init_k_per_rank[k_rank_map[global_ki]],
                "mpi_comm::t_allk_buff_index", HNESSI_KV(tpi), HNESSI_KV(my_Nt), HNESSI_KV(compi), HNESSI_KV(local_ki),
                HNESSI_KV(global_ki));
  return t_recv_displs[k_rank_map[global_ki]] + tpi*max_component_size_*Nk_per_rank[k_rank_map[global_ki]] + local_ki*max_component_size_ + compi;
};

int mpi_comm::tau_allk_buff_index(int taui, int compi, int local_ki, int global_ki)
{
  // MP-17 (tier 1)
  HNESSI_ASSERT(0 <= global_ki && global_ki < Nk_ && 0 <= taui && taui < my_Ntau && 0 <= compi &&
                compi < max_component_size_ && local_ki == global_ki - init_k_per_rank[k_rank_map[global_ki]],
                "mpi_comm::tau_allk_buff_index", HNESSI_KV(taui), HNESSI_KV(my_Ntau), HNESSI_KV(compi),
                HNESSI_KV(local_ki), HNESSI_KV(global_ki));
  return tau_recv_displs[k_rank_map[global_ki]] + taui*max_component_size_*Nk_per_rank[k_rank_map[global_ki]] + local_ki*max_component_size_ + compi;
};


void mpi_comm::mpi_get_and_comm_spawn
(int ti,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
)
{
  check_ti_("mpi_get_and_comm_spawn", ti);
  if(ti>=0) check_component_count_(getsLG.size(), "getsLG", "mpi_get_and_comm_spawn");
  check_component_count_(getstv.size(), "getstv", "mpi_get_and_comm_spawn");
  debug_collective_("mpi_get_and_comm_spawn", ti, ti >= 0 ? (int)getsLG.size() : 0, (int)getstv.size());

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
    set_my_t_block_(ti, mpi_rank, mpi_size);

    {
      detail::omp_exception_collector omp_errs;   // library-owned region: capture, rethrow after it
      #pragma omp parallel for schedule(static)
      for(int tpi=0; tpi<tp_size; tpi++){
        omp_errs.run([&] {
      for (int krank = 0; krank<my_Nk; krank++){
        for (int compi = 0; compi < getsLG_size; compi++){     
            int index = alltp_buff_index(my_Nk,tpi,krank,compi);
            t_alltp_buff[index] = getsLG[compi](krank, ti, tpi);
        }
      }
        });
      }
      omp_errs.rethrow();
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

  {
    detail::omp_exception_collector omp_errs;   // library-owned region: capture, rethrow after it
    #pragma omp parallel for schedule(static)
    for(int taui=0; taui<r_; taui++){
      omp_errs.run([&] {
    for (int krank = 0; krank<my_Nk; krank++){
      for (int compi = 0; compi < getstv_size; compi++){     
          int index = alltp_buff_index(my_Nk,taui,krank,compi);
          tau_alltp_buff[index] = getstv[compi](krank, ti, taui);
      }
    }
      });
    }
    omp_errs.rethrow();
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
  record_get_(ti, (int)getsLG.size(), (int)getstv.size(), false);   // ST-9
}

void mpi_comm::mpi_get_and_comm_nospawn
(int ti,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
 std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
)
{
  // checks before any barrier; identical on every thread (shared arguments), so all threads fail together
  check_ti_("mpi_get_and_comm_nospawn", ti);
  if(ti>=0) check_component_count_(getsLG.size(), "getsLG", "mpi_get_and_comm_nospawn");
  check_component_count_(getstv.size(), "getstv", "mpi_get_and_comm_nospawn");
  require_funneled_("mpi_get_and_comm_nospawn");

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // B21: partition over the current team (1 outside a parallel region), not omp_get_max_threads()
  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  if(thread_id == 0)
    debug_collective_("mpi_get_and_comm_nospawn", ti, ti >= 0 ? (int)getsLG.size() : 0, (int)getstv.size());

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
      set_my_t_block_(ti, mpi_rank, mpi_size);
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
    record_get_(ti, (int)getsLG.size(), (int)getstv.size(), false);   // ST-9 (thread 0, published by the barrier)
  }

  #pragma omp barrier
}

void mpi_comm::mpi_comm_and_set_spawn
(int ti,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
)
{
  check_ti_("mpi_comm_and_set_spawn", ti);
  if(ti>=0) check_component_count_(setsLR.size(), "setsLR", "mpi_comm_and_set_spawn");
  check_component_count_(setstv.size(), "setstv", "mpi_comm_and_set_spawn");
  require_get_("mpi_comm_and_set_spawn", ti, (int)setsLR.size(), (int)setstv.size(), false);   // ST-9
  debug_collective_("mpi_comm_and_set_spawn", ti, ti >= 0 ? (int)setsLR.size() : 0, (int)setstv.size());
  clear_get_();

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

    {
      detail::omp_exception_collector omp_errs;   // library-owned region: capture, rethrow after it
      #pragma omp parallel for schedule(static)
      for (int krank = 0; krank<my_Nk; krank++){
        omp_errs.run([&] {
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
        });
      }
      omp_errs.rethrow();
    }
  }

  int setstv_size = setstv.size();

  MPI_Alltoallv(tau_allk_buff.data(), tau_recv_counts.data(), tau_recv_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                tau_alltp_buff.data(), tau_send_counts.data(), tau_send_displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                MPI_COMM_WORLD);

  {
    detail::omp_exception_collector omp_errs;   // library-owned region: capture, rethrow after it
    #pragma omp parallel for schedule(static)
    for (int krank = 0; krank<my_Nk; krank++){
      omp_errs.run([&] {
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
      });
    }
    omp_errs.rethrow();
  }
};

void mpi_comm::mpi_comm_and_set_nospawn
(int ti,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
 std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
)
{
  // checks before any barrier; identical on every thread (shared arguments), so all threads fail together
  check_ti_("mpi_comm_and_set_nospawn", ti);
  if(ti>=0) check_component_count_(setsLR.size(), "setsLR", "mpi_comm_and_set_nospawn");
  check_component_count_(setstv.size(), "setstv", "mpi_comm_and_set_nospawn");
  require_funneled_("mpi_comm_and_set_nospawn");
  require_get_("mpi_comm_and_set_nospawn", ti, (int)setsLR.size(), (int)setstv.size(), false);   // ST-9

  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  // B21: partition over the current team (1 outside a parallel region), not omp_get_max_threads()
  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  // every thread has passed require_get_ before this barrier; then thread 0 consumes the get record (ST-9)
  #pragma omp barrier
  if(thread_id == 0) {
    debug_collective_("mpi_comm_and_set_nospawn", ti, ti >= 0 ? (int)setsLR.size() : 0, (int)setstv.size());
    clear_get_();
  }

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

  // H24: the set lambdas of other threads must have finished before any thread uses their results
  #pragma omp barrier
};

} // namespace
