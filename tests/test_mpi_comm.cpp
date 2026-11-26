// mpi_comm routing (doc 31) with simulated ranks.  Ported from agent_scratch/31/commtest.cpp.
// The MPI symbols used by libh_nessi's mpi_comm (MPI_Comm_size, MPI_Comm_rank, MPI_Alltoallv, MPI_Reduce,
// MPI_Query_thread) are interposed by definitions in this executable; P "ranks" run as std::threads in one process,
// so the library's multi-rank Alltoallv routing is exercised unchanged without an MPI launcher.
// Regression tests for B21 (_nospawn partitioned work with omp_get_max_threads), B20 (component-count checks),
// H24 (trailing barrier of mpi_comm_and_set_nospawn), H30 (my_first_t/my_Nt), H28 (thread-level warning); all
// fixed in WP5.
#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <omp.h>
#include <thread>

#include "common/test_util.hpp"
#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/mpi_comm_utils.hpp"

using namespace h_nessi;

// ------------------------------------------------------------------ MPI shim
static int g_P = 1;
static thread_local int g_rank = 0;
static std::barrier<> *g_bar = nullptr;
struct A2A { const char *sb; const int *sc; const int *sd; char *rb; const int *rc; const int *rd; };
static std::vector<A2A> g_a2a;
static std::vector<double> g_red;
static std::atomic<long> g_cnt_mismatch{0};
static int g_thread_level = MPI_THREAD_FUNNELED;  // what MPI_Query_thread reports (H28 test changes it)
static int g_mpi_initialized = 1;                 // what MPI_Initialized reports (MP-2 test changes it)
static std::vector<std::vector<int>> g_ared;      // MPI_Allreduce (MP-18, Debug builds) per simulated rank
extern "C" {
int MPI_Query_thread(int *provided) { *provided = g_thread_level; return 0; }
int MPI_Initialized(int *flag) { *flag = g_mpi_initialized; return 0; }
int MPI_Finalized(int *flag) { *flag = 0; return 0; }
int MPI_Abort(MPI_Comm, int code) {  // the mpi_comm std::terminate handler ends here (death test)
  std::fprintf(stderr, "shim MPI_Abort(%d)\n", code);
  std::fflush(stderr);
  std::_Exit(code);
}
// MPI_INT / MPI_MAX only (what mpi_comm's Debug collective check uses)
int MPI_Allreduce(const void *sb, void *rb, int n, MPI_Datatype, MPI_Op, MPI_Comm) {
  g_ared[g_rank].assign((const int *)sb, (const int *)sb + n);
  g_bar->arrive_and_wait();
  for (int i = 0; i < n; i++) {
    int m = g_ared[0][i];
    for (int p = 1; p < g_P; p++) m = std::max(m, g_ared[p][i]);
    ((int *)rb)[i] = m;
  }
  g_bar->arrive_and_wait();
  return 0;
}
int MPI_Comm_size(MPI_Comm, int *s) { *s = g_P; return 0; }
int MPI_Comm_rank(MPI_Comm, int *r) { *r = g_rank; return 0; }
int MPI_Alltoallv(const void *sb, const int sc[], const int sd[], MPI_Datatype st, void *rb, const int rc[], const int rd[],
                  MPI_Datatype rt, MPI_Comm) {
  if (st != MPI_CXX_DOUBLE_COMPLEX || rt != MPI_CXX_DOUBLE_COMPLEX) {
    fprintf(stderr, "bad type\n");
    abort();
  }
  const size_t es = sizeof(std::complex<double>);
  g_a2a[g_rank] = {(const char *)sb, sc, sd, (char *)rb, rc, rd};
  g_bar->arrive_and_wait();
  for (int p = 0; p < g_P; p++) {
    const A2A &s = g_a2a[p];
    if (s.sc[g_rank] != rc[p]) g_cnt_mismatch++;
    std::memcpy((char *)rb + (size_t)rd[p] * es, s.sb + (size_t)s.sd[g_rank] * es, (size_t)rc[p] * es);
  }
  g_bar->arrive_and_wait();
  return 0;
}
int MPI_Reduce(const void *sb, void *rb, int, MPI_Datatype, MPI_Op, int root, MPI_Comm) {
  g_red[g_rank] = *(const double *)sb;
  g_bar->arrive_and_wait();
  if (g_rank == root) {
    double s = 0;
    for (double x : g_red) s += x;
    *(double *)rb = s;
  }
  g_bar->arrive_and_wait();
  return 0;
}
}

namespace {
const double BETA = 5.0, LAMBDA = 50.0, EPSDLR = 1e-12;
const int XI = -1;
const double OM[3] = {-1.3, 0.4, 2.1};
double Kf(double w, double tau) { return std::exp(-w * tau) / (1.0 + std::exp(-BETA * w)); }
cplx fR(int kg, int T, int tp, int i, int j) { return cplx(1000. * kg + 100. * T + tp + 0.1 * i + 0.01 * j, 1. + 0.5 * kg - 0.25 * tp + i - j); }
cplx fL(int kg, int T, int tp, int i, int j) { return cplx(-1000. * kg - 100. * T - tp - 0.1 * i - 0.01 * j, 3. + 0.7 * kg + 0.125 * tp - i + 2. * j); }
cplx gR(int kg, int T, int tp, int i, int j) { return cplx(7. + kg + 0.3 * T + 0.02 * tp + i, -0.5 * kg + 0.1 * tp + 0.001 * j); }
cplx gL(int kg, int T, int tp, int i, int j) { return cplx(-7. - kg + 0.4 * T + 0.03 * tp + j, 0.6 * kg - 0.2 * tp + 0.002 * i); }
cplx fTV(int kg, int T, double tau, int i, int j) {
  cplx s = 0;
  for (int p = 0; p < 3; p++) s += cplx(0.1 * (p + 1) + 0.01 * kg + 0.001 * i, 0.05 * T - 0.02 * j + 0.003 * p) * Kf(OM[p], tau);
  return s;
}
cplx gTV(int kg, int T, int n, int i, int j) { return cplx(0.5 * kg + T + 0.01 * n, 0.25 * n - i + 0.5 * j); }
double fM(int kg, double tau, int i, int j) {
  double s = 0;
  for (int p = 0; p < 3; p++) s += (0.2 * (p + 1) + 0.01 * kg + 0.003 * i - 0.002 * j) * Kf(OM[p], tau);
  return s;
}
cplx gM(int kg, int n, int i, int j) { return cplx(3. * kg + 0.1 * n + i - j, 11. + kg); }

struct Err {
  double maxdiff = 0;
  long nbad = 0, nchk = 0;
  void chk(cplx a, cplx b, double tol) {
    nchk++;
    double d = std::abs(a - b);
    if (d > maxdiff) maxdiff = d;
    if (d > tol) nbad++;
  }
};
std::mutex g_print;
const char *NAMES[9] = {"distribution vectors", "down t (map_ret/map_les)", "down tau direct (map_tv/map_mat)", "down tau reversed",
                        "up ret/les -> S curr buffers", "up tv/mat -> S", "get leaves G.tv row T / G.mat unchanged",
                        "get leaves the other G.tv rows unchanged (no scratch in G, WP6c)", "allk index formula"};

// mode: 0 spawn/vector, 1 spawn/refwrap, 2 nospawn/vector, 3 nospawn/refwrap,
//       4 nospawn/vector called OUTSIDE omp parallel (team of 1), 5 nospawn/vector in a num_threads(2) team
void run_rank(int Nk, int nao, int k, int mode, std::vector<Err> &errs_out) {
  const int rank = g_rank, P = g_P;
  int r = 0;
  dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
  const int nt = 12, nlvl = 1;
  const double tol = 1e-10;
  mpi_comm comm(Nk, nt, r, nao);
  const int myNk = comm.my_Nk;
  std::vector<herm_matrix_hodlr> G, S;
  G.reserve(myNk);
  S.reserve(myNk);
  for (int i = 0; i < myNk; i++) {
    G.emplace_back(nt, r, nlvl, tol, nao, nao, XI, k);
    S.emplace_back(nt, r, nlvl, tol, nao, nao, XI, k);
  }
  std::vector<std::reference_wrapper<herm_matrix_hodlr>> Gr, Sr;
  for (int i = 0; i < myNk; i++) {
    Gr.push_back(G[i]);
    Sr.push_back(S[i]);
  }
  std::vector<Err> E(9);
  std::vector<double> taus(r);
  for (int n = 0; n < r; n++) taus[n] = dlr.it0B(n);
  {
    auto a = get_my_index(rank, P, Nk, 0);
    E[0].chk(comm.my_Nk, a[2], 0);
    E[0].chk(comm.my_global_k_list.size() ? comm.my_global_k_list[0] : a[0], a[0], 0);
    for (int kg = 0; kg < Nk; kg++) {
      int p = comm.k_rank_map[kg];
      E[0].chk(kg >= comm.init_k_per_rank[p] && kg < comm.init_k_per_rank[p] + comm.Nk_per_rank[p], 1, 0);
    }
    for (int i = 0; i < myNk; i++) E[0].chk(comm.kindex_rank[i], comm.my_global_k_list[i], 0);
    auto b = get_my_tauindex(rank, P, r);
    E[0].chk(comm.my_first_tau, b[0], 0);
    E[0].chk(comm.my_Ntau, b[2], 0);
    // WP6c: the get's own scratch (one r x nao x nao slice per local k) is allocated and counted in memory_usage()
    E[0].chk((double)comm.rev_buff_.size(), (double)myNk * r * nao * nao, 0);
    double cplx_bytes = 16.0 * (comm.t_alltp_buff.capacity() + comm.t_allk_buff.capacity() + comm.tau_alltp_buff.capacity() +
                                comm.tau_allk_buff.capacity() + comm.rev_buff_.capacity());
    E[0].chk(comm.memory_usage() * 1024.0 * 1024.0 >= cplx_bytes, 1, 0);
  }
  const int es = nao * nao;
  for (int T = -1; T <= k; T++) {
    for (int kl = 0; kl < myNk; kl++) {
      int kg = comm.my_global_k_list[kl];
      if (T == -1) {
        for (int n = 0; n < r; n++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) G[kl].matptr(n)[i * nao + j] = fM(kg, taus[n], i, j);
      } else {
        for (int tp = 0; tp <= T; tp++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              G[kl].curr_timestep_ret_ptr(T, tp)[i * nao + j] = fR(kg, T, tp, i, j);
              G[kl].curr_timestep_les_ptr(tp, T)[i * nao + j] = fL(kg, T, tp, i, j);
            }
        for (int n = 0; n < r; n++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) G[kl].tvptr(T, n)[i * nao + j] = fTV(kg, T, taus[n], i, j);
      }
    }
    auto body = [&]() {
      if (T >= 0) {
        auto ti = get_my_index(rank, P, T + 1, 0);
        E[0].chk(comm.my_first_t, ti[0], 0);
        E[0].chk(comm.my_Nt, ti[2], 0);
        for (int t = comm.my_first_t; t < comm.my_first_t + comm.my_Nt; t++)
          for (int kg = 0; kg < Nk; kg++) {
            int p = comm.k_rank_map[kg], kl = kg - comm.init_k_per_rank[p], tl = t - comm.my_first_t;
            E[8].chk(comm.t_allk_buff_index(tl, 0, kl, kg), 2 * es * (comm.my_Nt * comm.init_k_per_rank[p] + tl * comm.Nk_per_rank[p] + kl), 0);
            auto R = comm.map_ret(kg, t);
            auto L = comm.map_les(kg, t);
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                E[1].chk(R(i, j), fR(kg, T, t, i, j), 0);
                E[1].chk(L(i, j), fL(kg, T, t, i, j), 0);
              }
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                R(i, j) = gR(kg, T, t, i, j);
                L(i, j) = gL(kg, T, t, i, j);
              }
          }
      }
      for (int n = comm.my_first_tau; n < comm.my_first_tau + comm.my_Ntau; n++)
        for (int kg = 0; kg < Nk; kg++) {
          int p = comm.k_rank_map[kg], kl = kg - comm.init_k_per_rank[p], nl = n - comm.my_first_tau;
          E[8].chk(comm.tau_allk_buff_index(nl, 0, kl, kg), 2 * es * (comm.my_Ntau * comm.init_k_per_rank[p] + nl * comm.Nk_per_rank[p] + kl), 0);
          if (T >= 0) {
            auto A = comm.map_tv(kg, n);
            auto B = comm.map_tv_rev(kg, n);
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                E[2].chk(A(i, j), fTV(kg, T, taus[n], i, j), 0);
                E[3].chk(B(i, j), fTV(kg, T, BETA - taus[n], i, j), 1e-8);
              }
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                A(i, j) = gTV(kg, T, n, i, j);
                B(i, j) = cplx(1e6, 1e6);
              }
          } else {
            auto A = comm.map_mat(kg, n);
            auto B = comm.map_mat_rev(kg, n);
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                E[2].chk(A(i, j), fM(kg, taus[n], i, j), 0);
                E[3].chk(B(i, j), fM(kg, BETA - taus[n], i, j), 1e-8);
              }
            for (int i = 0; i < nao; i++)
              for (int j = 0; j < nao; j++) {
                A(i, j) = gM(kg, n, i, j);
                B(i, j) = cplx(1e6, 1e6);
              }
          }
        }
    };
    if (mode == 0) {
      comm.mpi_get_and_comm_spawn(T, G, dlr);
      body();
      comm.mpi_comm_and_set_spawn(T, S);
    } else if (mode == 1) {
      comm.mpi_get_and_comm_spawn(T, Gr, dlr);
      body();
      comm.mpi_comm_and_set_spawn(T, Sr);
    } else if (mode == 2 || mode == 3) {
#pragma omp parallel
      {
        g_rank = rank;
        if (mode == 2) comm.mpi_get_and_comm_nospawn(T, G, dlr);
        else comm.mpi_get_and_comm_nospawn(T, Gr, dlr);
#pragma omp master
        body();
#pragma omp barrier
        if (mode == 2) comm.mpi_comm_and_set_nospawn(T, S);
        else comm.mpi_comm_and_set_nospawn(T, Sr);
#pragma omp barrier
      }
    } else if (mode == 5) {
#pragma omp parallel num_threads(2)
      {
        g_rank = rank;
        comm.mpi_get_and_comm_nospawn(T, G, dlr);
#pragma omp master
        body();
#pragma omp barrier
        comm.mpi_comm_and_set_nospawn(T, S);
#pragma omp barrier
      }
    } else {
      comm.mpi_get_and_comm_nospawn(T, G, dlr);
      body();
      comm.mpi_comm_and_set_nospawn(T, S);
    }
    for (int kl = 0; kl < myNk; kl++) {
      int kg = comm.my_global_k_list[kl];
      if (T >= 0) {
        for (int tp = 0; tp <= T; tp++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              E[4].chk(S[kl].curr_timestep_ret_ptr(T, tp)[i * nao + j], gR(kg, T, tp, i, j), 1e-12);
              E[4].chk(S[kl].curr_timestep_les_ptr(tp, T)[i * nao + j], gL(kg, T, tp, i, j), 1e-12);
            }
        for (int n = 0; n < r; n++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              E[5].chk(S[kl].tvptr(T, n)[i * nao + j], gTV(kg, T, n, i, j), 1e-12);
              // WP6c (H27): the get used G.tv_trans as scratch for G(T,beta-tau); now it has its own buffer and G is
              // only read
              E[6].chk(G[kl].tvptr(T, n)[i * nao + j], fTV(kg, T, taus[n], i, j), 0);
              for (int t = 0; t < T; t++) E[7].chk(G[kl].tvptr(t, n)[i * nao + j], fTV(kg, t, taus[n], i, j), 0);
            }
      } else {
        for (int n = 0; n < r; n++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              E[5].chk(S[kl].matptr(n)[i * nao + j], gM(kg, n, i, j).real(), 1e-12);  // mat passes a DLR round trip
              E[6].chk(G[kl].matptr(n)[i * nao + j], fM(kg, taus[n], i, j), 0);
              E[7].chk(G[kl].tvptr(0, n)[i * nao + j], 0.0, 0);   // no G^M(beta-tau) scratch in G.tv row 0
            }
      }
    }
  }
  std::lock_guard<std::mutex> lk(g_print);
  for (int e = 0; e < 9; e++) {
    errs_out[e].nbad += E[e].nbad;
    errs_out[e].nchk += E[e].nchk;
    errs_out[e].maxdiff = std::max(errs_out[e].maxdiff, E[e].maxdiff);
  }
}

// low-level lambda API
// mode: 0 spawn, 1 nospawn in a full team, 2 nospawn in a num_threads(2) team, 3 nospawn OUTSIDE omp parallel
// E[4] checks my_first_t/my_Nt (H30): -1/0 before any get with ti >= 0, then the rank's t' block.
void run_rank_lambda(int Nk, int nao, int T, int mode, std::vector<Err> &errs_out) {
  const int rank = g_rank, P = g_P;
  const int nt = 12, r = 7, ncomp = 1;
  mpi_comm comm(Nk, nt, r, nao, ncomp);
  const int mcs = ncomp * nao * nao, nc = std::min(3, mcs);
  auto h1 = [](int c, int kg, int ti, int tp) { return cplx(100. * c + 10. * kg + tp, ti + 0.5 * c); };
  auto h3 = [](int c, int kg, int ti, int tp) { return cplx(-3. * c + kg - 0.1 * tp, 7. * ti + c * kg); };
  std::vector<std::function<cplx(int, int, int)>> getsLG, getstv;
  for (int c = 0; c < nc; c++) {
    getsLG.push_back([&comm, c, h1](int kl, int ti, int tp) { return h1(c, comm.my_global_k_list[kl], ti, tp); });
    getstv.push_back([&comm, c, h1](int kl, int ti, int n) { return h1(c + 10, comm.my_global_k_list[kl], ti, n); });
  }
  std::vector<std::vector<cplx>> recvLR(nc * comm.my_Nk), recvTV(nc * comm.my_Nk);
  std::vector<std::function<void(int, int, std::vector<cplx> &)>> setsLR, setstv;
  for (int c = 0; c < nc; c++) {
    setsLR.push_back([&recvLR, c, nc](int kl, int, std::vector<cplx> &v) { recvLR[kl * nc + c] = v; });
    setstv.push_back([&recvTV, c, nc](int kl, int, std::vector<cplx> &v) { recvTV[kl * nc + c] = v; });
  }
  std::vector<Err> E(5);
  E[4].chk(comm.my_first_t, -1, 0);
  E[4].chk(comm.my_Nt, 0, 0);
  auto body = [&]() {
    if (T >= 0) {
      auto ti = get_my_index(rank, P, T + 1, 0);
      E[4].chk(comm.my_first_t, ti[0], 0);
      E[4].chk(comm.my_Nt, ti[2], 0);
      for (int tl = 0; tl < ti[2]; tl++)
        for (int kg = 0; kg < Nk; kg++)
          for (int c = 0; c < nc; c++) {
            int p = comm.k_rank_map[kg], kl = kg - comm.init_k_per_rank[p];
            cplx &x = comm.t_allk_buff[comm.t_allk_buff_index(tl, c, kl, kg)];
            E[0].chk(x, h1(c, kg, T, ti[0] + tl), 0);
            x = h3(c, kg, T, ti[0] + tl);
          }
    }
    for (int nl = 0; nl < comm.my_Ntau; nl++)
      for (int kg = 0; kg < Nk; kg++)
        for (int c = 0; c < nc; c++) {
          int p = comm.k_rank_map[kg], kl = kg - comm.init_k_per_rank[p];
          cplx &x = comm.tau_allk_buff[comm.tau_allk_buff_index(nl, c, kl, kg)];
          E[1].chk(x, h1(c + 10, kg, T, comm.my_first_tau + nl), 0);
          x = h3(c + 10, kg, T, comm.my_first_tau + nl);
        }
    if (T < 0) {  // a ti = -1 get leaves the t' block unchanged
      E[4].chk(comm.my_first_t, -1, 0);
      E[4].chk(comm.my_Nt, 0, 0);
    }
  };
  if (mode == 0) {
    comm.mpi_get_and_comm_spawn(T, getsLG, getstv);
    body();
    comm.mpi_comm_and_set_spawn(T, setsLR, setstv);
  } else if (mode == 1) {
#pragma omp parallel
    {
      g_rank = rank;
      comm.mpi_get_and_comm_nospawn(T, getsLG, getstv);
#pragma omp master
      body();
#pragma omp barrier
      comm.mpi_comm_and_set_nospawn(T, setsLR, setstv);
#pragma omp barrier
    }
  } else if (mode == 2) {
#pragma omp parallel num_threads(2)
    {
      g_rank = rank;
      comm.mpi_get_and_comm_nospawn(T, getsLG, getstv);
#pragma omp master
      body();
#pragma omp barrier
      comm.mpi_comm_and_set_nospawn(T, setsLR, setstv);
    }
  } else {
    comm.mpi_get_and_comm_nospawn(T, getsLG, getstv);
    body();
    comm.mpi_comm_and_set_nospawn(T, setsLR, setstv);
  }
  for (int kl = 0; kl < comm.my_Nk; kl++)
    for (int c = 0; c < nc; c++) {
      int kg = comm.my_global_k_list[kl];
      if (T >= 0) {
        auto &v = recvLR[kl * nc + c];
        E[2].chk((double)v.size(), (double)(T + 1), 0);
        for (int tp = 0; tp <= T && tp < (int)v.size(); tp++) E[2].chk(v[tp], h3(c, kg, T, tp), 0);
      }
      auto &w = recvTV[kl * nc + c];
      E[3].chk((double)w.size(), (double)r, 0);
      for (int n = 0; n < r && n < (int)w.size(); n++) E[3].chk(w[n], h3(c + 10, kg, T, n), 0);
    }
  std::lock_guard<std::mutex> lk(g_print);
  for (int e = 0; e < 5; e++) {
    errs_out[e].nbad += E[e].nbad;
    errs_out[e].nchk += E[e].nchk;
    errs_out[e].maxdiff = std::max(errs_out[e].maxdiff, E[e].maxdiff);
  }
}

void launch(int P, const std::function<void()> &f) {
  g_P = P;
  g_a2a.assign(g_P, {});
  g_red.assign(g_P, 0.);
  g_ared.assign(g_P, {});
  std::barrier<> bar(g_P);
  g_bar = &bar;
  std::vector<std::thread> th;
  for (int p = 0; p < g_P; p++) th.emplace_back([p, &f]() {
    g_rank = p;
    f();
  });
  for (auto &t : th) t.join();
}

void check_highlevel(int P, int Nk, int nao, int k, int mode) {
  std::vector<Err> errs(9);
  g_cnt_mismatch = 0;
  launch(P, [&]() { run_rank(Nk, nao, k, mode, errs); });
  for (int e = 0; e < 9; e++) {
    EXPECT_GT(errs[e].nchk, 0) << NAMES[e];
    EXPECT_EQ(errs[e].nbad, 0) << NAMES[e] << " (maxdiff " << errs[e].maxdiff << ")";
  }
  EXPECT_EQ(g_cnt_mismatch.load(), 0) << "Alltoallv send/recv count mismatch";
}
}  // namespace

// (P, Nk, nao, k)
class MpiCommHighLevel : public ::testing::TestWithParam<std::tuple<int, int, int, int, int>> {};
TEST_P(MpiCommHighLevel, Routing) {
  auto [P, Nk, nao, k, mode] = GetParam();
  check_highlevel(P, Nk, nao, k, mode);
}
INSTANTIATE_TEST_SUITE_P(Comm, MpiCommHighLevel,
                         ::testing::Combine(::testing::Values(1, 3), ::testing::Values(7), ::testing::Values(1, 2), ::testing::Values(2),
                                            ::testing::Values(0, 1, 2, 3)));
INSTANTIATE_TEST_SUITE_P(CommMoreRanks, MpiCommHighLevel,
                         ::testing::Values(std::make_tuple(4, 4, 2, 3, 0), std::make_tuple(4, 4, 2, 3, 2), std::make_tuple(2, 9, 1, 1, 1),
                                           std::make_tuple(2, 9, 1, 1, 3), std::make_tuple(5, 3, 2, 2, 0), std::make_tuple(5, 3, 2, 2, 2)));

class MpiCommLambda : public ::testing::TestWithParam<std::tuple<int, int, int>> {};
void check_lambda(int P, int nao, int mode) {
  for (int T = -1; T <= 4; T++) {
    std::vector<Err> errs(5);
    g_cnt_mismatch = 0;
    launch(P, [&]() { run_rank_lambda(7, nao, T, mode, errs); });
    for (int e = 0; e < 5; e++) EXPECT_EQ(errs[e].nbad, 0) << "T=" << T << " check " << e << " (maxdiff " << errs[e].maxdiff << ")";
    EXPECT_GT(errs[4].nchk, 0);
    EXPECT_EQ(g_cnt_mismatch.load(), 0) << "Alltoallv send/recv count mismatch";
  }
}
TEST_P(MpiCommLambda, Routing) {
  auto [P, nao, mode] = GetParam();
  check_lambda(P, nao, mode);
}
INSTANTIATE_TEST_SUITE_P(Comm, MpiCommLambda,
                         ::testing::Combine(::testing::Values(1, 3), ::testing::Values(1, 2), ::testing::Values(0, 1)));

// B21 (fixed in WP5): _nospawn in a team smaller than omp_get_max_threads() (OMP_NUM_THREADS=4 for this
// executable), and _nospawn called outside any parallel region (team of 1: the caller does all the work; supported).
TEST(MpiCommNospawn, TeamSmallerThanMaxThreads) {
  hn_test::run_isolated([] {
    ASSERT_GE(omp_get_max_threads(), 3) << "test needs OMP_NUM_THREADS >= 3";
    check_highlevel(3, 7, 2, 2, 5);
  });
}
TEST(MpiCommNospawn, CalledOutsideParallel) {
  hn_test::run_isolated([] {
    ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
    check_highlevel(3, 7, 2, 2, 4);
  });
}
TEST(MpiCommNospawn, LambdaTeamSmallerThanMaxThreads) {
  ASSERT_GE(omp_get_max_threads(), 3) << "test needs OMP_NUM_THREADS >= 3";
  check_lambda(3, 2, 2);
}
TEST(MpiCommNospawn, LambdaCalledOutsideParallel) {
  ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
  check_lambda(3, 2, 3);
}

// B20 (fixed in WP5): the paper (App. C) promises an error when more components are sent than the buffers hold.
// The checks run at entry, before any parallel work or barrier, and throw std::invalid_argument.
namespace {
using GetF = std::function<cplx(int, int, int)>;
using SetF = std::function<void(int, int, std::vector<cplx> &)>;
std::vector<GetF> make_gets(int n) {
  std::vector<GetF> v;
  for (int c = 0; c < n; c++) v.push_back([c](int kl, int, int tp) { return cplx(c + kl, tp); });
  return v;
}
std::vector<SetF> make_sets(int n) {
  std::vector<SetF> v;
  for (int c = 0; c < n; c++) v.push_back([](int, int, std::vector<cplx> &) {});
  return v;
}
// calls f and records whether it threw std::invalid_argument (and its message)
void expect_invalid(const std::function<void()> &f, int &nthrow, std::string &msg) {
  try {
    f();
  } catch (const std::invalid_argument &e) {
    nthrow++;
    msg = e.what();
  }
}
}  // namespace

// (a) high-level API with ncomponents = 1 (it always writes component index nao^2): every entry point throws.
TEST(MpiCommChecks, TooFewComponentsHighLevel) {
  hn_test::run_isolated([] {
    int nthrow = 0, ncalls = 0;
    std::string msg;
    launch(1, [&]() {
      int nao = 2, Nk = 3, k = 2, nt = 8, r = 0;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      mpi_comm comm(Nk, nt, r, nao, 1);
      std::vector<herm_matrix_hodlr> G, S;
      for (int i = 0; i < comm.my_Nk; i++) {
        G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
        S.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      }
      std::vector<std::reference_wrapper<herm_matrix_hodlr>> Gr(G.begin(), G.end()), Sr(S.begin(), S.end());
      // the _nospawn calls are made outside a parallel region (team of 1), where the exception is catchable
      std::vector<std::function<void()>> calls = {
          [&] { comm.mpi_get_and_comm_spawn(1, G, dlr); },  [&] { comm.mpi_get_and_comm_spawn(1, Gr, dlr); },
          [&] { comm.mpi_get_and_comm_spawn(-1, G, dlr); }, [&] { comm.mpi_get_and_comm_nospawn(1, G, dlr); },
          [&] { comm.mpi_get_and_comm_nospawn(1, Gr, dlr); }, [&] { comm.mpi_comm_and_set_spawn(1, S); },
          [&] { comm.mpi_comm_and_set_spawn(-1, Sr); },     [&] { comm.mpi_comm_and_set_nospawn(1, S); },
          [&] { comm.mpi_comm_and_set_nospawn(1, Sr); }};
      for (auto &c : calls) {
        ncalls++;
        expect_invalid(c, nthrow, msg);
      }
    });
    EXPECT_EQ(nthrow, ncalls) << "not every high-level entry point rejects ncomponents=1";
    EXPECT_NE(msg.find("ncomponents >= 2"), std::string::npos) << msg;
  });
}

// (b) lambda API with more lambdas than ncomponents*nao^2: get and set, spawn and nospawn, both vectors.
TEST(MpiCommChecks, TooManyLambdas) {
  hn_test::run_isolated([] {
    int nthrow = 0, ncalls = 0, nok_throw = 0;
    std::string msg, msg_ok;
    launch(1, [&]() {
      int nao = 1, Nk = 3, nt = 8, r = 5, T = 2;
      mpi_comm comm(Nk, nt, r, nao, 1);  // room for 1 component
      auto g1 = make_gets(1), g3 = make_gets(3);
      auto s1 = make_sets(1), s3 = make_sets(3);
      std::vector<std::function<void()>> calls = {
          [&] { comm.mpi_get_and_comm_spawn(T, g3, g1); },   [&] { comm.mpi_get_and_comm_spawn(T, g1, g3); },
          [&] { comm.mpi_get_and_comm_spawn(-1, g1, g3); },  [&] { comm.mpi_get_and_comm_nospawn(T, g3, g1); },
          [&] { comm.mpi_get_and_comm_nospawn(T, g1, g3); }, [&] { comm.mpi_comm_and_set_spawn(T, s3, s1); },
          [&] { comm.mpi_comm_and_set_spawn(T, s1, s3); },   [&] { comm.mpi_comm_and_set_nospawn(T, s3, s1); },
          [&] { comm.mpi_comm_and_set_nospawn(T, s1, s3); }};
      for (auto &c : calls) {
        ncalls++;
        expect_invalid(c, nthrow, msg);
      }
      // getsLG/setsLR are not used for ti = -1, so their size is not checked there
      expect_invalid([&] { comm.mpi_get_and_comm_spawn(-1, g3, g1); }, nok_throw, msg_ok);
      expect_invalid([&] { comm.mpi_comm_and_set_spawn(-1, s3, s1); }, nok_throw, msg_ok);
      // exactly max_component_size_ lambdas is fine
      expect_invalid([&] { comm.mpi_get_and_comm_spawn(T, g1, g1); }, nok_throw, msg_ok);
      expect_invalid([&] { comm.mpi_comm_and_set_spawn(T, s1, s1); }, nok_throw, msg_ok);
    });
    EXPECT_EQ(nthrow, ncalls) << "not every lambda entry point rejects 3 lambdas with max_component_size 1";
    EXPECT_NE(msg.find("exceeds max_component_size_"), std::string::npos) << msg;
    EXPECT_EQ(nok_throw, 0) << msg_ok;
  });
}

// (c) WP6a side finding 2 (WP6b): the component-count checks follow the checks.hpp OpenMP policy. Inside a user parallel
// region (where _nospawn runs by contract) a failed check cannot be thrown out of the region: every thread prints the
// message (with MPI rank and thread) and the program aborts. Outside a region the same call throws (tests (a), (b)).
namespace {
void nospawn_bad_count_in_region(int which) {
  launch(1, [&]() {
    int nao = 1, Nk = 5, k = 2, nt = 8, r = 0, T = 2;
    dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
    mpi_comm comm(Nk, nt, r, nao, 1);
    std::vector<herm_matrix_hodlr> G;
    for (int i = 0; i < comm.my_Nk; i++) G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
    auto g3 = make_gets(3), g1 = make_gets(1);
#pragma omp parallel
    {
      g_rank = 0;
      if (which == 0) comm.mpi_get_and_comm_nospawn(T, G, dlr);
      else comm.mpi_get_and_comm_nospawn(T, g3, g1);
    }
  });
}
}  // namespace
TEST(MpiCommChecksDeathTest, NospawnCheckInsideRegionAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
  EXPECT_DEATH(nospawn_bad_count_in_region(0),
               "check failed inside a user OpenMP parallel region: h_nessi::mpi_comm::mpi_get_and_comm_nospawn: "
               "max_component_size_ >= 2\\*nao_\\*nao_ violated");
  EXPECT_DEATH(nospawn_bad_count_in_region(1), "exceeds max_component_size_");
}
// WP6b side finding 4 (fixed in WP9a): with MPI initialised, the abort of a failed check inside a user parallel region
// is MPI_Abort(MPI_COMM_WORLD, 1) (the shim's MPI_Abort prints and exits with the code), so other ranks end too.
TEST(MpiCommChecksDeathTest, CheckInsideRegionCallsMpiAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(nospawn_bad_count_in_region(0), ::testing::ExitedWithCode(1),
              "calling MPI_Abort\\(MPI_COMM_WORLD, 1\\)(.|\n)*shim MPI_Abort\\(1\\)");
}

// WP6a OpenMP error policy, item 3: an error raised inside a library-owned parallel region of a _spawn routine (a
// library check, or an exception from a user lambda) is captured and rethrown on the calling thread, so it is
// catchable; afterwards the object still works.
TEST(MpiCommChecks, SpawnRegionErrorIsRethrownCatchable) {
  hn_test::run_isolated([] {
    launch(1, [&]() {
      int nao = 1, Nk = 5, k = 2, nt = 8, r = 0, T = 2;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      int r2 = 0;
      dlr_info dlr2(r2, LAMBDA, 1e-6, BETA, nao, XI);  // another DLR rank: get_tv_reversed's dlr pairing check fails
      ASSERT_NE(r, r2);
      mpi_comm comm(Nk, nt, r, nao, 2);
      std::vector<herm_matrix_hodlr> G;
      for (int i = 0; i < comm.my_Nk; i++) G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      try {
        comm.mpi_get_and_comm_spawn(T, G, dlr2);
        ADD_FAILURE() << "no exception";
      } catch (const h_nessi::precondition_error &e) {
        // WP6b: the dlr pairing is checked at entry (MP-9) before the library region; the in-region capture is
        // exercised by the throwing user lambdas below
        EXPECT_NE(std::string(e.what()).find("h_nessi::mpi_comm::mpi_get_and_comm_spawn: dlr.r() == r_"), std::string::npos)
            << e.what();
      }
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(-1, G, dlr2), h_nessi::precondition_error);
      std::vector<std::function<cplx(int, int, int)>> bad = {[](int, int, int) -> cplx { throw std::out_of_range("user lambda"); }};
      auto g1 = make_gets(1);
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(T, bad, g1), std::out_of_range);
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(T, g1, bad), std::out_of_range);
      EXPECT_NO_THROW(comm.mpi_get_and_comm_spawn(T, G, dlr));
      EXPECT_NO_THROW(comm.mpi_comm_and_set_spawn(T, G));
    });
  });
}

// H24 (fixed in WP5): mpi_comm_and_set_nospawn ends with a barrier. The set lambda of the last local k-point
// (handled by the last thread) is slowed down; thread 0 checks right after the call, without its own barrier.
TEST(MpiCommNospawn, SetLambdaEndsWithBarrier) {
  ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
  std::atomic<long> nbad{0}, nchk{0};
  launch(1, [&]() {
    const int nao = 1, Nk = 8, nt = 8, r = 5, T = 3;
    mpi_comm comm(Nk, nt, r, nao, 1);
    const int myNk = comm.my_Nk;
    std::unique_ptr<std::atomic<int>[]> doneLR(new std::atomic<int>[myNk]), doneTV(new std::atomic<int>[myNk]);
    for (int i = 0; i < myNk; i++) doneLR[i] = doneTV[i] = 0;
    auto g1 = make_gets(1);
    std::vector<SetF> sLR = {[&](int kl, int, std::vector<cplx> &) {
      if (kl == myNk - 1) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      doneLR[kl].store(1, std::memory_order_release);
    }};
    std::vector<SetF> sTV = {[&](int kl, int, std::vector<cplx> &) {
      if (kl == myNk - 1) std::this_thread::sleep_for(std::chrono::milliseconds(100));
      doneTV[kl].store(1, std::memory_order_release);
    }};
#pragma omp parallel
    {
      g_rank = 0;
      comm.mpi_get_and_comm_nospawn(T, g1, g1);
      comm.mpi_comm_and_set_nospawn(T, sLR, sTV);
      // no barrier here on purpose
      for (int kl = 0; kl < myNk; kl++) {
        nchk++;
        if (doneLR[kl].load(std::memory_order_acquire) != 1 || doneTV[kl].load(std::memory_order_acquire) != 1) nbad++;
      }
    }
  });
  EXPECT_GT(nchk.load(), 0);
  EXPECT_EQ(nbad.load(), 0) << "a thread returned from mpi_comm_and_set_nospawn before all set lambdas finished";
}

// H24, high-level API: right after mpi_comm_and_set_nospawn (no barrier of its own) every thread reads every local
// k-point of S, including those written by other threads, and must see the final data (both ti >= 0 and ti = -1).
TEST(MpiCommNospawn, HighLevelSetEndsWithBarrier) {
  ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
  std::atomic<long> nbad{0}, nchk{0};
  launch(1, [&]() {
    const int nao = 2, Nk = 8, nt = 8, k = 2, es = nao * nao;
    const double tol = 1e-12;  // values are computed in different code locations (FMA contraction may differ)
    int r = 0;
    dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
    mpi_comm comm(Nk, nt, r, nao);
    std::vector<herm_matrix_hodlr> G, S;
    for (int i = 0; i < comm.my_Nk; i++) {
      G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      S.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
    }
    for (int T : {-1, 2}) {
#pragma omp parallel
      {
        g_rank = 0;
        comm.mpi_get_and_comm_nospawn(T, G, dlr);
#pragma omp master
        for (int kg = 0; kg < Nk; kg++) {
          if (T >= 0) {
            for (int t = 0; t <= T; t++)
              for (int i = 0; i < nao; i++)
                for (int j = 0; j < nao; j++) {
                  comm.map_ret(kg, t)(i, j) = gR(kg, T, t, i, j);
                  comm.map_les(kg, t)(i, j) = gL(kg, T, t, i, j);
                }
            for (int n = 0; n < r; n++)
              for (int i = 0; i < nao; i++)
                for (int j = 0; j < nao; j++) comm.map_tv(kg, n)(i, j) = gTV(kg, T, n, i, j);
          } else {
            for (int n = 0; n < r; n++)
              for (int i = 0; i < nao; i++)
                for (int j = 0; j < nao; j++) comm.map_mat(kg, n)(i, j) = gM(kg, n, i, j);
          }
        }
#pragma omp barrier
        comm.mpi_comm_and_set_nospawn(T, S);
        // no barrier here on purpose
        for (int kl = 0; kl < comm.my_Nk; kl++) {
          int kg = comm.my_global_k_list[kl];
          for (int e = 0; e < es; e++) {
            int i = e / nao, j = e % nao;
            nchk++;
            if (T >= 0) {
              for (int t = 0; t <= T; t++)
                if (std::abs(S[kl].curr_timestep_ret_ptr(T, t)[e] - gR(kg, T, t, i, j)) > tol ||
                    std::abs(S[kl].curr_timestep_les_ptr(t, T)[e] - gL(kg, T, t, i, j)) > tol)
                  nbad++;
              for (int n = 0; n < r; n++)
                if (std::abs(S[kl].tvptr(T, n)[e] - gTV(kg, T, n, i, j)) > tol) nbad++;
            } else {
              for (int n = 0; n < r; n++)
                if (std::abs(S[kl].matptr(n)[e] - gM(kg, n, i, j).real()) > tol) nbad++;
            }
          }
        }
#pragma omp barrier
      }
    }
  });
  EXPECT_GT(nchk.load(), 0);
  EXPECT_EQ(nbad.load(), 0);
}

// H30 (fixed in WP5): my_first_t/my_Nt are -1/0 after construction (both constructors), are set by every get with
// ti >= 0 (high-level and lambda, spawn and nospawn) to the rank's t' block, and are unchanged by a ti = -1 get.
TEST(MpiCommState, TBlockMembers) {
  {
    mpi_comm c0;
    EXPECT_EQ(c0.my_first_t, -1);
    EXPECT_EQ(c0.my_Nt, 0);
  }
  for (int P : {1, 3}) {
    std::atomic<long> nbad{0};
    launch(P, [&]() {
      const int rank = g_rank;
      auto expect_block = [&](mpi_comm &c, int ti) {
        int f = -1, n = 0;
        if (ti >= 0) {
          auto b = get_my_index(rank, P, ti + 1, 0);
          f = b[0];
          n = b[2];
        }
        if (c.my_first_t != f || c.my_Nt != n) nbad++;
      };
      mpi_comm comm(7, 12, 5, 1, 2);
      expect_block(comm, -1);
      auto g1 = make_gets(1);
      comm.mpi_get_and_comm_spawn(-1, g1, g1);
      expect_block(comm, -1);
      comm.mpi_get_and_comm_spawn(4, g1, g1);
      expect_block(comm, 4);
      comm.mpi_get_and_comm_spawn(-1, g1, g1);
      expect_block(comm, 4);
      comm.mpi_get_and_comm_nospawn(6, g1, g1);  // outside a region
      expect_block(comm, 6);
#pragma omp parallel
      {
        g_rank = rank;
        comm.mpi_get_and_comm_nospawn(2, g1, g1);
        expect_block(comm, 2);  // every thread, right after the call
      }
      // high-level API
      int r = 0, nao = 1, nt = 8, k = 2;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      mpi_comm ch(7, nt, r, nao);
      expect_block(ch, -1);
      std::vector<herm_matrix_hodlr> G;
      for (int i = 0; i < ch.my_Nk; i++) G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      ch.mpi_get_and_comm_spawn(-1, G, dlr);
      expect_block(ch, -1);
      ch.mpi_get_and_comm_spawn(1, G, dlr);
      expect_block(ch, 1);
#pragma omp parallel
      {
        g_rank = rank;
        ch.mpi_get_and_comm_nospawn(2, G, dlr);
        expect_block(ch, 2);
      }
    });
    EXPECT_EQ(nbad.load(), 0) << "P=" << P;
  }
}

// H28 (WP5): the constructor warns once (rank 0, stderr) if MPI_Query_thread reports less than MPI_THREAD_FUNNELED.
TEST(MpiCommThreadLevel, WarnsBelowFunneled) {
  for (int level : {MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_MULTIPLE}) {
    g_thread_level = level;
    testing::internal::CaptureStderr();
    launch(3, [&]() { mpi_comm comm(7, 12, 5, 1); });
    std::string err = testing::internal::GetCapturedStderr();
    size_t n = 0;
    for (size_t pos = err.find("h_nessi::mpi_comm warning"); pos != std::string::npos; pos = err.find("h_nessi::mpi_comm warning", pos + 1)) n++;
    EXPECT_EQ(n, level < MPI_THREAD_FUNNELED ? 1u : 0u) << "level " << level << ": " << err;
    if (level < MPI_THREAD_FUNNELED) EXPECT_NE(err.find("MPI_Init_thread"), std::string::npos) << err;
  }
  g_thread_level = MPI_THREAD_FUNNELED;
}

// ------------------------------------------------------------------ WP6b entry checks (doc 61 MP-*, ST-9)
TEST(MpiCommChecksWP6b, ConstructorAndThreadLevel) {
  hn_test::run_isolated([] {
    launch(1, [&]() {
      EXPECT_THROW(mpi_comm(0, 8, 5, 1), h_nessi::precondition_error);  // MP-1
      EXPECT_THROW(mpi_comm(4, 8, 0, 1), h_nessi::precondition_error);
      EXPECT_THROW(mpi_comm(4, 8, 5, 1, 0), h_nessi::precondition_error);
      g_mpi_initialized = 0;  // MP-2
      EXPECT_THROW(mpi_comm(4, 8, 5, 1), h_nessi::state_error);
      g_mpi_initialized = 1;
      // MP-3: below FUNNELED the constructor warns, the _nospawn routines refuse, the _spawn routines work
      g_thread_level = MPI_THREAD_SINGLE;
      testing::internal::CaptureStderr();
      mpi_comm comm(4, 8, 5, 1, 2);
      testing::internal::GetCapturedStderr();
      g_thread_level = MPI_THREAD_FUNNELED;
      auto g1 = make_gets(1);
      auto s1 = make_sets(1);
      EXPECT_THROW(comm.mpi_get_and_comm_nospawn(2, g1, g1), h_nessi::state_error);
      EXPECT_NO_THROW(comm.mpi_get_and_comm_spawn(2, g1, g1));
      EXPECT_THROW(comm.mpi_comm_and_set_nospawn(2, s1, s1), h_nessi::state_error);
      EXPECT_NO_THROW(comm.mpi_comm_and_set_spawn(2, s1, s1));
    });
  });
}

TEST(MpiCommChecksWP6b, TimeIndexObjectsAndDlr) {
  hn_test::run_isolated([] {
    launch(1, [&]() {
      int nao = 1, Nk = 4, k = 2, nt = 8, r = 0;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      int r2 = 0;
      dlr_info dlr_nao(r2, LAMBDA, EPSDLR, BETA, 2, XI);
      mpi_comm comm(Nk, nt, r, nao, 2);
      std::vector<herm_matrix_hodlr> G, Gshort, G2;
      for (int i = 0; i < comm.my_Nk; i++) {
        G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
        G2.emplace_back(nt, r, 1, 1e-10, 2, 2, XI, k);
      }
      for (int i = 0; i + 1 < comm.my_Nk; i++) Gshort.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      auto g1 = make_gets(1);
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(nt, g1, g1), h_nessi::precondition_error);  // MP-6
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(-2, G, dlr), h_nessi::precondition_error);
      try {
        comm.mpi_get_and_comm_spawn(1, Gshort, dlr);  // MP-8: one object per local k-point
        ADD_FAILURE() << "no exception";
      } catch (const h_nessi::precondition_error &e) {
        EXPECT_NE(std::string(e.what()).find("one object per local k-point"), std::string::npos) << e.what();
      }
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(1, G2, dlr), h_nessi::precondition_error);  // nao
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(k + 1, G, dlr), h_nessi::precondition_error);  // row k+1 not in window
      EXPECT_THROW(comm.mpi_get_and_comm_spawn(1, G, dlr_nao), h_nessi::precondition_error);  // MP-9
      EXPECT_NO_THROW(comm.mpi_get_and_comm_spawn(k, G, dlr));
      EXPECT_THROW(comm.mpi_comm_and_set_spawn(k, Gshort), h_nessi::precondition_error);
      EXPECT_NO_THROW(comm.mpi_comm_and_set_spawn(k, G));
    });
  });
}

// ST-9 (MP-5, MP-11): a set reuses the counts/displacements of the preceding get
TEST(MpiCommChecksWP6b, SetFollowsMatchingGet) {
  hn_test::run_isolated([] {
    launch(1, [&]() {
      int nao = 1, Nk = 4, k = 2, nt = 8, r = 0;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      mpi_comm comm(Nk, nt, r, nao, 3);
      std::vector<herm_matrix_hodlr> G;
      for (int i = 0; i < comm.my_Nk; i++) G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      auto g1 = make_gets(1), g2 = make_gets(2);
      auto s1 = make_sets(1), s2 = make_sets(2);
      try {
        comm.mpi_comm_and_set_spawn(1, s1, s1);  // no get yet
        ADD_FAILURE() << "no exception";
      } catch (const h_nessi::state_error &e) {
        EXPECT_NE(std::string(e.what()).find("a set must follow a get with the same ti"), std::string::npos) << e.what();
      }
      comm.mpi_get_and_comm_spawn(1, g1, g1);
      EXPECT_THROW(comm.mpi_comm_and_set_spawn(2, s1, s1), h_nessi::state_error);  // other ti
      EXPECT_THROW(comm.mpi_comm_and_set_spawn(1, s2, s1), h_nessi::state_error);  // other counts
      EXPECT_THROW(comm.mpi_comm_and_set_spawn(1, G), h_nessi::state_error);       // other API level
      EXPECT_NO_THROW(comm.mpi_comm_and_set_spawn(1, s1, s1));
      EXPECT_THROW(comm.mpi_comm_and_set_spawn(1, s1, s1), h_nessi::state_error);  // one set per get
      comm.mpi_get_and_comm_spawn(-1, g2, g1);
      EXPECT_NO_THROW(comm.mpi_comm_and_set_spawn(-1, s1, s1));  // ti = -1: getsLG/setsLR are not used
      comm.mpi_get_and_comm_spawn(k, G, dlr);
      EXPECT_THROW(comm.mpi_comm_and_set_nospawn(k, s2, s2), h_nessi::state_error);
      EXPECT_NO_THROW(comm.mpi_comm_and_set_nospawn(k, G));
    });
  });
}

// MP-13 (Q-9: warn): the ti = -1 set drops an imaginary Sigma^M
TEST(MpiCommChecksWP6b, ImaginarySigmaMatWarnsOnce) {
  hn_test::run_isolated([] {
    std::string err;
    launch(1, [&]() {
      int nao = 1, Nk = 4, k = 2, nt = 8, r = 0;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      mpi_comm comm(Nk, nt, r, nao, 2);
      std::vector<herm_matrix_hodlr> G;
      for (int i = 0; i < comm.my_Nk; i++) G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      testing::internal::CaptureStderr();
      for (int rep = 0; rep < 2; rep++) {
        comm.mpi_get_and_comm_spawn(-1, G, dlr);
        for (int kg = 0; kg < Nk; kg++)
          for (int n = 0; n < r; n++) comm.map_mat(kg, n)(0, 0) = cplx(1.0, 0.01);
        comm.mpi_comm_and_set_spawn(-1, G);
      }
      err = testing::internal::GetCapturedStderr();
    });
    size_t n = 0;
    for (size_t pos = err.find("relative imaginary part"); pos != std::string::npos; pos = err.find("relative imaginary part", pos + 1)) n++;
    EXPECT_EQ(n, 1u) << err;
  });
}

// WP6a side finding 4: an uncaught exception after an mpi_comm was constructed ends in MPI_Abort (with the message)
namespace {
void terminate_after_mpi_comm() {
  launch(1, [&]() { mpi_comm comm(4, 8, 5, 1); });
  try {
    throw std::runtime_error("uncaught on purpose");
  } catch (...) {
    std::terminate();
  }
}
}  // namespace
TEST(MpiCommChecksWP6bDeathTest, TerminateHandlerCallsMpiAbort) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(terminate_after_mpi_comm(), "std::terminate on MPI rank 0: uncaught on purpose");
  EXPECT_DEATH(terminate_after_mpi_comm(), "shim MPI_Abort\\(1\\)");
}

// MP-15..17 (tier 1): the buffer maps
TEST(MpiCommChecksWP6bDeathTest, Tier1Maps) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#ifndef NDEBUG
  auto body = [](int which) {
    launch(1, [&]() {
      mpi_comm comm(4, 8, 5, 1, 2);
      if (which == 0) comm.map_ret(0, 0);  // before any get: my_first_t = -1
      auto g1 = make_gets(1);
      comm.mpi_get_and_comm_spawn(3, g1, g1);
      if (which == 1) comm.map_les(4, 0);  // k out of range
      if (which == 2) comm.map_tv(0, 5);   // tau out of range
      if (which == 3) comm.t_allk_buff_index(4, 0, 0, 0);  // t' out of this rank's block (my_Nt = 4)
    });
  };
  EXPECT_DEATH(body(0), "mpi_comm::map_ret");
  EXPECT_DEATH(body(1), "mpi_comm::map_les");
  EXPECT_DEATH(body(2), "mpi_comm::map_tv");
  EXPECT_DEATH(body(3), "mpi_comm::t_allk_buff_index");
#else
  GTEST_SKIP() << "asserts are off (NDEBUG): the map checks are tier 1";
#endif
}
