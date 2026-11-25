// mpi_comm routing (doc 31) with simulated ranks.  Ported from agent_scratch/31/commtest.cpp.
// The four MPI symbols used by libh_nessi's mpi_comm (MPI_Comm_size, MPI_Comm_rank, MPI_Alltoallv, MPI_Reduce)
// are interposed by definitions in this executable; P "ranks" run as std::threads in one process, so the
// library's multi-rank Alltoallv routing is exercised unchanged without an MPI launcher.
// Known bugs: B21 (_nospawn partitions work with omp_get_max_threads), B20 (no component-count check).
#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
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
extern "C" {
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
                        "up ret/les -> S curr buffers", "up tv/mat -> S", "S.tv_trans untouched (stays 0)",
                        "G.tv_trans == scratch G(.,beta-tau) untransposed", "allk index formula"};

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
              E[6].chk(S[kl].tvptr_trans(T, n)[i * nao + j], 0.0, 0);
              E[7].chk(G[kl].tvptr_trans(T, n)[i * nao + j], fTV(kg, T, BETA - taus[n], i, j), 1e-8);
            }
      } else {
        for (int n = 0; n < r; n++)
          for (int i = 0; i < nao; i++)
            for (int j = 0; j < nao; j++) {
              E[5].chk(S[kl].matptr(n)[i * nao + j], gM(kg, n, i, j).real(), 1e-12);  // mat passes a DLR round trip
              E[7].chk(G[kl].tvptr_trans(0, n)[i * nao + j], fM(kg, BETA - taus[n], i, j), 1e-8);
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
void run_rank_lambda(int Nk, int nao, int T, bool nospawn, std::vector<Err> &errs_out) {
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
  std::vector<Err> E(4);
  auto body = [&]() {
    if (T >= 0) {
      auto ti = get_my_index(rank, P, T + 1, 0);
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
  };
  if (!nospawn) {
    comm.mpi_get_and_comm_spawn(T, getsLG, getstv);
    body();
    comm.mpi_comm_and_set_spawn(T, setsLR, setstv);
  } else {
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
  for (int e = 0; e < 4; e++) {
    errs_out[e].nbad += E[e].nbad;
    errs_out[e].nchk += E[e].nchk;
    errs_out[e].maxdiff = std::max(errs_out[e].maxdiff, E[e].maxdiff);
  }
}

void launch(int P, const std::function<void()> &f) {
  g_P = P;
  g_a2a.assign(g_P, {});
  g_red.assign(g_P, 0.);
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
TEST_P(MpiCommLambda, Routing) {
  auto [P, nao, nospawn] = GetParam();
  for (int T = -1; T <= 4; T++) {
    std::vector<Err> errs(4);
    launch(P, [&]() { run_rank_lambda(7, nao, T, nospawn, errs); });
    for (int e = 0; e < 4; e++) EXPECT_EQ(errs[e].nbad, 0) << "T=" << T << " check " << e;
  }
}
INSTANTIATE_TEST_SUITE_P(Comm, MpiCommLambda,
                         ::testing::Combine(::testing::Values(1, 3), ::testing::Values(1, 2), ::testing::Values(0, 1)));

// B21: _nospawn in a team smaller than omp_get_max_threads() (OMP_NUM_THREADS=4 for this executable).
TEST(MpiCommNospawn, KnownBug_B21_TeamSmallerThanMaxThreads) {
  hn_test::run_isolated([] {
    ASSERT_GE(omp_get_max_threads(), 3) << "test needs OMP_NUM_THREADS >= 3";
    check_highlevel(3, 7, 2, 2, 5);
  });
}
TEST(MpiCommNospawn, KnownBug_B21_CalledOutsideParallel) {
  hn_test::run_isolated([] {
    ASSERT_GE(omp_get_max_threads(), 2) << "test needs OMP_NUM_THREADS >= 2";
    check_highlevel(3, 7, 2, 2, 4);
  });
}

// B20: the paper (App. C) promises an error when more components are sent than the buffers hold.
// (a) high-level API with ncomponents = 1 (it always writes component index nao^2): must throw.
TEST(MpiCommChecks, KnownBug_B20_TooFewComponentsHighLevel) {
  hn_test::run_isolated([] {
    bool threw = false;
    launch(1, [&]() {
      int nao = 2, Nk = 3, k = 2, nt = 8, r = 0;
      dlr_info dlr(r, LAMBDA, EPSDLR, BETA, nao, XI);
      mpi_comm comm(Nk, nt, r, nao, 1);
      std::vector<herm_matrix_hodlr> G, S;
      for (int i = 0; i < comm.my_Nk; i++) {
        G.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
        S.emplace_back(nt, r, 1, 1e-10, nao, nao, XI, k);
      }
      try {
        comm.mpi_get_and_comm_spawn(1, G, dlr);
        comm.mpi_comm_and_set_spawn(1, S);
      } catch (...) {
        threw = true;
      }
    });
    EXPECT_TRUE(threw) << "no error for ncomponents=1 with the high-level API";
  });
}
// (b) lambda API with more lambdas than ncomponents*nao^2: must throw.
TEST(MpiCommChecks, KnownBug_B20_TooManyLambdas) {
  hn_test::run_isolated([] {
    bool threw = false;
    launch(1, [&]() {
      int nao = 1, Nk = 3, nt = 8, r = 5, T = 2;
      mpi_comm comm(Nk, nt, r, nao, 1);  // room for 1 component
      std::vector<std::function<cplx(int, int, int)>> gets, getstv;
      for (int c = 0; c < 3; c++) {
        gets.push_back([c](int, int, int tp) { return cplx(c, tp); });
        getstv.push_back([c](int, int, int n) { return cplx(c, n); });
      }
      try {
        comm.mpi_get_and_comm_spawn(T, gets, getstv);
      } catch (...) {
        threw = true;
      }
    });
    EXPECT_TRUE(threw) << "no error for 3 lambdas with max_component_size 1";
  });
}
