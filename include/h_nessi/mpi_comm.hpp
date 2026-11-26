/**
 * @file mpi_comm.hpp
 * @brief Defines the mpi_comm class for distributed communication and data exchange using MPI and OpenMP.
 */
#pragma once

#include <iostream>
#include <cstddef>
#include <complex>
#include <vector>
#include <functional>
#include <mpi.h>
#include <omp.h>
#include <climits>

#include "mpi_comm_utils.hpp"
#include "herm_matrix_hodlr.hpp"

namespace h_nessi {

/**
 * @class mpi_comm
 * @brief Facilitates distributed communication and data exchange for herm_matrix_hodlr objects using MPI and OpenMP.
 *
 * The class is designed for flexible data movement between distributed herm_matrix_hodlr objects and internal buffers.
 * Users provide custom get and set functions:
 *   - Get functions: Copy data from herm_matrix_hodlr objects to the class buffers.
 *   - Set functions: Copy data from the class buffers back to herm_matrix_hodlr objects.
 * This allows the class to be agnostic to the actual data layout and storage of the matrices.
 *
 * The class manages different buffers, using them as needed depending on the type of data required at each step of the calculation.
 * It also handles buffer management, data indexing, and communication patterns for distributed arrays, abstracting away MPI/OpenMP details from the user.
 *
 * Threading contract:
 *   - `*_spawn` routines must be called outside an OpenMP parallel region; they open their own `omp parallel for`.
 *   - `*_nospawn` routines must be called by **every** thread of the current team (they contain orphaned
 *     `#pragma omp barrier`s). Work is partitioned over the current team (`omp_get_num_threads()`), so any team
 *     size works, including a `num_threads(n)` team and a call outside any parallel region (team of 1, the
 *     calling thread does all the work). MPI is called only by thread 0 of the team, between barriers, so MPI
 *     must provide at least `MPI_THREAD_FUNNELED` and thread 0 must be the thread that initialised MPI (true for
 *     a non-nested region started from the main thread). The constructor warns on rank 0 if
 *     `MPI_Query_thread` reports a lower level, and every `_nospawn` call requires it (MP-3, error). Every
 *     `_nospawn` routine ends with a barrier, so on return all threads see all data written by the routine
 *     (buffers and herm_matrix_hodlr objects).
 *   - The user get/set lambdas of the lambda API run concurrently on several threads (different k-points). A
 *     dlr_info is single-thread for `eval_point` (ST-10): a lambda that calls it on a shared dlr_info (directly or
 *     through get_mat_tau*, get_tv_tau*, density_matrix(-1), ...) throws h_nessi::state_error (in a `_nospawn`
 *     region: aborts) when two threads overlap. Give each thread its own dlr_info. The high-level gets only read
 *     the dlr_info's matrices (get_*_reversed), which is safe to share.
 *
 * Errors (WP6b, h_nessi/checks.hpp policy): every get/set checks at entry, before any parallel work or barrier,
 *   - -1 <= ti < Nt (MP-6), the component counts (MP-4/5/7, see below),
 *   - high-level API: one herm_matrix_hodlr per local k-point, each nao x nao with DLR rank r, nt > ti and, for
 *     ti >= 0, row ti in its current window (MP-8); the dlr_info of a get has rank r and nao orbitals (MP-9),
 *   - a set follows a get with the same ti, API level and component counts (ST-9: the set reuses the get's
 *     counts and displacements; one set per get),
 *   - `_nospawn`: MPI_THREAD_FUNNELED (MP-3).
 * A failure throws h_nessi::precondition_error (a std::invalid_argument) or h_nessi::state_error outside a
 * parallel region (all `_spawn` calls, and `_nospawn` called outside a region). Inside a user OpenMP region (a
 * `_nospawn` call made, by contract, by every thread) the checks depend only on shared arguments, so every thread
 * fails identically: each prints the message (MPI rank, thread) and the program aborts; it cannot be caught there.
 * Errors inside the library-owned regions of the `_spawn` routines (also exceptions from user lambdas) are
 * captured and rethrown on the calling thread. In Debug/RelWithAsserts builds every get/set also verifies with one
 * MPI_Allreduce that all ranks pass the same ti and counts (MP-18).
 * Uncaught exceptions: the constructor installs (once per process) a std::terminate handler that prints the
 * exception with the MPI rank and calls MPI_Abort(MPI_COMM_WORLD, 1) while MPI is initialised (WP6a side
 * finding 4), so an uncaught error on one rank ends the whole job instead of leaving the other ranks waiting in
 * a collective. A handler installed before the first mpi_comm is chained (called if MPI is not active).
 *
 * Component counts: each (time, k) slot holds `max_component_size_ = ncomponents*nao*nao` complex numbers.
 * The high-level (herm_matrix_hodlr) API uses components `[0, 2*nao*nao)` and therefore requires
 * `ncomponents >= 2`. The lambda API requires `getsLG.size()`, `getstv.size()`, `setsLR.size()`,
 * `setstv.size() <= max_component_size_` (getsLG/setsLR are checked only when ti >= 0, where they are used).
 */
class mpi_comm{
public:

    /**
     * @brief Default constructor.
     */
    mpi_comm();

    /**
     * @brief Parameterized constructor for initializing communication parameters.
     * @param Nk Number of k indices
     * @param Nt Number of t indices
     * @param r Number of representative tau points in the DLR representation
     * @param nao Number of orbital indices
     * @param ncomponents Number of Keldysh components to communicate (default 2)
     */
    mpi_comm(int Nk, 
       int Nt,
       int r,
       int nao,
       int ncomponents = 2
      );

    /**
     * @brief Returns approximate heap memory used by this instance as a human-readable value (MiB).
     *
     * This sums the heap allocations of the internal std::vector buffers (using capacity()),
     * including nested vectors, and converts the total to mebibytes (MiB = 1024*1024 bytes).
     * It does not include allocator bookkeeping or fragmentation.
     * @return Approximate memory allocated by internal containers in MiB (double)
     */
    double memory_usage() const;

    /**
     * @brief Computes the index in a buffer with all tp points.
     * @param local_Nk Local Nk
     * @param tpi t' index
     * @param local_ki local k index
     * @param compi Component index
     * @return Index in the alltp buffer
     */
    int alltp_buff_index(int local_Nk, int tpi, int local_ki, int compi);

    /**
     * @brief Computes the index in the buffer with all k points for time step t.
     * @param tpi t' index
     * @param compi Component index
     * @param local_ki local k index 
     * @param global_ki global k index
     * @return Index in the allk buffer for t
     */
    int t_allk_buff_index(int tpi, int compi, int local_ki, int global_ki);

    /**
     * @brief L
     * @param taui tau index
     * @param compi Component index
     * @param local_ki local k index 
     * @param global_ki global k index
     * @return Index in the allk buffer for tau
     */
    int tau_allk_buff_index(int taui, int compi, int local_ki, int global_ki);

    /**
     * @brief Gathers and communicates data
     * @param ti Time index
     * @param hmh_vec vector of herm_matrix_hodlr
     */
    void mpi_get_and_comm_spawn
    (int ti,
     std::vector<herm_matrix_hodlr> &hmh_vec,
     dlr_info &dlr
    );
    void mpi_get_and_comm_spawn
    (int ti,
     std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec,
     dlr_info &dlr
    );
    void mpi_get_and_comm_nospawn
    (int ti,
     std::vector<herm_matrix_hodlr> &hmh_vec,
     dlr_info &dlr
    );
    void mpi_get_and_comm_nospawn
    (int ti,
     std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec,
     dlr_info &dlr
    );



    /**
     * @brief Communicates and sets data, spawning threads at each call.
     * @param ti Time index
     * @param hmh_vec vector of herm_matrix_hodlr
     */
    void mpi_comm_and_set_spawn
    (int ti,
     std::vector<herm_matrix_hodlr> &hmh_vec
    );
    void mpi_comm_and_set_spawn
    (int ti,
     std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec
    );
    void mpi_comm_and_set_nospawn
    (int ti,
     std::vector<herm_matrix_hodlr> &hmh_vec
    );
    void mpi_comm_and_set_nospawn
    (int ti,
     std::vector<std::reference_wrapper<herm_matrix_hodlr>> &hmh_vec
    );



    /**
     * @brief Gathers and communicates data using provided get functions, spawning threads at each call.
     * @param ti Time index
     * @param getsLG get functions for Lesser and Greater components
     * @param getstv get functions for tv component
     */
    void mpi_get_and_comm_spawn
    (int ti,
     std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
     std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
    );

    /**
     * @brief Gathers and communicates data using provided get functions, without spawning threads at each call. Assumes omp parallel region.
     * @param ti Time index
     * @param getsLG get functions for Lesser and Greater components
     * @param getstv get functions for tv component
     */
    void mpi_get_and_comm_nospawn
    (int ti,
     std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
     std::vector<std::function<std::complex<double>(int, int, int)>> &getstv
    );


  /**
   * @brief Communicates and sets data using provided set functions, spawning threads at each call.
   * @param ti Time index
   * @param setsLR set functions for Lesser and Retarded components
   * @param setstv set functions for tv component
   */
  void mpi_comm_and_set_spawn
  (int ti,
   std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
   std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
  );

  /**
   * @brief Communicates and sets data using provided set functions, without spawning threads at each call.
   * @param ti Time index
   * @param setsLR set functions for Lesser and Retarded components
   * @param setstv set functions for tv component
   */
  void mpi_comm_and_set_nospawn
  (int ti,
   std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLR,
   std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv
  );


  // Easy access to arrays.  map_ret/map_les use my_first_t: valid only after a get with ti >= 0.  Only give access to *_allk_buff, as that is what the user needs to modify
  ZMatrixMap map_ret(int k_global, int t_global);
  ZMatrixMap map_les(int k_global, int t_global);
  ZMatrixMap map_tv(int k_global, int tau_global);
  ZMatrixMap map_mat(int k_global, int tau_global);
  ZMatrixMap map_tv_rev(int k_global, int tau_global);
  ZMatrixMap map_mat_rev(int k_global, int tau_global);


  /// Buffers containing all tp points for tau and t data
  std::vector<std::complex<double>> tau_alltp_buff, t_alltp_buff;
  /// Buffers containing all k points for tau and t data
  std::vector<std::complex<double>> tau_allk_buff, t_allk_buff;
  /// Scratch for the high-level get: G^rceil(ti, beta-tau) (ti >= 0) or G^M(beta-tau) (ti = -1) of local k-point
  /// krank, r_ x nao_ x nao_ per local k-point (my_Nk slices; each slice is used only by the thread handling krank).
  /// Replaces the former use of G.tv_trans as scratch (WP6c, ISSUES H27).
  std::vector<std::complex<double>> rev_buff_;

  /// Displacement and count vectors for tau communication
  std::vector<int> tau_send_displs, tau_send_counts;
  std::vector<int> tau_recv_displs, tau_recv_counts;

  /// Displacement and count vectors for t communication
  std::vector<int> t_send_displs, t_send_counts;
  std::vector<int> t_recv_displs, t_recv_counts;

  /// @brief Maps local k-point indices at each MPI rank to the global k-point index.
  std::vector<int> kindex_rank;
  /// @brief Maps global k index to the MPI rank responsible for that k-point. k_rank_map[global_k] = rank_that_does_that_k
  std::vector<int> k_rank_map;

  /// Number of k points per rank
  std::vector<int> init_k_per_rank;
  std::vector<int> Nk_per_rank;
  
  /// Distribution of tau indices per rank
  std::vector<int> init_tau_per_rank;
  std::vector<int> Ntau_per_rank;
  int my_Ntau = 0;
  int my_first_tau = 0;
  /// Block of real-time points t' = my_first_t .. my_first_t+my_Nt-1 that this rank holds in t_allk_buff.
  /// Set (identically) by every get with ti >= 0, high-level and lambda API: the block of
  /// get_my_index(rank, mpi_size, ti+1, 0). Before the first such get: my_first_t = -1, my_Nt = 0.
  /// A get with ti = -1 leaves them unchanged.
  int my_Nt = 0;
  int my_first_t = -1;
  int my_Nk = 0;
  std::vector<int> my_global_k_list;

  /// Parameters for buffer and communication sizing
  int Nk_ = 0;
  int Nt_ = 0;
  /// Number of representative tau points in the DLR representation
  int r_ = 0;
  /// Maximum number of components of herm_matrix_hodlr objects to be communicated, can also be orbital components
  int max_component_size_ = 0;
  /// Number of orbitals, max_component_size_ = ncomponents*nao*nao
  int nao_ = 0;

private:
  /// Slice of rev_buff_ for local k-point krank (r_ x nao_ x nao_).
  std::complex<double> *rev_slice_(int krank) { return rev_buff_.data() + (std::size_t)krank * r_ * nao_ * nao_; }
  // ---- checks (checks.hpp policy: h_nessi::precondition_error / state_error; inside a user OpenMP region (the
  // _nospawn routines) a failed check prints and aborts, identically on every thread) ------------------------------
  /// MP-4, MP-5 (B20): ncomp <= max_component_size_ (lambda API).
  void check_component_count_(std::size_t ncomp, const char *what, const char *func) const;
  /// MP-7 (B20): the high-level API needs max_component_size_ >= 2*nao_*nao_ (ncomponents >= 2).
  void check_highlevel_components_(const char *func) const;
  /// MP-6: -1 <= ti < Nt_.
  void check_ti_(const char *func, int ti) const;
  /// MP-8: one object per local k-point, each nao x nao with DLR rank r_, row ti allocated and in its window.
  void check_objects_(const char *func, int ti, std::size_t n, const std::function<herm_matrix_hodlr &(int)> &at) const;
  /// MP-9: the dlr_info of a high-level get has the rank and orbital size of this communicator.
  void check_dlr_(const char *func, const dlr_info &dlr) const;
  /// MP-3: the _nospawn routines call MPI from inside an OpenMP region: MPI_THREAD_FUNNELED is required.
  void require_funneled_(const char *func) const;
  /// ST-9 (MP-5, MP-11): a set reuses the counts and displacements of the preceding get, which must have the same
  /// ti, API level and component counts. record_get_ is called by the get (thread 0 in _nospawn), require_get_ by
  /// the set (every thread, before any barrier), clear_get_ after the set (thread 0 in _nospawn).
  void record_get_(int ti, int nLG, int ntv, bool highlevel);
  void require_get_(const char *func, int ti, int nLR, int ntv, bool highlevel) const;
  void clear_get_() { last_get_ti_ = INT_MIN; }
  /// MP-18 (tier 1, Debug/RelWithAsserts only): every rank calls with the same ti and counts (one MPI_Allreduce).
  void debug_collective_(const char *func, int ti, int n1, int n2) const;
  /// MP-13 (Q-9: warn): the imaginary part of Sigma^M dropped by the ti = -1 set (largest |Im| / largest |.|).
  void warn_imag_mat_(const char *func, double rel);

  int last_get_ti_ = INT_MIN;      ///< ST-9: ti of the last get (INT_MIN: none since the last set)
  int last_get_nLG_ = 0;           ///< ST-9: component counts of the last get
  int last_get_ntv_ = 0;
  bool last_get_highlevel_ = false;
  int mpi_thread_level_ = MPI_THREAD_SINGLE;  ///< MP-3: provided thread level (queried by the constructor)
  bool imag_warned_ = false;       ///< MP-13: the warning is printed once per object
  std::vector<double> imag_rel_k_; ///< MP-13: per local k-point relative |Im Sigma^M| of the last ti = -1 set
  /// Sets my_first_t/my_Nt to this rank's t' block for timestep ti >= 0 (H30).
  void set_my_t_block_(int ti, int mpi_rank, int mpi_size);
};

} // namespace
