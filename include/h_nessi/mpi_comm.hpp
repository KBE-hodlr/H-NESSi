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


  // Easy access to arrays.  Only give access to *_allk_buff, as that is what the user needs to modify
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
  int my_Ntau;
  int my_first_tau;
  int my_Nt;
  int my_first_t;
  int my_Nk;
  std::vector<int> my_global_k_list;

  /// Parameters for buffer and communication sizing
  int Nk_;
  int Nt_;
  /// Number of representative tau points in the DLR representation
  int r_;
  /// Maximum number of components of herm_matrix_hodlr objects to be communicated, can also be orbital components
  int max_component_size_;
  /// Number of orbitals, max_component_size_ = 2*nao*nao
  int nao_;
};

} // namespace
