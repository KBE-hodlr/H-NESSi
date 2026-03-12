#pragma once

/**
 * @file utilities.hpp
 * @brief Utility functions and structures for index calculations and memory information.
 */
#include <array>
#include <string>
#include <fstream>
#include <unordered_map>


/**
 * @brief Computes the index range for a thread given its ID and total number of threads.
 * @param thread_id ID of the thread
 * @param thread_num Total number of threads
 * @param size Total size to split
 * @param t0i Starting index offset
 * @return Array with [start, end, count] for the thread
 */
std::array<int, 3> get_my_index(int thread_id, int thread_num, int size, int t0i);

/**
 * @brief Computes the tau index range for a thread given its ID and total number of tasks.
 * @param tid Thread/task ID
 * @param ntasks Total number of tasks
 * @param r Number of tau points
 * @return Array with [start, end, count] for the thread
 */
std::array<int, 3> get_my_tauindex(int tid, int ntasks, int r);

/**
 * @brief Flattens 2D indices (kxi, kyi) into a single index for a grid of size Nk x Nk.
 * @param kxi X index
 * @param kyi Y index
 * @param Nk Grid size
 * @return Flattened index
 */
int iflatten2(int kxi, int kyi, int Nk);

/**
 * @brief Flattens 3D indices (kxi, kyi, taui) into a single index for a grid of size Nk x Nk x Ntau.
 * @param kxi X index
 * @param kyi Y index
 * @param taui Tau index
 * @param Nk Grid size
 * @param Ntau Number of tau points
 * @return Flattened index
 */
int iflatten3(int kxi, int kyi, int taui, int Nk, int Ntau);

/**
 * @brief Returns 0 if x is 0, otherwise returns Nk - x.
 * @param x Input index
 * @param Nk Modulo value
 * @return 0 if x == 0, else Nk - x
 */
int minus_xi(int x, int Nk);

/**
 * @brief Returns Nk - 1 - x.
 * @param x Input index
 * @param Nk Modulo value
 * @return Nk - 1 - x
 */
int inverse_xi(int x, int Nk);


/**
 * @struct MemInfo
 * @brief Holds memory information (total, free, available).
 */
struct MemInfo {
  double total = 0;      ///< Total memory
  double free = 0;       ///< Free memory
  double available = 0;  ///< Available memory

  /**
   * @brief Returns the used memory (total - free).
   * @return Used memory
   */
  double used() const {
    return total - free;
  }
};


/**
 * @brief Retrieves current memory information from the system.
 * @return MemInfo struct with total, free, and available memory
 */
MemInfo getMemoryInfo(void);
