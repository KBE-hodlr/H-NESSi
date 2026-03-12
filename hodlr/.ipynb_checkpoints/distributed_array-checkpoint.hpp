#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cassert>
#include <mpi.h>

#include "utils.hpp"

namespace hodlr{

/** \brief <b> Auxiliary data structure for handling 
set of data blocks and includes usual MPI processes on them.</b>
 *
 * <!-- ====== DOCUMENTATION ====== -->
 *
 *  \par Purpose
 * <!-- ========= -->
 *
 * Auxiliary data structure for handling of data blocks (total number is n_) ,
 * which are stored in  * contiguous form in member data_ and
 * includes usual MPI routines. The class identity is marked by * tid_  \f$ \in (0,\ldots,ntasks_-1) \f$
 * and the value of the tid_ is just the MPI rank or 0 if MPI is not defined. 
 * Each data block j is owned by precisely one process, which is given by \f$ tid * _map(j) = tid_\f$.
 * The member maxlen marks the maximum size reserved for the block. 
 * NOTE: even if the block is not owned by the process, the space for 
 * the data is allocated; the ownership plays a role when the data are 
 * manipulated
 */


class distributed_array{
public:
  /* construction, destruction */
  distributed_array();
  ~distributed_array();
  distributed_array(const distributed_array &g);
  distributed_array &operator=(const distributed_array &g);
  distributed_array(int n,int maxlen,bool mpi,int blocksize=0);

  // Important functions
  void clear(void);
  void reset_blocksize(int blocksize);
  void mpi_sum_complex();
  void mpi_bcast_all_complex(void);
  void mpi_bcast_all_double(void);

  // Handles to data
  std::complex<double> * block(int j);
  std::complex<double> * data() { return vec_.data(); }
  ZMatrix& vec() { return vec_; }
  int vecsize() {return vec_.rows() * vec_.cols(); }

  // Handles to gemoetry
  std::vector<int>& block_displacement_array() { return block_displacement_array_; }
  std::vector<int>& displacement_array() { return displacement_array_; }
  std::vector<int>& datalength_array() { return datalength_array_; }

  // Local block information
  int numblock_rank(void);
  int firstblock_rank(void);
  int firstdata_rank(void);

  // constants
  int n_blocks(void) const {return n_blocks_;}
  int blocksize(void) const {return blocksize_;}
  int maxlen(void) const {return maxlen_;}
  int tid(void) const {return tid_;}
  int ntasks(void) const {return ntasks_;}

private:
  int n_blocks_;                    /*!< Number of data blocks */ 
  int blocksize_;            /*!< Current size of one element in the block. For timestep this is (2*tstp+ntau) and includes (ret,les,tv). This number is changed with increasing timesteps. */ 
  int maxlen_;               /*!< Maximum size of one element in the block.  For timestep this is (2*nt+ntau) and includes (ret,les,tv). This number is fixed with increasing timesteps. */  
  ZMatrix vec_;                  /*!< Vector of contiguous data */
  int tid_;                  /*!< mpi rank if MPI is defined, else 0 */
  int ntasks_;               /*!< mpi size if MPI is defined, else 1 */
  std::vector<int> block_displacement_array_;
  std::vector<int> displacement_array_;
  std::vector<int> datalength_array_;
};

} //namespace hodlr
