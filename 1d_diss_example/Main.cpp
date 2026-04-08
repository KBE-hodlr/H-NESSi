#include <filesystem>

#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/mpi_comm_utils.hpp"
#include "h_nessi/read_inputfile.hpp"
#include "Hubb_diss.hpp"


using namespace h_nessi;

int main(int argc, char *argv[]) {

  std::string output_dir;
  bool write_GS, write_tops;
  bool checkpoint_exists;
  double time_limit, mem_limit;

  int SolverOrder = 5;

  int Nt, L;
  int nlvl, xi, r;

  double mu, beta, dt, J, U, Amax, Ainitx;

  double svdtol; // HODLR
  double dlrlambda, epsdlr; //dlr parameters
  int MatsMaxIter, BootMaxIter, StepMaxIter, tstpPrint;
  double MatsMaxErr, BootMaxErr, StepMaxErr;

  int tstp;
  int size = 1;
  int mpi_rank,mpi_size,mpi_root;

  //============================================================================
  //                             MPI INITIALIZATION
  //============================================================================

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  mpi_root=0;
  if(mpi_rank==mpi_root){
    std::cout << "mpi_size = " << mpi_size << "\n";
  }

  //============================================================================
  //                               SET THREADS
  //=============================================================================

  Eigen::setNbThreads(1);
  char *env = getenv("SLURM_CPUS_PER_TASK");
  int nthreads = env ? atoi(env) : 1;
  nthreads = omp_get_max_threads();
  if(mpi_rank==mpi_root){
    std::cout << "nthreads = " << nthreads << "\n";
  }  
  omp_set_num_threads(nthreads);

  double start, end, start_tot, end_tot;
  if(mpi_rank==mpi_root){
    std::cout << "Hubbard 2D SOPT" << "\n";
    start_tot = MPI_Wtime();
    std::cout << "reading input file ..." << "\n";
  }

  //============================================================================
  //                           READ INPUT
  //============================================================================

  if(argc<4) throw("COMMAND LINE ARGUMENT MISSING");

  }

