#include <filesystem>

#include "hodlr/mpi_comm.hpp"
#include "hodlr/mpi_comm_utils.hpp"
#include "lattice.hpp"
#include "kpoint.hpp"
#include "selfene.hpp"
#include "observables.hpp"

#include "hodlr/read_inputfile.hpp"

int main(int argc, char *argv[]) {

  std::string output_dir;
  bool write_GS, write_tops;
  bool checkpoint_exists;
  double time_limit, mem_limit;

  int SolverOrder = 5;

  int Nt, Ntau, L;
  int nlvl, xi, r;

  double mu, beta, dt, J, U, Amax, Ainitx;

  double svdtol; // HODLR
  double dlrlambda, epsdlr; //dlr parameters
  int MatsMaxIter, BootMaxIter, StepMaxIter, tstpPrint;
  double MatsMaxErr, BootMaxErr, StepMaxErr;

  int tstp;
  int size = 2;
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

  if (argc<4) {
    // Tell the user how to run the program
    std::cerr << " Please provide a prefix for the input and output files. Exiting ..." << "\n";
    /* "Usage messages" are a conventional way of telling the user
      * how to run a program if they enter the command incorrectly.
      */
    return 1;
  }

  char* flin;
  flin=argv[1];
  char* flout;
  flout=argv[2];
  output_dir=flout;

  char* flcheck;
  flcheck=argv[3];
  std::string checkpoint_dir = flcheck;

  // Check if output_dir exists
  if(!std::filesystem::exists(output_dir)) {
    std::cerr << "Error: output directory '" << output_dir << "' does not exist!\n";
    return 1;
  }
  
  // Check if checkpoint_dir exists
  if(!std::filesystem::exists(checkpoint_dir)) {
    std::cerr << "Error: checkpoint directory '" << checkpoint_dir << "' does not exist!\n";
    return 1;
  }

  // Ensure output_dir ends with '/'
  if(!output_dir.empty() && output_dir.back() != '/') {
    output_dir += '/';
  }

  if(mpi_rank==mpi_root) std::cout << "input file: " << flin << "\n";
  if(mpi_rank==mpi_root) std::cout << "output dir: " << output_dir << "\n";
  if(mpi_rank==mpi_root) std::cout << "checkpoint dir: " << checkpoint_dir << "\n";

  find_param(flin,"__checkpoint=",checkpoint_exists);
  if(mpi_rank==mpi_root) std::cout << "checkpoint_exists = " << checkpoint_exists << "\n";
  find_param(flin,"__time_limit=",time_limit);
  if(mpi_rank==mpi_root) std::cout << "time_limit = " << time_limit << "\n";
  find_param(flin,"__memory_limit=",mem_limit);
  if(mpi_rank==mpi_root) std::cout << "mem_limit = " << mem_limit << "\n";
  find_param(flin,"__write_GS=",write_GS);
  if(mpi_rank==mpi_root) std::cout << "write_GS = " << write_GS << "\n";
  find_param(flin,"__write_tops=",write_tops);
  if(mpi_rank==mpi_root) std::cout << "write_tops = " << write_tops << "\n";

  // // system parameters
  find_param(flin,"__L=",L);
  if(mpi_rank==mpi_root) std::cout << "L = " << L << "\n";
  find_param(flin,"__HoppingJ=",J);
  if(mpi_rank==mpi_root) std::cout << "J = " << J << "\n";
  find_param(flin,"__HubbardU=",U);
  if(mpi_rank==mpi_root) std::cout << "U = " << U << "\n";
  find_param(flin,"__MuChem=",mu);
  if(mpi_rank==mpi_root) std::cout << "mu = " << mu << "\n";
  find_param(flin,"__beta=",beta);
  if(mpi_rank==mpi_root) std::cout << "beta = " << beta << "\n";
  find_param(flin,"__Amax=",Amax);
  if(mpi_rank==mpi_root) std::cout << "Amax = " << Amax << "\n";
  find_param(flin,"__Ainitx=",Ainitx);
  if(mpi_rank==mpi_root) std::cout << "Ainitx = " << Ainitx << "\n";
  
  // solver parameters
  find_param(flin,"__Nt=",Nt);
  if(mpi_rank==mpi_root) std::cout << "Nt = " << Nt << "\n";
  find_param(flin,"__Ntau=",Ntau);
  if(mpi_rank==mpi_root) std::cout << "Ntau = " << Ntau << "\n";
  find_param(flin,"__Ntau=",r);
  if(mpi_rank==mpi_root) std::cout << "r = " << r << "\n";
  find_param(flin,"__dt=",dt);
  if(mpi_rank==mpi_root) std::cout << "dt = " << dt << "\n";
  find_param(flin,"__MatsMaxIter=",MatsMaxIter);
  if(mpi_rank==mpi_root) std::cout << "MatsMaxIter = " << MatsMaxIter << "\n";
  find_param(flin,"__BootMaxIter=",BootMaxIter);
  if(mpi_rank==mpi_root) std::cout << "BootMaxIter = " << BootMaxIter << "\n";
  find_param(flin,"__StepMaxIter=",StepMaxIter);
  if(mpi_rank==mpi_root) std::cout << "StepMaxIter = " << StepMaxIter << "\n"; 
  find_param(flin,"__MatsMaxErr=",MatsMaxErr);
  if(mpi_rank==mpi_root) std::cout << "MatsMaxErr = " << MatsMaxErr << "\n";
  find_param(flin,"__BootMaxErr=",BootMaxErr);
  if(mpi_rank==mpi_root) std::cout << "BootMaxErr = " << BootMaxErr << "\n";
  find_param(flin,"__StepMaxErr=",StepMaxErr);
  if(mpi_rank==mpi_root) std::cout << "StepMaxErr = " << StepMaxErr << "\n";
    
  find_param(flin,"__nlevel=",nlvl);
  if(mpi_rank==mpi_root) std::cout << "nlvl = " << nlvl << "\n";
  find_param(flin,"__tstpPrint=",tstpPrint);
  if(mpi_rank==mpi_root) std::cout << "tstpPrint = " << tstpPrint << "\n";
  find_param(flin,"__svdtol=",svdtol);
  if(mpi_rank==mpi_root) std::cout << "svdtol = " << svdtol << "\n";
  find_param(flin,"__dlrlambda=",dlrlambda);
  if(mpi_rank==mpi_root) std::cout << "dlrlambda = " << dlrlambda << "\n";
  find_param(flin,"__epsdlr=",epsdlr);
  if(mpi_rank==mpi_root) std::cout << "epsdlr = " << epsdlr << "\n";

  // ==========================================================================================
  //         INIT DYSON SOLVERS, MPI COMMUNICATOR, GREEN'S FUNCTIONS, AND SELF-ENERGIES
  // ==========================================================================================

  // dyson solvers
  int rho_version = 1;
  xi = -1;
  bool profile = true;
  hodlr::dlr_info dlr(r, dlrlambda, epsdlr, beta, size, xi);
  hodlr::dyson dyson_sol(Nt, size, SolverOrder, dlr, rho_version); 
  Integration::Integrator I(SolverOrder);
  
  if(mpi_rank==mpi_root){
    std::cout << "Number of representative tau points r = " << r << "\n";
    dyson_sol.print_memory_usage();
  }

  std::vector<hodlr::dlr_info> dlr_vec;
  dlr_vec.reserve(nthreads);
  for(int i = 0; i < nthreads; i++) dlr_vec.emplace_back(r, dlrlambda, epsdlr, beta, size, xi);

  std::vector<hodlr::dyson> dyson_sol_vec;
  dyson_sol_vec.reserve(nthreads);
  for(int i=0; i<nthreads; i++){
    dyson_sol_vec.emplace_back(Nt, size, SolverOrder, dlr_vec[i], rho_version, profile);
  }

  //lattice
  lattice_2d_ysymm lattice(L, Nt, dt, J, Amax, mu, size);
  int Nk = lattice.Nk_;
  if(mpi_rank==mpi_root) std::cout << "Total number of k-points Nk = " << Nk << "\n";

  //mpi communicatior
  mpi_comm comm(Nk, Nt, r, size);
  int Nk_rank = comm.Nk_per_rank[mpi_rank];

  //kpoint contains Gs and Sigmas
  std::vector<hodlr::herm_matrix_hodlr> G_vec;
  std::vector<hodlr::herm_matrix_hodlr> S_vec;
  G_vec.reserve(Nk_rank);
  S_vec.reserve(Nk_rank);

    for(int i=0;i<Nk_rank;i++){
      G_vec.emplace_back(Nt,r,nlvl,svdtol,size,size,-1,SolverOrder);
      S_vec.emplace_back(Nt,r,nlvl,svdtol,size,size,-1,SolverOrder);
    }

    if(mpi_rank==mpi_root){
      std::cout << "---- MATSUBARA PHASE " << "\n";
    }

  // ==========================================================================================
  //         Fill G^M with some random data
  // ==========================================================================================

    double err = 0;
    for(int k=0; k<Nk_rank; k++){
      int global_k = comm.kindex_rank[k];
      for(int t = 0; t < r; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            G_vec[k].map_mat(t)(i,j) = global_k * r*size*size + t*size*size + i*size + j;
          }
        }
      }
    }
  // ==========================================================================================
  //         communicate G^M
  // ==========================================================================================
#pragma omp parallel 
{
    comm.mpi_get_and_comm_nospawn(-1, G_vec, dlr);
}
  // ==========================================================================================
  //    Sigma^M = 2*G^M.  each mpi does all k, and assigned tau
  // ==========================================================================================
    for(int k = 0; k < Nk; k++) {
      for(int t = comm.my_first_tau; t < comm.my_first_tau + comm.my_Ntau; t++) {
        comm.map_mat(k,t) = 2 * comm.map_mat(k,t);
      }
    }
  // ==========================================================================================
  //         communicate S^M
  // ==========================================================================================
#pragma omp parallel 
{
    comm.mpi_comm_and_set_nospawn(-1, S_vec);
}
  // ==========================================================================================
  //         check that we communicated correctly
  // ==========================================================================================
    for(int k=0; k < Nk_rank; k++){
      for(int t = 0; t < r; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            err += abs(2*G_vec[k].map_mat(t)(i,j) - S_vec[k].map_mat(t)(i,j));
          }
        }
      }
    }
    std::cout << err << std::endl;

  // ==========================================================================================
  //         advance G and S to tstp = 30 to rest real-time communication
  // ==========================================================================================
    tstp = 30;
    for(int t = 0; t < tstp-SolverOrder; t++) {
      for(int k=0; k<Nk_rank; k++){
        G_vec[k].update_blocks(I);
        S_vec[k].update_blocks(I);
      } 
    }

  // ==========================================================================================
  //         fill G^<, G^R, G^\rceil
  // ==========================================================================================
    for(int k=0; k<Nk_rank; k++){
      int global_k = comm.kindex_rank[k];
      for(int t = 0; t <= tstp; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            G_vec[k].map_ret_curr(tstp,t)(i,j) = global_k * (tstp+1)*size*size + t*size*size + i*size + j;
            G_vec[k].map_les_curr(t,tstp)(i,j) = cplx(0,1) * (double)(global_k * (tstp+1)*size*size) + (double)(t*size*size + i*size + j);
          }
        }
      }
      for(int t = 0; t < r; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            G_vec[k].map_tv(tstp,t)(i,j) = (double)(global_k * (tstp+1)*size*size + t*size*size) + cplx(0,1)*(double)(i*size) + (double)(j);
          }
        }
      }
    }
  // ==========================================================================================
  //         communicate three Keldysh components
  // ==========================================================================================
#pragma omp parallel 
{
    comm.mpi_get_and_comm_nospawn(tstp, G_vec, dlr);
}
  // ==========================================================================================
  //         S = 2*G.  Do all k, and assigned t/tau
  // ==========================================================================================
    for(int k = 0; k < Nk; k++) {
      for(int t = comm.my_first_tau; t < comm.my_first_tau + comm.my_Ntau; t++) {
        comm.map_tv(k,t) = 2 * comm.map_tv(k,t);
      }
    }
    for(int k = 0; k < Nk; k++) {
      for(int t = comm.my_first_t; t < comm.my_first_t + comm.my_Nt; t++) {
        comm.map_ret(k,t) = 2 * comm.map_ret(k,t);
        comm.map_les(k,t) = 2 * comm.map_les(k,t);
      }
    }
  // ==========================================================================================
  //         communicate three Keldysh components
  // ==========================================================================================
#pragma omp parallel 
{
    comm.mpi_comm_and_set_nospawn(tstp, S_vec);
}

  // ==========================================================================================
  //         calculate error.   Make sure we communicated correctly
  // ==========================================================================================
    for(int k=0; k < Nk_rank; k++){
      for(int t = 0; t < r; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            err += abs(2.*G_vec[k].map_tv(tstp,t)(i,j) - S_vec[k].map_tv(tstp,t)(i,j));
          }
        }
      }
    }
    for(int k=0; k < Nk_rank; k++){
      for(int t = 0; t <= tstp; t++) {
        for(int i = 0; i < size; i++) {
          for(int j = 0; j < size; j++) {
            err += abs(2.*G_vec[k].map_ret_curr(tstp,t)(i,j) - S_vec[k].map_ret_curr(tstp,t)(i,j));
            err += abs(2.*G_vec[k].map_les_curr(t,tstp)(i,j) - S_vec[k].map_les_curr(t,tstp)(i,j));
          }
        }
      }
    }
    std::cout << err << std::endl;








  MPI_Finalize();
  return 0;
}

