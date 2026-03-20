#include <filesystem>

#include "h_nessi/mpi_comm.hpp"
#include "h_nessi/mpi_comm_utils.hpp"
#include "h_nessi/read_inputfile.hpp"
#include "lattice.hpp"
#include "kpoint.hpp"
#include "Hubb_2B.hpp"
#include "observables.hpp"


using namespace hodlr;

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

  // ============================================================================
  //    INIT LATTICE, SELF-ENERGY SOLVER, AND OBSERVABLES
  // ============================================================================

  //lattice
  lattice_2d_ysymm lattice(L, Nt, dt, J, Amax, mu, size);
  int Nk = lattice.Nk_;
  if(mpi_rank==mpi_root) std::cout << "Total number of k-points Nk = " << Nk << "\n";

  // self-energy evaluator
  Hubb_2B se_eval(U,L,nthreads);

  // observables
  std::vector<double> density_k(Nk), jxt_local(Nt), jyt_local(Nt), Ekint_local(Nt), Navgt_local(Nt);
  std::vector<double> jxt_total(Nt), jyt_total(Nt), Ekint_total(Nt), Navgt_total(Nt);
  std::vector<std::vector<double>> jxt_iter_total(BootMaxIter+1,std::vector<double>(Nt));
  std::vector<std::vector<double>> jxt_iter_local(BootMaxIter+1,std::vector<double>(Nt));

  //convergance error
  std::vector<double> errk(Nk);

  // ==========================================================================================
  //         INIT DYSON SOLVERS, MPI COMMUNICATOR, GREEN'S FUNCTIONS, AND SELF-ENERGIES
  // ==========================================================================================

  // dyson solvers
  int rho_version = 1;
  xi = -1;
  bool profile = true;
  dlr_info dlr(r, dlrlambda, epsdlr, beta, size, xi);
  dyson dyson_sol(Nt, size, SolverOrder, dlr, rho_version); 
  Integration::Integrator I(SolverOrder);
  
  if(mpi_rank==mpi_root){
    std::cout << "Number of representative tau points r = " << r << "\n";
    dyson_sol.print_memory_usage();
  }

  std::vector<dlr_info> dlr_vec;
  dlr_vec.reserve(nthreads);
  for(int i = 0; i < nthreads; i++) dlr_vec.emplace_back(r, dlrlambda, epsdlr, beta, size, xi);

  std::vector<dyson> dyson_sol_vec;
  dyson_sol_vec.reserve(nthreads);
  for(int i=0; i<nthreads; i++){
    dyson_sol_vec.emplace_back(Nt, size, SolverOrder, dlr_vec[i], rho_version, profile);
  }

  //mpi communicatior
  mpi_comm comm(Nk, Nt, r, size);
  int Nk_rank = comm.Nk_per_rank[mpi_rank];

  //kpoint contains Gs and Sigmas
  std::vector<std::unique_ptr<kpoint>> corrK_rank;
  corrK_rank.resize(Nk_rank);

  //create kpoints either from scratch or from checkpoint
  if(!checkpoint_exists){
    #pragma omp parallel for
    for(int i=0;i<Nk_rank;i++){
      int global_k = comm.kindex_rank[i];
      corrK_rank[i] = std::make_unique<kpoint>(Nt,r,nlvl,svdtol,size,beta,dt,SolverOrder,lattice.kpoints_[global_k],lattice,mu,Ainitx);
    }
  }
  else{
    for(int k=0;k<Nk;k++){
      if(comm.k_rank_map[k]==mpi_rank){
        h5e::File checkpoint_file(checkpoint_dir+"GSigma" + std::to_string(k)+".h5", h5e::File::ReadOnly);
        corrK_rank.push_back(std::make_unique<kpoint>(Nt,r,nlvl,svdtol,size,beta,dt,SolverOrder,lattice.kpoints_[k],lattice,mu,Ainitx,checkpoint_file));        
      }
    }
  }

  //create reference vector of G and Sigma - for passing between communicator
  std::vector<std::reference_wrapper<herm_matrix_hodlr>> Grefs;
  std::vector<std::reference_wrapper<herm_matrix_hodlr>> Srefs;
  for(int i=0;i<Nk_rank;i++){
    Grefs.push_back(corrK_rank[i]->G_);
    Srefs.push_back(corrK_rank[i]->Sigma_);
  }

  int t0;

  if(!checkpoint_exists){
    //============================================================================
    //                                MATSUBARA
    //============================================================================

    if(mpi_rank==mpi_root){
      std::cout << "---- MATSUBARA PHASE " << "\n";
    }

    start = MPI_Wtime();

    for(int iter=0;iter<=MatsMaxIter;iter++){
      double err=0.0;
      double tot_err = 0.0;

      // Evaluate self-energy
      if(iter!=0){
        se_eval.Sigma_spawn(-1, comm, Grefs, Srefs, dlr);
      }

      // Solve Dyson
      for(int k=0; k<Nk_rank; k++){
        errk[k] = corrK_rank[k]->step_dyson(-1, SolverOrder, lattice, I, dyson_sol, dlr);
      }

      //Error estimation
      err = std::reduce(errk.begin(),errk.end());
      MPI_Allreduce(&err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
      if(mpi_rank==mpi_root) std::cout << "------ Matsubara Iteration: "  << iter << " tot err: " << tot_err << "\n";
      if((tot_err <= MatsMaxErr) && (iter!=0)) break;
    }

    end = MPI_Wtime();

    if(mpi_rank==mpi_root){
      double elapsed_seconds = end-start;
      std::cout << "------ Time [equilibrium calculation] = " << elapsed_seconds << "\n";
    }

    // Setting the convolution tensor
    for(int k=0; k<Nk_rank;k++){
      corrK_rank[k]->G_.initGMConvTensor(dlr);
      corrK_rank[k]->Sigma_.initGMConvTensor(dlr);
    }

    //============================================================================
    //           BOOTSTRAPPING PHASE
    //============================================================================

    if(mpi_rank==mpi_root){
      std::cout << "---- BOOTSTRAPPING PHASE \n";
    }

    start = MPI_Wtime();

    for(int iter=0;iter<=BootMaxIter;iter++){
      
      double err=0.0;
      double tot_err = 0.0;

      if(mpi_rank==mpi_root) std::cout << "---- Time step ti = " << tstp << "\n";

      // Evaluate self-energy
      if(iter!=0){
        for(int tstp = 0; tstp <= SolverOrder; tstp ++){
          se_eval.Sigma_spawn(tstp, comm, Grefs, Srefs, dlr);
        }
      }

      // Solve Dyson
      for(int k=0;k<Nk_rank;k++){
        errk[k] = corrK_rank[k]->step_dyson(SolverOrder, SolverOrder, lattice, I, dyson_sol, dlr);
      }
        
      // Error estimation
      err = std::reduce(errk.begin(),errk.end());

      // observables calculation
      for(int tstp = 0; tstp <= SolverOrder; tstp++){
        std::array<double, 4> obs = observables::get_obs_local(tstp, Nk_rank, lattice, comm.kindex_rank, corrK_rank);
        jxt_local[tstp]=obs[0]/Amax;
        jyt_local[tstp]=obs[1]/Amax;
        Ekint_local[tstp]=obs[2];
        Navgt_local[tstp]=obs[3];

        jxt_iter_local[iter][tstp]=obs[0]/Amax;
      }

      MPI_Allreduce(&err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  

      if(mpi_rank==mpi_root) std::cout << "---- Bootstrap Iteration: "  << iter << " tot err: " << tot_err << "\n";
      if((tot_err <= BootMaxErr) && (iter!=0)) break;

    } // Bootstrap iteration loop

    end = MPI_Wtime();

    if(mpi_rank==mpi_root){
      double elapsed_seconds= end -start;
      std::cout << "------ Time [bootstrapping] = " << elapsed_seconds << "\n";
    }

  } //checkpoint if

  // ============================================================================
  //             TIME PROPAGATION
  // ============================================================================

  if(mpi_rank==mpi_root){std::cout << "----- TIME PROPAGATION PHASE \n";}

  //timing vectors
  std::vector<double> timing_upBlocks(Nt);
  std::vector<double> timing_Extrapolate(Nt);
  std::vector<double> timing_SE(Nt);
  std::vector<double> timing_dyson(Nt);

  start = MPI_Wtime();

  // initial time step
  if(checkpoint_exists){
    t0=corrK_rank[0]->G_.tstpmk() + corrK_rank[0]->G_.k() + 1;
  } else{
    t0=SolverOrder+1;
  }

  // conditions to stop time stepping
  int mem_stop_mess = 0; // memory limit reached
  int time_stop_mess = 0; // local time limit reached 
  int tot_mem_stop_mess = 0; // global memory limit reached
  int err_mess_tstp = 0; // convergance reached at time step

#pragma omp parallel shared(comm, se_eval, time_stop_mess, tot_mem_stop_mess, mem_stop_mess, err_mess_tstp)
  {
    int thread_id = omp_get_thread_num();

    std::array<int,3> th_Nks_mpi_indxs = get_my_index(thread_id,nthreads,Nk_rank,0);
    int th_my_init_k = th_Nks_mpi_indxs[0];
    int th_my_end_k = th_Nks_mpi_indxs[1];
    int th_my_Nk = th_Nks_mpi_indxs[2];

    for(int tstp=t0; tstp<Nt; tstp++){

      //check stopping conditions
      if(time_stop_mess == 1) continue;
      if(tot_mem_stop_mess >= 1) continue;

      //reset error message for time step
      err_mess_tstp = 0;
      if(mpi_rank==mpi_root && thread_id==0) std::cout << "---- Time step ti = " << tstp << std::endl;

      //update blocks of G and Sigma
      double start_upBlocks = MPI_Wtime();
      if(th_my_Nk != 0){
        for(int k=th_my_init_k;k<=th_my_end_k;k++){
          corrK_rank[k]->G_.update_blocks(I);
          corrK_rank[k]->Sigma_.update_blocks(I);
        }
      }
      double end_upBlocks = MPI_Wtime();
      if(thread_id==0) timing_upBlocks[tstp] = end_upBlocks-start_upBlocks;

      #pragma omp barrier

      //Extrapolation
      double start_Extrapolate = MPI_Wtime();
      if(th_my_Nk != 0){
        for(int k=th_my_init_k;k<=th_my_end_k;k++){
          dyson_sol_vec[thread_id].extrapolate(corrK_rank[k]->G_, I);
        }
      }
      double end_Extrapolate = MPI_Wtime();
      if(thread_id==0) timing_Extrapolate[tstp] = end_Extrapolate-start_Extrapolate;

      #pragma omp barrier

      for(int iter=0; iter<=StepMaxIter; iter++){

        //check convergance from previous iteration
        if(err_mess_tstp == 1) continue;

        //reset error for iteration
        double err=0.0;
        double tot_err = 0.0;

        double start_Sigma_tstp, end_Sigma_tstp;

        // Evaluate Self-Energy
        start_Sigma_tstp = MPI_Wtime();

        // Evaluate new version of self-energy
        se_eval.Sigma_nospawn(tstp, comm, Grefs, Srefs, dlr);

        if (thread_id == 0) {
            end_Sigma_tstp = MPI_Wtime();
            timing_SE[tstp] = end_Sigma_tstp-start_Sigma_tstp;
        }

        if(thread_id==0 && mpi_rank==mpi_root) std::cout << "Sigma took " << end_Sigma_tstp - start_Sigma_tstp << " seconds" << std::endl;

        #pragma omp barrier

        // Solve Dyson
        double start_dyson_tstp, end_dyson_tstp;
        if(th_my_Nk != 0){
          start_dyson_tstp = MPI_Wtime();
          for(int k=th_my_init_k;k<=th_my_end_k;k++){
            errk[k] = corrK_rank[k]->step_dyson(tstp, SolverOrder, lattice, I, dyson_sol_vec[thread_id], dlr_vec[thread_id]);
          }
          end_dyson_tstp = MPI_Wtime();
          timing_dyson[tstp] = end_dyson_tstp-start_dyson_tstp;
        }

        if(thread_id==0){
          if(mpi_rank==mpi_root) std::cout << "Dyson took " << end_dyson_tstp - start_dyson_tstp << " seconds" << std::endl;
        }

        #pragma omp barrier

        // Error estimation and observables calculation
        if(thread_id==0){
          err = std::reduce(errk.begin(),errk.end());
          MPI_Allreduce(&err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

          std::array<double,4> obs = observables::get_obs_local(tstp,Nk_rank,lattice,comm.kindex_rank,corrK_rank);
          jxt_local[tstp]=obs[0]/Amax;
          jyt_local[tstp]=obs[1]/Amax;
          Ekint_local[tstp]=obs[2];
          Navgt_local[tstp]=obs[3];
          jxt_iter_local[iter][tstp]=obs[0]/Amax;

          if(mpi_rank==mpi_root) std::cout << "------ Time Stepping Iteration: "  << iter << ", tot err: " << tot_err << std::endl;

          if((tot_err <= StepMaxErr) && (iter!=0)){
            err_mess_tstp = 1;
          }
        }
        #pragma omp barrier
      }

      //check for time and memory limits and perform checkpoint if needed
      if(thread_id==0){

        double time_now = MPI_Wtime();
        double stop_time = time_now - start_tot;
        if(stop_time>time_limit){
          time_stop_mess = 1;
        }
        MPI_Bcast(&time_stop_mess,1,MPI_INT,mpi_root,MPI_COMM_WORLD);

        MemInfo mem = getMemoryInfo();
        if(mem.available<mem_limit){
          mem_stop_mess = 1;
        }

        MPI_Allreduce(&mem_stop_mess, &tot_mem_stop_mess, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if(time_stop_mess==1 || tot_mem_stop_mess>=1){

          // 1. SAFELY CREATE DIRECTORIES (tops, output, checkpoint)
          if(mpi_rank == mpi_root) {
              if(!std::filesystem::exists(output_dir + "tops")){
                std::filesystem::create_directories(output_dir + "tops");
              }
              if(!std::filesystem::exists(checkpoint_dir)){
                std::filesystem::create_directories(checkpoint_dir);
              }
              if(!std::filesystem::exists(output_dir)){
                std::filesystem::create_directories(output_dir);
              }
          }
          // Ensure directories are created before other ranks proceed
          MPI_Barrier(MPI_COMM_WORLD); 

          // 2. WRITE TOP OUTPUT
          std::ostringstream filename;
          filename << output_dir << "tops/checkpoint_top_rank" << mpi_rank << ".txt";
          std::ofstream file(filename.str());
          FILE* topOutput = popen("top -b -n 1 -o %MEM", "r");
          char buffer[256];
          while (fgets(buffer, sizeof(buffer), topOutput)) {
            file << buffer;
          }
          pclose(topOutput);
          file.close();

          if(mpi_rank==mpi_root){
            if(time_stop_mess) std::cout << "REACHED THE TIME LIMIT\n";
          }

          if(mem_stop_mess && (mem.available<mem_limit)) {
              std::cout << "REACHED THE MEMORY LIMIT AT MPI RANK " << mpi_rank 
                        <<  " WITH AVAILABLE MEMORY = " << mem.available << "\n";
          }

          // 3. WRITE G AND SIGMA CHECKPOINTS
          for(int k = 0; k < Nk_rank; k++) {
              h5e::File checkpoint_file(checkpoint_dir + "GSigma" + std::to_string(comm.kindex_rank[k]) + ".h5", 
                                        h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
              corrK_rank[k]->G_.write_checkpoint_hdf5(checkpoint_file, "G/");
              corrK_rank[k]->Sigma_.write_checkpoint_hdf5(checkpoint_file, "S/");
          }

          // 4. REDUCE OBSERVABLES
          MPI_Reduce(jxt_local.data(), jxt_total.data(), jxt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
          MPI_Reduce(jyt_local.data(), jyt_total.data(), jyt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
          MPI_Reduce(Ekint_local.data(), Ekint_total.data(), Ekint_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
          MPI_Reduce(Navgt_local.data(), Navgt_total.data(), Navgt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);

          for(int iter=0; iter<=BootMaxIter; iter++){
            MPI_Reduce(jxt_iter_local[iter].data(), jxt_iter_total[iter].data(), jxt_iter_total[iter].size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
          }

          // 5. ROOT PROCESS HANDLES THE MASTER OBSERVABLES FILE
          if(mpi_rank==mpi_root){
            if(checkpoint_exists){
              // Read and update observables from previous checkpoint
              // The brackets ensure jcheckpoint_file closes before out_file opens
              {
                  h5e::File jcheckpoint_file(output_dir + "obs_checkpoint.h5", h5e::File::ReadOnly);
                  std::vector<double> jxtcheckpoint = jcheckpoint_file.getDataSet("jxtnorm").read<std::vector<double>>();
                  std::vector<double> jytcheckpoint = jcheckpoint_file.getDataSet("jytnorm").read<std::vector<double>>();
                  std::vector<double> Ekintcheckpoint = jcheckpoint_file.getDataSet("Ekint").read<std::vector<double>>();
                  std::vector<double> Navgtcheckpoint = jcheckpoint_file.getDataSet("Navgt").read<std::vector<double>>();

                  for(int ti=0; ti<t0; ti++){
                    jxt_total[ti] = jxtcheckpoint[ti];
                    jyt_total[ti] = jytcheckpoint[ti];
                    Ekint_total[ti] = Ekintcheckpoint[ti];
                    Navgt_total[ti] = Navgtcheckpoint[ti];
                  }
              }
            }

            // Write new checkpoint observables
            h5e::File out_file(output_dir + "obs_checkpoint.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
            h5e::dump<std::vector<double>>(out_file, "jxtnorm", jxt_total);
            h5e::dump<std::vector<double>>(out_file, "jytnorm", jyt_total);
            h5e::dump<std::vector<double>>(out_file, "Ekint", Ekint_total);
            h5e::dump<std::vector<double>>(out_file, "Navgt", Navgt_total);
            h5e::dump<std::vector<std::vector<double>>>(out_file, "jxt_iter", jxt_iter_total);
          }
        }// checkpoint
      }// thread_id==0
      #pragma omp barrier
    } // time propagation loop  
  }//omp
  // gather observables from all ranks
  MPI_Reduce(jxt_local.data(), jxt_total.data(), jxt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(jyt_local.data(), jyt_total.data(), jyt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(Ekint_local.data(), Ekint_total.data(), Ekint_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(Navgt_local.data(), Navgt_total.data(), Navgt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);

  for(int iter=0; iter<=BootMaxIter; iter++){
    MPI_Reduce(jxt_iter_local[iter].data(), jxt_iter_total[iter].data(), jxt_iter_total[iter].size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  }

  for(int t = 0; t <= SolverOrder; t++) {
    for(int k = 0; k < Nk_rank; k++) {
      corrK_rank[k]->G_.update_blocks(I);
      corrK_rank[k]->Sigma_.update_blocks(I);
    }
  }

  double local_mem_usage = 0.0;
  double total_mem_usage = 0.0;
  for(int k = 0; k < Nk_rank; k++) {
    local_mem_usage += corrK_rank[k]->G_.get_memory_usage(false) + corrK_rank[k]->Sigma_.get_memory_usage(false);
  }

  MPI_Reduce(&local_mem_usage, &total_mem_usage, 1, MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);

  if(mpi_rank==mpi_root){
    std::cout << "Total memory usage G and Sigma across all ranks: " << total_mem_usage  << " GB\n";
  }
  end = MPI_Wtime();

  if(mpi_rank==mpi_root){
    double elapsed_seconds= end -start;
    std::cout << "Time [propagation] = " << elapsed_seconds << "\n";
  }

  if(mpi_rank==mpi_root){

    std::cout << "Writing output files...\n";
    // update observables from previous checkpoint
    if(checkpoint_exists){
      h5e::File jcheckpoint_file(output_dir + "obs_checkpoint.h5", h5e::File::ReadOnly);
      std::vector<double> jxtcheckpoint = jcheckpoint_file.getDataSet("jxtnorm").read<std::vector<double>>();
      std::vector<double> jytcheckpoint = jcheckpoint_file.getDataSet("jytnorm").read<std::vector<double>>();
      std::vector<double> Ekintcheckpoint = jcheckpoint_file.getDataSet("Ekint").read<std::vector<double>>();
      std::vector<double> Navgcheckpoint = jcheckpoint_file.getDataSet("Navgt").read<std::vector<double>>();

      for(int ti=0; ti<t0; ti++){
        jxt_total[ti] = jxtcheckpoint[ti];
        jyt_total[ti] = jytcheckpoint[ti];
        Ekint_total[ti] = Ekintcheckpoint[ti];
        Navgt_total[ti] = Navgcheckpoint[ti];
      }
    }

    //write new observables
    h5e::File new_out_file(output_dir + "obs.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    h5e::dump<std::vector<double>>(new_out_file, "jxtnorm", jxt_total);
    h5e::dump<std::vector<double>>(new_out_file, "jytnorm", jyt_total);
    h5e::dump<std::vector<double>>(new_out_file, "Ekint", Ekint_total);
    h5e::dump<std::vector<double>>(new_out_file, "Navgt", Navgt_total);
    h5e::dump<std::vector<std::vector<double>>>(new_out_file, "jxt_iter", jxt_iter_total);

    //write timings
    h5e::File dyson_file(output_dir + "dyson_timings.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    dyson_sol_vec[0].write_timing(dyson_file);

    h5e::File time_file(output_dir + "timings_file.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    h5e::dump<std::vector<double>>(time_file, "updateBlocks", timing_upBlocks);
    h5e::dump<std::vector<double>>(time_file, "Extrapolate", timing_Extrapolate);
    h5e::dump<std::vector<double>>(time_file, "SelfEnergy", timing_SE);
    h5e::dump<std::vector<double>>(time_file, "dyson", timing_dyson);
  }

  if(write_GS==1){
    if(!std::filesystem::exists(output_dir + "GS")){
      std::filesystem::create_directories(output_dir + "GS");
      std::cout << "Created " << output_dir << "GS/ directory\n";
    }
    for(int k=0;k<Nk_rank;k++){
      {
        h5e::File out_file(output_dir + "GS/GSigma" + std::to_string(comm.kindex_rank[k]) + ".h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
        corrK_rank[k]->G_.write_to_hdf5(out_file, "G/");
        corrK_rank[k]->Sigma_.write_to_hdf5(out_file, "S/");
      }
    }
  }

  // write top output for each rank for memory usage monitoring
  if(write_tops==1){
    if(!std::filesystem::exists(output_dir + "tops")){
      std::filesystem::create_directories(output_dir + "tops");
      std::cout << "Created " << output_dir << "tops/ directory\n";
    }

    std::ostringstream filename;
    filename << output_dir << "tops/top_rank" << mpi_rank << ".txt";

    std::ofstream file(filename.str());
    if (!file) {
        std::cerr << "Error opening file for rank " << mpi_rank << "\n";
        MPI_Finalize();
        return 1;
    }

    FILE* topOutput = popen("top -b -n 1 -o %MEM", "r");
    if (!topOutput) { // Check if popen() failed
        std::cerr << "Error running top command for rank " << mpi_rank << "\n";
        file << "Error: popen() failed\n";
        file.close();
        MPI_Finalize();
        return 1;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), topOutput)) {
        file << buffer;
    }

    pclose(topOutput);
    file.close();
  }


  if(mpi_rank==mpi_root){
    end_tot = MPI_Wtime();
    double elapsed_seconds= end_tot -start_tot;
    std::cout << "Total Time = " << elapsed_seconds << "\n";    
  }


  MPI_Finalize();
  return 0;
}

