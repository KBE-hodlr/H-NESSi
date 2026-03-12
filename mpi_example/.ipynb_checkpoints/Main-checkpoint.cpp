#include <cblas.h>

#include "mpi_comm.hpp"
#include "utilities.hpp"
#include "lattice.hpp"
#include "kpoint.hpp"
#include "selfene.hpp"
#include "observables.hpp"

#include "hodlr/read_inputfile.hpp"

bool parse_bool(const std::string& s) {
  std::string str = s;
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);

  if (str == "1" || str == "true" || str == "yes") return true;
  if (str == "0" || str == "false" || str == "no") return false;

  throw std::invalid_argument("Invalid boolean value: " + s);
}

int main(int argc, char *argv[]) {
  
  std::unordered_map<std::string, std::string> args;
  std::unordered_map<std::string, bool> bool_flags;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto pos = arg.find('=');
    if (pos != std::string::npos) {
      std::string key = arg.substr(0, pos);
      std::string value = arg.substr(pos + 1);
      args[key] = value;
    } else {
      // Treat as a boolean flag (no value means true)
      bool_flags[arg] = true;
    }
  }

  // Input file
  if (!args.count("input")) {
    std::cerr << "Error: input file not specified. Use input=filename\n";
    return 1;
  }
  char* flin = const_cast<char*>(args["input"].c_str());

  std::string output_dir = args.count("output_dir") ? args["output_dir"] : "/ceph/hpc/data/s25o02-09-users/"; // default "/ceph/hpc/data/s25o02-09-users"
  if (!output_dir.empty() && output_dir.back() != '/') {
    output_dir += '/';
  }

  // Time limit
  double time_limit = args.count("time_limit") ? std::stod(args["time_limit"]) : 0.0;
  double mem_limit = args.count("memory_limit") ? std::stod(args["memory_limit"]) : 0.0;


  // Checkpoint (bool)
  bool checkpoint_exists = false;
  try {
      if (args.count("checkpoint")) {
          checkpoint_exists = parse_bool(args["checkpoint"]);
      } else if (bool_flags.count("checkpoint")) {
          checkpoint_exists = true;
      }
  } catch (const std::invalid_argument& e) {
      std::cerr << "Error parsing checkpoint: " << e.what() << "\n";
      return 1;
  }

  int SolverOrder = 5;

  int Nt, Ntau, L;
  int nlvl, xi, r;

  double mu, beta, dt, J, U, Eint, Ainitx;

  double svdtol; // HODLR
  double lambda, epsdlr; //dlr convergence parameters
  int matsMaxIter, BootstrapMaxIter, correctorSteps, tstpPrint;
  double matsMaxErr, timeMaxErr;

  int tstp;
  int size = 1;
  int mpi_rank,mpi_size,mpi_root;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  mpi_root=0;
  if(mpi_rank==mpi_root){
    std::cout << "mpi_size =" << mpi_size << "\n";
  }

  double start, end, start_tot, end_tot;
  if(mpi_rank==mpi_root){
    std::cout << "Hubbard 2D SOPT" << "\n";
    start_tot = MPI_Wtime();
    std::cout << " reading input file ..." << "\n";
  }

  //============================================================================
  //                           READ INPUT
  //============================================================================

  if(argc<2) throw("COMMAND LINE ARGUMENT MISSING");

  if (argc < 2) {
    // Tell the user how to run the program
    std::cerr << " Please provide a prefix for the input files. Exiting ..." << "\n";
    /* "Usage messages" are a conventional way of telling the user
      * how to run a program if they enter the command incorrectly.
      */
    return 1;
  }

  // system parameters
  find_param(flin,"__L=",L);
  find_param(flin,"__HoppingJ=",J);
  find_param(flin,"__HubbardU=",U);
  find_param(flin,"__MuChem=",mu);
  find_param(flin,"__beta=",beta);
  find_param(flin,"__Eint=",Eint);
  find_param(flin,"__Ainitx=",Ainitx);
  
  // solver parameters
  find_param(flin,"__Nt=",Nt);
  find_param(flin,"__Ntau=",Ntau);
  find_param(flin,"__dt=",dt);
  find_param(flin,"__MatsMaxIter=",matsMaxIter);
  find_param(flin,"__BootstrapMaxIter=",BootstrapMaxIter);
  find_param(flin,"__CorrectorSteps=",correctorSteps);
  find_param(flin,"__MatsMaxErr=",matsMaxErr);
  find_param(flin,"__TimeMaxErr=",timeMaxErr);
    
  find_param(flin,"__nlevel=",nlvl);
  find_param(flin,"__tstpPrint=",tstpPrint);
  find_param(flin,"__svdtol=",svdtol);
  find_param(flin,"__dlrlambda=",lambda);
  find_param(flin,"__epsdlr=",epsdlr);

  if(mpi_rank==mpi_root){
    std::cout << "input done" << "\n";
  }
  //============================================================================
  //                       INITIALIZE SOVLER AND BUFFERS
  //============================================================================

  Eigen::setNbThreads(1);
  char *env = getenv("SLURM_CPUS_PER_TASK");
  int nthreads = env ? atoi(env) : 1;
  if(mpi_rank==mpi_root){
    std::cout << " nthreads = " << nthreads << "\n";
  }  
  omp_set_num_threads(nthreads);
  openblas_set_num_threads(1);
  Integration::Integrator I(SolverOrder);

  int rho_version = 1;
  xi = -1;
  r = Ntau;
  hodlr::dlr_info dlr(r, lambda, epsdlr, beta, size, xi);
  hodlr::dyson dyson_sol(Nt, size, SolverOrder, dlr, rho_version);


  //============================================================================
  //                INITIALIZE lattice_2d_1b AND CORRESPONDING PROPAGATORS
  //============================================================================
  lattice_2d_ysymm lattice(L, Nt, dt, J, Eint, mu, size);
  int Nk = lattice.Nk_;

  std::vector<double> density_k(Nk), jxt_local(Nt), jyt_local(Nt), Ekint_local(Nt), Numt_local(Nt);
  std::vector<double> jxt_total(Nt), jyt_total(Nt), Ekint_total(Nt), Numt_total(Nt);
  std::vector<std::vector<double>> jxt_iter_total(BootstrapMaxIter+1,std::vector<double>(Nt));
  std::vector<std::vector<double>> jxt_iter_local(BootstrapMaxIter+1,std::vector<double>(Nt));

  int max_component_size = 2;
  mpi_comm comm(Nk, Nt, r, size, size, 2);
  int Nk_rank = comm.Nk_per_rank[mpi_rank];
  
  std::vector<double> errk(Nk);

  int num = omp_get_max_threads();

  std::vector<hodlr::dlr_info> dlr_vec;
  dlr_vec.reserve(num);
  for(int i = 0; i < num; i++) dlr_vec.emplace_back(r, lambda, epsdlr, beta, size, xi);

  std::vector<std::unique_ptr<hodlr::dyson>> dysonk;
  for(int i=0; i<num; i++){
    dysonk.push_back(std::make_unique<hodlr::dyson>(Nt, size, SolverOrder, dlr_vec[i], rho_version));
  }

  std::vector<std::unique_ptr<kpoint>> corrK_rank;
  int iq=0;
  
  if(!checkpoint_exists){
    for(int k=0;k<Nk;k++){

      if(comm.k_rank_map[k]==mpi_rank){
        corrK_rank.push_back(std::make_unique<kpoint>(Nt,r,nlvl,svdtol,size,beta,dt,SolverOrder,lattice.kpoints_[k],lattice,mu,Ainitx));
        
        iq++;
      }
    }
  } 
  else{
    for(int k=0;k<Nk;k++){

      if(comm.k_rank_map[k]==mpi_rank){
        h5e::File checkpoint_file(output_dir+"GSigma" + std::to_string(k)+".h5", h5e::File::ReadOnly);
        corrK_rank.push_back(std::make_unique<kpoint>(Nt,r,nlvl,svdtol,size,beta,dt,SolverOrder,lattice.kpoints_[k],lattice,mu,Ainitx,checkpoint_file));        
      }
    }
  }

  // ============================================================================
  //                       INITIALIZE GETS AND SETS
  // ============================================================================

  auto getMat = [&corrK_rank, &size](int k, int t, int taui){
    hodlr::DMatrix tmp(size,size);
    corrK_rank[k]->G_.get_mat(taui,tmp);
    return tmp(0,0);
  };

  auto getMatbeta = [&corrK_rank, &size, &dlr](int k, int t, int taui){
    hodlr::DMatrix tmp(size,size);
    int rtmp = corrK_rank[k]->r_;
    int sizetmp = corrK_rank[k]->size_;
    double dest[rtmp*sizetmp*sizetmp];
    corrK_rank[k]->G_.get_mat_reversed(dlr,dest);
    tmp(0,0) = dest[taui];
    return tmp(0,0);
  };

  std::vector<std::function<std::complex<double>(int, int, int)>> getsMat = {getMat,getMatbeta};

  auto getLess = [&corrK_rank, &size, &mpi_rank](int k, int ti, int tpi){
    hodlr::ZMatrix tmp(size,size);
    corrK_rank[k]->G_.get_les(ti,tpi,tmp);
    return tmp(0,0);
  };

  auto getGreat = [&corrK_rank, &size](int k, int ti, int tpi){
    hodlr::ZMatrix tmpRet(size,size),tmpLess(size,size),tmpGreat(size,size);
    if(ti>=tpi){
      corrK_rank[k]->G_.get_les(ti,tpi,tmpLess);
      corrK_rank[k]->G_.get_ret(ti,tpi,tmpRet);
      tmpGreat(0,0) = tmpRet(0,0)+tmpLess(0,0);
    } 
    else{
      corrK_rank[k]->G_.get_les(tpi,ti,tmpLess);
      corrK_rank[k]->G_.get_ret(tpi,ti,tmpRet);
      tmpGreat(0,0) = -std::conj(tmpRet(0,0)+tmpLess(0,0));
    }
    return tmpGreat(0,0);
  };

  std::vector<std::function<std::complex<double>(int, int, int)>> getsLG = {getLess,getGreat};

  auto gettv = [&corrK_rank, &size](int k, int ti, int taui){
    hodlr::ZMatrix tmp(size,size);
    corrK_rank[k]->G_.get_tv(ti,taui,tmp);
    return tmp(0,0);
  };

  auto gettvbeta = [&corrK_rank, &size, &dlr](int k, int ti, int taui){
    int rtmp = corrK_rank[k]->r_;
    int sizetmp = corrK_rank[k]->size_;
    hodlr::cplx dest[rtmp*sizetmp*sizetmp];
    corrK_rank[k]->G_.get_tv_reversed(ti,dlr,dest);
    return dest[taui];
  };

  std::vector<std::function<std::complex<double>(int, int, int)>> getstv = {gettv,gettvbeta};

  auto setMat = [&corrK_rank, &size, &r, &mpi_rank](int k, int t, std::vector<std::complex<double>> &Sigma){
    hodlr::DMatrixMap(corrK_rank[k]->Sigma_.matptr(0), r * size, size).noalias() = hodlr::ZMatrixMap(Sigma.data(), r * size, size).real();
  };

  auto setLess = [&corrK_rank, &size](int k, int ti, std::vector<std::complex<double>> &Sigma){
    std::vector<std::complex<double>> Sigmac(ti+1);
    for(int j = 0; j<=ti; j++){
      Sigmac[j] = -std::conj(Sigma[j]);
    }
    hodlr::ZMatrixMap(corrK_rank[k]->Sigma_.curr_timestep_les_ptr(0,ti), (ti + 1) * size, size).noalias() = hodlr::ZMatrixMap(Sigmac.data(), (ti + 1) * size, size);
  };

  auto setRet = [&corrK_rank, &size, &mpi_rank](int k, int ti, std::vector<std::complex<double>> &Sigma){
    hodlr::ZMatrixMap(corrK_rank[k]->Sigma_.curr_timestep_ret_ptr(ti,0), (ti + 1) * size, size).noalias() = hodlr::ZMatrixMap(Sigma.data(), (ti + 1) * size, size);
  };

  auto settv = [&corrK_rank, &size,&r](int k, int ti, std::vector<std::complex<double>> &Sigma){
    hodlr::ZMatrixMap(corrK_rank[k]->Sigma_.tvptr(ti,0), r * size, size).noalias() = hodlr::ZMatrixMap(Sigma.data(), r * size, size);
  };

  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> setsMat = {setMat};
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> setsLR = {setLess,setRet};   
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> setstv = {settv};


  born_approx_se se_eval(L,Nk,2);  

  int stop_mess = 0;
  int mem_stop_mess = 0;
  int tot_mem_stop_mess = 0;
  int err_mess_mat = 0;
  int err_mess_boot = 0;
  int err_mess_tstp = 0;
  int t0;

  if(!checkpoint_exists){
    //============================================================================
    //                                MATSUBARA
    //============================================================================

    if(mpi_rank==mpi_root){
      std::cout << "---- MATSUBARA PHASE " << "\n";
    }

    start = MPI_Wtime();

    for(int iter=0;iter<=matsMaxIter;iter++){
      double err=0.0;
      double tot_err = 0.0;

      if(err_mess_mat == 1){
        continue;
      }

      // Set Self-energy
      if(iter!=0){
        se_eval.Sigma(-1, U, comm, getsMat, getstv, getsLG, setsMat, setstv, setsLR);
      }
      
      for(int k=0; k<Nk_rank; k++){
        errk[k] = corrK_rank[k]->step_dyson(-1, SolverOrder, lattice, I, dyson_sol, dlr);
      }

      err = std::reduce(errk.begin(),errk.end());

      MPI_Allreduce(&err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 

      if(iter != 0){
        if(tot_err <= matsMaxErr){
          err_mess_mat = 1;
        }
      }
    
      if(mpi_rank==mpi_root){
        std::cout << "------ Matsubara Iteration: "  << iter << " tot err: " << tot_err << "\n";
      }

    }

    end = MPI_Wtime();

    if(mpi_rank==mpi_root){
      double elapsed_seconds = end-start;
      std::cout << "------ Time [equilibrium calculation] = " << elapsed_seconds << "\n";
    }

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

    for(int iter=0;iter<=BootstrapMaxIter;iter++){
      
      if(err_mess_boot == 1){
        continue;
      }

      double err_ele=0.0;
      double tot_err = 0.0;
      for (int tstp = 0;tstp <= SolverOrder;tstp ++){

        if(mpi_rank==mpi_root){
          std::cout << "---- Time step ti = " << tstp << "\n";
        }


        if(iter != 0){
          se_eval.Sigma(tstp, U, comm, getsMat, getstv, getsLG, setsMat, setstv, setsLR);
        }

        for(int k=0;k<Nk_rank;k++){
          errk[k] = corrK_rank[k]->step_dyson(tstp, SolverOrder, lattice, I, dyson_sol, dlr);
        }
          
        err_ele = std::reduce(errk.begin(),errk.end());

        std::array<double, 4> obs = observables::get_obs_local(tstp, Nk_rank, lattice, comm.kindex_rank, corrK_rank);
  
        jxt_local[tstp]=2.0*obs[0]/Eint;
        jyt_local[tstp]=2.0*obs[1]/Eint;
        Ekint_local[tstp]=2.0*obs[2];
        Numt_local[tstp]=obs[3];

        jxt_iter_local[iter][tstp]=2.0*obs[0]/Eint;

      } // Bootstrap time loop

      MPI_Allreduce(&err_ele, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  

      if((tot_err <= timeMaxErr) && (iter!=0)){
        err_mess_boot = 1;
      }

      if(mpi_rank==mpi_root){
        
        std::cout << "---- Bootstrap Iteration: "  << iter << " tot err: " << tot_err << "\n";
      }  

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

  if(mpi_rank==mpi_root){
    std::cout << "----- TIME PROPAGATION PHASE " << "\n";
  }

  start = MPI_Wtime();

  
  if(checkpoint_exists){
    t0=corrK_rank[0]->G_.tstpmk() + corrK_rank[0]->G_.k() + 1;
  } else{
    t0=SolverOrder+1;
  }

  for(int tstp=t0; tstp<Nt; tstp++){

    if(stop_mess == 1){
      continue;
    }

    if(tot_mem_stop_mess >= 1){
      continue;
    }

    err_mess_tstp = 0;
    if(mpi_rank==mpi_root){
      std::cout << "---- Time step ti = " << tstp << "\n";
    }

    // Update Blocks
    #pragma omp parallel for schedule(static)
    for(int k=0;k<Nk_rank;k++){
      corrK_rank[k]->G_.update_blocks(I);
      corrK_rank[k]->Sigma_.update_blocks(I);
    }

    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      #pragma omp for schedule(static)
      for(int k=0;k<Nk_rank;k++){
        dysonk[thread_id] -> Extrapolate(corrK_rank[k]->G_, I);
      }
    }

    for(int iter=0;iter<=correctorSteps;iter++){

      if(err_mess_tstp == 1){
        continue;
      }

      double err_ele=0.0;
      double tot_err = 0.0;

      se_eval.Sigma(tstp, U, comm, getsMat, getstv, getsLG, setsMat, setstv, setsLR);

      #pragma omp parallel
      {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(int k=0;k<Nk_rank;k++){
          errk[k] = corrK_rank[k]->step_dyson(tstp, SolverOrder, lattice, I, *dysonk[thread_id], dlr_vec[thread_id]);
        }
      }

      err_ele = std::reduce(errk.begin(),errk.end()); 

      MPI_Allreduce(&err_ele, &tot_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  

      if((tot_err <= timeMaxErr) && (iter!=0)){err_mess_tstp = 1;}

      std::array<double,4> obs = observables::get_obs_local(tstp,Nk_rank,lattice,comm.kindex_rank,corrK_rank);
  
      jxt_local[tstp]=2.0*obs[0]/Eint;
      jyt_local[tstp]=2.0*obs[1]/Eint;
      Ekint_local[tstp]=2.0*obs[2];
      Numt_local[tstp]=obs[3];
      jxt_iter_local[iter][tstp]=2.0*obs[0]/Eint;
      
      if(mpi_rank==mpi_root){
        std::cout << "------ Time Stepping Iteration: "  << iter << " tot err: " << tot_err << "\n";
      }
      
    }

    double time_now = MPI_Wtime();
    double stop_time = time_now - start_tot;
    if(stop_time>time_limit){
      stop_mess = 1;
    }
    MPI_Bcast(&stop_mess,1,MPI_INT,mpi_root,MPI_COMM_WORLD);


    MemInfo mem = getMemoryInfo();

    if(mem.available<mem_limit){
      mem_stop_mess = 1;
    }

    MPI_Allreduce(&mem_stop_mess, &tot_mem_stop_mess, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);  

    if(stop_mess==1 || tot_mem_stop_mess>=1){

      std::ostringstream filename;
      filename << "checkpoint_top_rank" << mpi_rank << ".txt";

      std::ofstream file(filename.str());
      // if (!file) {
      //   std::cerr << "Error opening file for rank " << mpi_rank << "\n";
      //   MPI_Finalize();
      //   return 1;
      // }

      FILE* topOutput = popen("top -b -n 1 -o %MEM", "r");
      // if (!topOutput) { // Check if popen() failed
      //   std::cerr << "Error running top command for rank " << mpi_rank << "\n";
      //   file << "Error: popen() failed\n";
      //   file.close();
      //   MPI_Finalize();
      //   return 1;
      // }

      char buffer[256];
      while (fgets(buffer, sizeof(buffer), topOutput)) {
        file << buffer;
      }

      pclose(topOutput);
      file.close();

      if(mpi_rank==mpi_root){
        if(stop_mess) std::cout << "REACHED THE TIME LIMIT\n";
      }

      if(mem_stop_mess && (mem.available<mem_limit)) std::cout << "REACHED THE MEMORY LIMIT AT MPI RANK " << mpi_rank <<  " WITH AVAILABLE MEMORY = " << mem.available << "\n";

      // h5e::File checkpoint_file(argv[4], h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
      for(int k = 0; k < Nk_rank; k++) {
        {
          h5e::File checkpoint_file(output_dir + "GSigma" + std::to_string(comm.kindex_rank[k]) + ".h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
          corrK_rank[k]->G_.write_checkpoint_hdf5(checkpoint_file, "G/");
          corrK_rank[k]->Sigma_.write_checkpoint_hdf5(checkpoint_file, "S/");
        }
      }

      MPI_Reduce(jxt_local.data(), jxt_total.data(), jxt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
      MPI_Reduce(jyt_local.data(), jyt_total.data(), jyt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
      MPI_Reduce(Ekint_local.data(), Ekint_total.data(), Ekint_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
      MPI_Reduce(Numt_local.data(), Numt_total.data(), Numt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);

      for(int iter=0; iter<=BootstrapMaxIter; iter++){
        MPI_Reduce(jxt_iter_local[iter].data(), jxt_iter_total[iter].data(), jxt_iter_total[iter].size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
      }

      if(mpi_rank==mpi_root){
        if(checkpoint_exists){
          h5e::File jcheckpoint_file("Currents/jxEpeak_hodlr_checkpoint.h5", h5e::File::ReadOnly);
          std::vector<double> jcheckpoint = jcheckpoint_file.getDataSet("jxtnorm").read<std::vector<double>>();

          for(int ti=0; ti<t0; ti++){
            jxt_total[ti] = jcheckpoint[ti];
          }
        }

        h5e::File out_file("Currents/jx_iter_checkpoint.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
        h5e::dump<std::vector<std::vector<double>>>(out_file, "jxt_iter", jxt_iter_total);

        h5e::File out_file2("Currents/jxEpeak_hodlr_checkpoint.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
        h5e::dump<std::vector<double>>(out_file2, "jxtnorm", jxt_total);
        h5e::dump<std::vector<double>>(out_file2, "jytnorm", jyt_total);
        h5e::dump<std::vector<double>>(out_file2, "Ekint", Ekint_total);
        h5e::dump<std::vector<double>>(out_file2, "Numt", Numt_total);
      }
    }// time checkpoint 

  } // Time propagation loop  

  MPI_Reduce(jxt_local.data(), jxt_total.data(), jxt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(jyt_local.data(), jyt_total.data(), jyt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(Ekint_local.data(), Ekint_total.data(), Ekint_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  MPI_Reduce(Numt_local.data(), Numt_total.data(), Numt_total.size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);

  for(int iter=0; iter<=BootstrapMaxIter; iter++){
    MPI_Reduce(jxt_iter_local[iter].data(), jxt_iter_total[iter].data(), jxt_iter_total[iter].size(), MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
  }

  for(int t = 0; t <= SolverOrder; t++) {
    for(int k = 0; k < Nk_rank; k++) {
      corrK_rank[k]->G_.update_blocks(I);
      corrK_rank[k]->Sigma_.update_blocks(I);
    }
  }

  
  if(mpi_rank==comm.k_rank_map[0]){
    std::cout << " ****** k = " << 0 << "\n";
    std::cout << "GREEN'S FUNCTION " << "\n";
    corrK_rank[0]->G_.print_memory_usage();
    std::cout << "SELF-ENERGY " << "\n";
    corrK_rank[0]->Sigma_.print_memory_usage();
    std::cout << " *********** "<< "\n";
  }
  
  end = MPI_Wtime();

  if(mpi_rank==mpi_root){
    double elapsed_seconds= end -start;
    std::cout << "Time [propagation] = " << elapsed_seconds << "\n";
  }


  if(mpi_rank==mpi_root){

    if(checkpoint_exists){
      h5e::File jcheckpoint_file("Currents/jxEpeak_hodlr_checkpoint_omp.h5", h5e::File::ReadOnly);
      std::vector<double> jcheckpoint = jcheckpoint_file.getDataSet("jxtnorm").read<std::vector<double>>();

      for(int ti=0; ti<t0; ti++){
        jxt_total[ti] = jcheckpoint[ti];
      }
    }

    h5e::File out_file("Currents/jx_iter_omp.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    h5e::dump<std::vector<std::vector<double>>>(out_file, "jxt_iter", jxt_iter_total);

    h5e::File out_file2("Currents/jxEpeak_omp_hodlr.h5", h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
    h5e::dump<std::vector<double>>(out_file2, "jxtnorm", jxt_total);
    h5e::dump<std::vector<double>>(out_file2, "jytnorm", jyt_total);
    h5e::dump<std::vector<double>>(out_file2, "Ekint", Ekint_total);
    h5e::dump<std::vector<double>>(out_file2, "Numt", Numt_total);
    
  }

  std::ostringstream filename;
  filename << "top_rank" << mpi_rank << ".txt";

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

  if(mpi_rank==mpi_root){
    end_tot = MPI_Wtime();
    double elapsed_seconds= end_tot -start_tot;
    std::cout << "Total Time = " << elapsed_seconds << "\n";    
  }


  MPI_Finalize();
  return 0;
}

