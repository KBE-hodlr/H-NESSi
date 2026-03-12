#include "selfene.hpp"

born_approx_se::born_approx_se(){};
born_approx_se::~born_approx_se()
{
  fftw_destroy_plan(Gk_to_Gr);
  fftw_destroy_plan(Sigmar_to_Sigmak);
  fftw_free(in_dummy);
  fftw_free(out_dummy);
  fftw_cleanup();

  for(auto G_plan: G_plans_vec){
    fftw_destroy_plan(G_plan);
  }

  for(auto Sigma_plan: Sigma_plans_vec){
    fftw_destroy_plan(Sigma_plan);
  }
};

born_approx_se::born_approx_se
( double U,
  int L,
  int Nk,
  int nao,
  int nthreads
): U_(U), L_(L), Nk_(Nk), max_component_size_(2*nao*nao)
{  
  in_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_ * L_);
  out_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_ * L_);

  Gk_to_Gr = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_BACKWARD, FFTW_ESTIMATE);
  Sigmar_to_Sigmak = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_FORWARD, FFTW_ESTIMATE);

  Gk = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Gr = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Sigmar = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Sigmak = std::vector<std::vector<std::complex<double>>> (max_component_size_,std::vector<std::complex<double>>(L*L));

  G_plans_vec = std::vector<fftw_plan>(nthreads);
  Sigma_plans_vec = std::vector<fftw_plan>(nthreads);

  for(int tidi=0; tidi<nthreads; tidi++ ){
    G_plans_vec[tidi] = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_BACKWARD, FFTW_ESTIMATE);
    Sigma_plans_vec[tidi] = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_FORWARD, FFTW_ESTIMATE);
  }

  Gk_thread_vec = std::vector<std::vector<std::vector<std::complex<double>>>>(nthreads,std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L)));
  Gr_thread_vec = std::vector<std::vector<std::vector<std::complex<double>>>>(nthreads,std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L)));
  Sigmar_thread_vec = std::vector<std::vector<std::vector<std::complex<double>>>>(nthreads,std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L)));
  Sigmak_thread_vec = std::vector<std::vector<std::vector<std::complex<double>>>>(nthreads,std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L)));

  lambda_GLGr_to_SigmaLRr = [U] (std::vector<std::vector<std::complex<double>>> GLGr){
    return GLGr_to_SigmaLRr(U,GLGr);
  };
  
  lambda_Gtvr_to_Sigmatvr = [U] (std::vector<std::vector<std::complex<double>>> Gtvr){
    return Gtvr_to_Sigmatvr(U,Gtvr);
  };

  lambda_GMatr_to_SigmaMatr = [U] (std::vector<std::vector<std::complex<double>>> GMatr){
    return GMatr_to_SigmaMatr(U, GMatr);
  };

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  double mib = memory_usage();
  double mib_total;
  MPI_Reduce(&mib, &mib_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(mpi_rank==0){
    std::cout << "----------------------------------------\n";
    std::cout << "born_approx_se instance memory usage per rank: " << mib << " MiB\n";
    std::cout << "born_approx_se instance memory usage across all ranks: " << mib_total << " MiB, " << mib_total/(1024.0) <<" GiB\n";
    std::cout << "----------------------------------------\n";
  }

};

void born_approx_se::cleanup()
{
  fftw_destroy_plan(Gk_to_Gr);
  fftw_destroy_plan(Sigmar_to_Sigmak);
  fftw_free(in_dummy);
  fftw_free(out_dummy);
  fftw_cleanup();

  for(auto G_plan: G_plans_vec){
    fftw_destroy_plan(G_plan);
  }

  for(auto Sigma_plan: Sigma_plans_vec){
    fftw_destroy_plan(Sigma_plan);
  }
};

double born_approx_se::memory_usage() const
{
  std::size_t bytes = 0;

  // account for outer vectors (vector of vectors)
  // Gk, Gr, Sigmar, Sigmak
  bytes += Gk.capacity() * sizeof(std::vector<std::complex<double>>);
  bytes += Gr.capacity() * sizeof(std::vector<std::complex<double>>);
  bytes += Sigmar.capacity() * sizeof(std::vector<std::complex<double>>);
  bytes += Sigmak.capacity() * sizeof(std::vector<std::complex<double>>);

  // inner vectors for the above
  for (const auto &v : Gk) bytes += v.capacity() * sizeof(std::complex<double>);
  for (const auto &v : Gr) bytes += v.capacity() * sizeof(std::complex<double>);
  for (const auto &v : Sigmar) bytes += v.capacity() * sizeof(std::complex<double>);
  for (const auto &v : Sigmak) bytes += v.capacity() * sizeof(std::complex<double>);

  // FFTW plan vectors
  bytes += G_plans_vec.capacity() * sizeof(fftw_plan);
  bytes += Sigma_plans_vec.capacity() * sizeof(fftw_plan);

  // thread-local 3D vectors: outer vector of threads, then component vectors, then data
  bytes += Gk_thread_vec.capacity() * sizeof(std::vector<std::vector<std::complex<double>>>);
  bytes += Gr_thread_vec.capacity() * sizeof(std::vector<std::vector<std::complex<double>>>);
  bytes += Sigmar_thread_vec.capacity() * sizeof(std::vector<std::vector<std::complex<double>>>);
  bytes += Sigmak_thread_vec.capacity() * sizeof(std::vector<std::vector<std::complex<double>>>);

  for (const auto &thread_vec : Gk_thread_vec) {
    bytes += thread_vec.capacity() * sizeof(std::vector<std::complex<double>>);
    for (const auto &comp_vec : thread_vec) bytes += comp_vec.capacity() * sizeof(std::complex<double>);
  }
  for (const auto &thread_vec : Gr_thread_vec) {
    bytes += thread_vec.capacity() * sizeof(std::vector<std::complex<double>>);
    for (const auto &comp_vec : thread_vec) bytes += comp_vec.capacity() * sizeof(std::complex<double>);
  }
  for (const auto &thread_vec : Sigmar_thread_vec) {
    bytes += thread_vec.capacity() * sizeof(std::vector<std::complex<double>>);
    for (const auto &comp_vec : thread_vec) bytes += comp_vec.capacity() * sizeof(std::complex<double>);
  }
  for (const auto &thread_vec : Sigmak_thread_vec) {
    bytes += thread_vec.capacity() * sizeof(std::vector<std::complex<double>>);
    for (const auto &comp_vec : thread_vec) bytes += comp_vec.capacity() * sizeof(std::complex<double>);
  }

  // FFTW allocated arrays (in_dummy/out_dummy)
  if (in_dummy) bytes += (std::size_t)(L_ * L_) * sizeof(fftw_complex);
  if (out_dummy) bytes += (std::size_t)(L_ * L_) * sizeof(fftw_complex);

  // small fixed-size members (std::function objects)
  bytes += sizeof(lambda_GLGr_to_SigmaLRr);
  bytes += sizeof(lambda_Gtvr_to_Sigmatvr);
  bytes += sizeof(lambda_GMatr_to_SigmaMatr);

  // convert to MiB
  double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
  return mib;
};

std::vector<std::vector<std::complex<double>>> GMatr_to_SigmaMatr
(double U, std::vector<std::vector<std::complex<double>>> &GMatr)
{
  int Nk = std::sqrt(GMatr[0].size());
  std::vector<std::vector<std::complex<double>>> SigmaMatr(1, std::vector<std::complex<double>>(Nk*Nk));
  for (int x = 0; x < Nk; x++) {
    for (int y = 0; y < Nk; y++) {

      int ri = iflatten2(x, y, Nk);

      int ix = minus_xi(x, Nk);
      int iy = minus_xi(y, Nk);
      int iri = iflatten2(ix,iy,Nk);

      double wk = Nk*Nk;                  

      SigmaMatr[0][ri]= U*U*pow(GMatr[0][ri],2)*std::conj(GMatr[1][iri]);
    }
  }
  return SigmaMatr;

}


std::vector<std::vector<std::complex<double>>> Gtvr_to_Sigmatvr
(double U, std::vector<std::vector<std::complex<double>>> &Gtvr)
{
  int Nk = std::sqrt(Gtvr[0].size());
  std::vector<std::vector<std::complex<double>>> Sigmatvr(1, std::vector<std::complex<double>>(Nk*Nk));
  for (int x = 0; x < Nk; x++) {
    for (int y = 0; y < Nk; y++) {

      int ri = iflatten2(x, y, Nk);

      double wk = Nk*Nk;                  

      Sigmatvr[0][ri]= U*U*pow(Gtvr[0][ri],2)*std::conj(Gtvr[1][ri]);
    }
  }
  return Sigmatvr;
}


std::vector<std::vector<std::complex<double>>> GLGr_to_SigmaLRr
(double U, std::vector<std::vector<std::complex<double>>> &GLGr)
{
  int Nk = std::sqrt(GLGr[0].size());
  std::vector<std::vector<std::complex<double>>> SigmaLRr(2, std::vector<std::complex<double>>(Nk*Nk));

  for (int x = 0; x < Nk; x++) {
    for (int y = 0; y < Nk; y++) {

      int ri = iflatten2(x, y, Nk);

      double wk = Nk*Nk;                  

      std::complex<double> ML = -U*U*pow(GLGr[0][ri],2)*std::conj(GLGr[1][ri]);
      std::complex<double> MG = -U*U*pow(GLGr[1][ri],2)*std::conj(GLGr[0][ri]);

      SigmaLRr[0][ri]= ML;

      SigmaLRr[1][ri]= MG-ML;
    }
  }
  return SigmaLRr;
};

std::vector<std::complex<double>> born_approx_se::fft_Green(std::vector<std::complex<double>> &Gk, const fftw_plan &Gk_to_Gr)
{
  int Nk = std::sqrt(Gk.size());
  const std::complex<double> I(0.0,1.0);

  fftw_complex *in_Gk, *out_Gr;

  in_Gk = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nk * Nk);
  out_Gr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nk * Nk);

  if (!in_Gk || !out_Gr) {
      std::cerr << "Error allocating memory for FFTW arrays" << std::endl;
  }

  for (int ki=0; ki<Nk*Nk; ki++){
      in_Gk[ki][0] = std::real(Gk[ki]);
      in_Gk[ki][1] = std::imag(Gk[ki]);
  }
  
  fftw_execute_dft(Gk_to_Gr, in_Gk, out_Gr);

  std::vector<std::complex<double>> Gr(Nk*Nk);
  double wk = Nk*Nk;
  for (int ri=0; ri<Nk*Nk; ri++){
      Gr[ri] = out_Gr[ri][0]/wk+I*out_Gr[ri][1]/wk;
  }

  fftw_free(in_Gk);
  fftw_free(out_Gr);

  return Gr;
};

std::vector<std::complex<double>> born_approx_se::fft_Sigma(std::vector<std::complex<double>> &Sigmar,const fftw_plan &Sigmar_to_Sigmak)
{
  const std::complex<double> I(0.0,1.0);
  int Nk = std::sqrt(Sigmar.size());
  fftw_complex *in_Sigmar, *out_Sigmak;

  in_Sigmar = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nk * Nk);
  out_Sigmak = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nk * Nk);


  if (!in_Sigmar || !out_Sigmak) {
      std::cerr << "Error allocating memory for FFTW arrays" << std::endl;
  }

  for (int ri=0; ri<Nk*Nk; ri++){
      in_Sigmar[ri][0] = std::real(Sigmar[ri]);
      in_Sigmar[ri][1] = std::imag(Sigmar[ri]);
  }

  fftw_execute_dft(Sigmar_to_Sigmak, in_Sigmar, out_Sigmak);

  std::vector<std::complex<double>> Sigmak(Nk*Nk);
  for (int ki=0; ki<Nk*Nk; ki++){
      Sigmak[ki] = out_Sigmak[ki][0]+I*out_Sigmak[ki][1];
  }

  fftw_free(in_Sigmar);
  fftw_free(out_Sigmak);

  return Sigmak;
};

void born_approx_se::Sigma_tau_spawn
( int my_Ntau,
  mpi_comm &comm,
  int getsSize,
  int setsSize,
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
)
{ 
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  #pragma omp parallel firstprivate(Gk_to_Gr,Sigmar_to_Sigmak,tmp,Gk,Gr,Sigmar,Sigmak)
  { 
    #pragma omp for schedule(static) 
    for(int taui=0; taui<my_Ntau; taui++){
      int local_k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<getsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int ikyi = minus_xi(kyi,L_);
          int index = comm.tau_allk_buff_index(taui,compi,local_k_index,ki);

          tmp = comm.tau_allk_buff[index];

          Gk[compi][kxi*L_+kyi]=tmp;
          Gk[compi][kxi*L_+ikyi]=tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }

      for (int compi = 0; compi<getsSize; compi++){
        Gr[compi] = fft_Green(Gk[compi], Gk_to_Gr);
      }

      Sigmar = Gr_to_Sigmar(Gr);

      for (int compi = 0; compi<setsSize; compi++){
        Sigmak[compi] = fft_Sigma(Sigmar[compi],Sigmar_to_Sigmak);
      }
      
      local_k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<setsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int index = comm.tau_allk_buff_index(taui,compi,local_k_index,ki);
          tmp = Sigmak[compi][kxi*L_+kyi];
          comm.tau_allk_buff[index] = tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }
    }
  } //omp
};

void born_approx_se::Sigma_tau_nospawn
( int my_Ntau,
  mpi_comm &comm,
  int getsSize,
  int setsSize,
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
)
{ 
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();
  
  std::complex<double> tmp(0.0,0.0);

  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Ntau, 0);

  int th_my_init_tau = indxs[0];
  int th_my_end_tau = indxs[1];
  int th_my_Ntau = indxs[2];

  if(th_my_Ntau != 0){ 
    for(int taui=th_my_init_tau; taui<=th_my_end_tau; taui++){
      int local_k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<getsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int ikyi = minus_xi(kyi,L_);
          int index = comm.tau_allk_buff_index(taui,compi,local_k_index,ki);

          tmp = comm.tau_allk_buff[index];

          Gk_thread_vec[thread_id][compi][kxi*L_+kyi]=tmp;
          Gk_thread_vec[thread_id][compi][kxi*L_+ikyi]=tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }

      for (int compi = 0; compi<getsSize; compi++){
        Gr_thread_vec[thread_id][compi] = fft_Green(Gk_thread_vec[thread_id][compi], G_plans_vec[thread_id]);
      }

      Sigmar_thread_vec[thread_id] = Gr_to_Sigmar(Gr_thread_vec[thread_id]);

      for (int compi = 0; compi<setsSize; compi++){
        Sigmak_thread_vec[thread_id][compi] = fft_Sigma(Sigmar_thread_vec[thread_id][compi], Sigma_plans_vec[thread_id]);
      }
      
      local_k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<setsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int index = comm.tau_allk_buff_index(taui,compi,local_k_index,ki);
          tmp = Sigmak_thread_vec[thread_id][compi][kxi*L_+kyi];
          comm.tau_allk_buff[index] = tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }
    }
  } //omp

  #pragma omp barrier 
};

void born_approx_se::Sigma_t_spawn
( int my_Ntp,
  mpi_comm &comm,
  int getsSize,
  int setsSize,
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
)
{ 
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  #pragma omp parallel firstprivate(Gk_to_Gr,Sigmar_to_Sigmak,tmp,Gk,Gr,Sigmar,Sigmak)
  { 
    #pragma omp for schedule(static) 
    for(int tpi=0; tpi<my_Ntp; tpi++){
      int local_k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
          for (int compi = 0; compi<getsSize; compi++){
            int kxi = ki/(L_/2+1);
            int kyi = ki%(L_/2+1);
            int ikyi = minus_xi(kyi,L_);
            int index = comm.t_allk_buff_index(tpi,compi,local_k_index,ki);

            tmp = comm.t_allk_buff[index];

            Gk[compi][kxi*L_+kyi]=tmp;
            Gk[compi][kxi*L_+ikyi]=tmp;
          }

          local_k_index = local_k_index+1;
          if(local_k_index>=comm.Nk_per_rank[mpi_index]){
            local_k_index = 0;
            mpi_index = mpi_index + 1;
          }

      }

      for (int compi = 0; compi<getsSize; compi++){
          Gr[compi] = fft_Green(Gk[compi], Gk_to_Gr);
      }

      Sigmar = Gr_to_Sigmar(Gr);

      for (int compi = 0; compi<setsSize; compi++){
          Sigmak[compi] = fft_Sigma(Sigmar[compi],Sigmar_to_Sigmak);
      }
      
      local_k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
          for (int compi = 0; compi<setsSize; compi++){
            int kxi = ki/(L_/2+1);
            int kyi = ki%(L_/2+1);
            int index = comm.t_allk_buff_index(tpi,compi,local_k_index,ki);
            tmp = Sigmak[compi][kxi*L_+kyi];
            comm.t_allk_buff[index] = tmp;
          }

          local_k_index = local_k_index+1;
          if(local_k_index>=comm.Nk_per_rank[mpi_index]){
            local_k_index = 0;
            mpi_index = mpi_index + 1;
          }
      }
    }
  } //omp
};

void born_approx_se::Sigma_t_nospawn
( int my_Ntp,
  mpi_comm &comm,
  int getsSize,
  int setsSize,
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
)
{ 
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int thread_id = omp_get_thread_num();
  int nthreads = omp_get_max_threads();
  
  std::complex<double> tmp(0.0,0.0);

  std::array<int,3> indxs = get_my_index(thread_id,nthreads,my_Ntp, 0);

  int th_my_init_tp = indxs[0];
  int th_my_end_tp = indxs[1];
  int th_my_Ntp = indxs[2];

  if(th_my_Ntp != 0){
    for(int tpi=th_my_init_tp; tpi<=th_my_end_tp; tpi++){
      int local_k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<getsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int ikyi = minus_xi(kyi,L_);
          int index = comm.t_allk_buff_index(tpi,compi,local_k_index,ki);

          tmp = comm.t_allk_buff[index];

          Gk_thread_vec[thread_id][compi][kxi*L_+kyi]=tmp;
          Gk_thread_vec[thread_id][compi][kxi*L_+ikyi]=tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }

      }

      for (int compi = 0; compi<getsSize; compi++){
        Gr_thread_vec[thread_id][compi] = fft_Green(Gk_thread_vec[thread_id][compi], G_plans_vec[thread_id]);
      }

      Sigmar_thread_vec[thread_id] = Gr_to_Sigmar(Gr_thread_vec[thread_id]);

      for (int compi = 0; compi<setsSize; compi++){
        Sigmak_thread_vec[thread_id][compi] = fft_Sigma(Sigmar_thread_vec[thread_id][compi],Sigma_plans_vec[thread_id]);
      }
      
      local_k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<setsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int index = comm.t_allk_buff_index(tpi,compi,local_k_index,ki);
          tmp = Sigmak_thread_vec[thread_id][compi][kxi*L_+kyi];
          comm.t_allk_buff[index] = tmp;
        }

        local_k_index = local_k_index+1;
        if(local_k_index>=comm.Nk_per_rank[mpi_index]){
          local_k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }
    }
  } //omp

  #pragma omp barrier 
};

void born_approx_se::Sigma_spawn
( int tstp,
  mpi_comm &comm,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getsMat,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getstv,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsMat,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLG
)
{
  
  double full_time = MPI_Wtime();
  int mpi_rank, mpi_size;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(tstp==-1){
    comm.mpi_get_and_comm_spawn(tstp, getsLG, getsMat);
    
    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau_spawn(my_Ntau, comm, getsMat.size(), setsMat.size(), lambda_GMatr_to_SigmaMatr);
    }

    comm.mpi_comm_and_set_spawn(tstp,setsLG,setsMat);

  } else{
    comm.mpi_get_and_comm_spawn(tstp, getsLG, getstv);
    
    std::array<int,3> indxs = get_my_index(mpi_rank,mpi_size,tstp+1,0);
    int my_Ntp = indxs[2];
    
    if(my_Ntp != 0){
      Sigma_t_spawn(my_Ntp, comm, getsLG.size(), setsLG.size(), lambda_GLGr_to_SigmaLRr);
    }

    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau_spawn(my_Ntau, comm, getstv.size(), setstv.size(), lambda_Gtvr_to_Sigmatvr);
    }
    comm.mpi_comm_and_set_spawn(tstp,setsLG,setstv);
  }
};

void born_approx_se::Sigma_nospawn
( int tstp,
  mpi_comm &comm,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getsMat,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getstv,
  std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsMat,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv,
  std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLG
)
{
  
  double full_time = MPI_Wtime();
  int mpi_rank, mpi_size;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(tstp==-1){
    #pragma omp barrier
    
    comm.mpi_get_and_comm_nospawn(tstp, getsLG, getsMat);
    
    #pragma omp barrier

    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau_nospawn(my_Ntau, comm, getsMat.size(), setsMat.size(), lambda_GMatr_to_SigmaMatr);
    }

    #pragma omp barrier

    comm.mpi_comm_and_set_nospawn(tstp,setsLG,setsMat);

    #pragma omp barrier

  } else{
    #pragma omp barrier

    comm.mpi_get_and_comm_nospawn(tstp, getsLG, getstv);

    #pragma omp barrier
    
    std::array<int,3> indxs = get_my_index(mpi_rank,mpi_size,tstp+1,0);
    int my_Ntp = indxs[2];
    if(my_Ntp != 0){
      Sigma_t_nospawn(my_Ntp, comm, getsLG.size(), setsLG.size(), lambda_GLGr_to_SigmaLRr);
    }

    #pragma omp barrier

    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau_nospawn(my_Ntau, comm, getstv.size(), setstv.size(), lambda_Gtvr_to_Sigmatvr);
    }

    #pragma omp barrier
    
    comm.mpi_comm_and_set_nospawn(tstp,setsLG,setstv);

    #pragma omp barrier

  }
};

