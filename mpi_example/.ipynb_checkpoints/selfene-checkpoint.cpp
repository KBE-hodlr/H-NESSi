#include "selfene.hpp"

born_approx_se::born_approx_se(){};
born_approx_se::~born_approx_se()
{
  fftw_destroy_plan(Gk_to_Gr);
  fftw_destroy_plan(Sigmar_to_Sigmak);
  fftw_free(in_dummy);
  fftw_free(out_dummy);
  fftw_cleanup();
};

born_approx_se::born_approx_se
( int L,
  int Nk,
  int max_component_size
): L_(L), Nk_(Nk), max_component_size_(max_component_size)
{  
  in_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_ * L_);
  out_dummy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * L_ * L_);

  Gk_to_Gr = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_BACKWARD, FFTW_ESTIMATE);
  Sigmar_to_Sigmak = fftw_plan_dft_2d(L_, L_, in_dummy, out_dummy, FFTW_FORWARD, FFTW_ESTIMATE);

  Gk = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Gr = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Sigmar = std::vector<std::vector<std::complex<double>>>(max_component_size_,std::vector<std::complex<double>>(L*L));
  Sigmak = std::vector<std::vector<std::complex<double>>> (max_component_size_,std::vector<std::complex<double>>(L*L));

};

void born_approx_se::cleanup()
{
  fftw_destroy_plan(Gk_to_Gr);
  fftw_destroy_plan(Sigmar_to_Sigmak);
  fftw_free(in_dummy);
  fftw_free(out_dummy);
  fftw_cleanup();
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

void born_approx_se::Sigma_tau
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
      int k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<getsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int ikyi = minus_xi(kyi,L_);
          int index = comm.tau_big_buff_index(taui,compi,k_index,ki);

          tmp = comm.tau_big_buff[index];

          Gk[compi][kxi*L_+kyi]=tmp;
          Gk[compi][kxi*L_+ikyi]=tmp;
        }

        k_index = k_index+1;
        if(k_index>=comm.Nk_per_rank[mpi_index]){
          k_index = 0;
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
      
      k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
        for (int compi = 0; compi<setsSize; compi++){
          int kxi = ki/(L_/2+1);
          int kyi = ki%(L_/2+1);
          int index = comm.tau_big_buff_index(taui,compi,k_index,ki);
          tmp = Sigmak[compi][kxi*L_+kyi];
          comm.tau_big_buff[index] = tmp;
        }

        k_index = k_index+1;
        if(k_index>=comm.Nk_per_rank[mpi_index]){
          k_index = 0;
          mpi_index = mpi_index + 1;
        }
      }
    }
  } //omp
};

void born_approx_se::Sigma_t
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
      int k_index = 0;
      int mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
          for (int compi = 0; compi<getsSize; compi++){
            int kxi = ki/(L_/2+1);
            int kyi = ki%(L_/2+1);
            int ikyi = minus_xi(kyi,L_);
            int index = comm.t_big_buff_index(tpi,compi,k_index,ki);

            tmp = comm.t_big_buff[index];

            Gk[compi][kxi*L_+kyi]=tmp;
            Gk[compi][kxi*L_+ikyi]=tmp;
          }

          k_index = k_index+1;
          if(k_index>=comm.Nk_per_rank[mpi_index]){
            k_index = 0;
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
      
      k_index = 0;
      mpi_index = 0;
      for(int ki = 0; ki < Nk_; ki++){
          for (int compi = 0; compi<setsSize; compi++){
            int kxi = ki/(L_/2+1);
            int kyi = ki%(L_/2+1);
            int index = comm.t_big_buff_index(tpi,compi,k_index,ki);
            tmp = Sigmak[compi][kxi*L_+kyi];
            comm.t_big_buff[index] = tmp;
          }

          k_index = k_index+1;
          if(k_index>=comm.Nk_per_rank[mpi_index]){
            k_index = 0;
            mpi_index = mpi_index + 1;
          }
      }
    }
  } //omp
};

void born_approx_se::Sigma
( int tstp,
  double U,
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

  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> 
  lambda_GLGr_to_SigmaLRr = [U] (std::vector<std::vector<std::complex<double>>> GLGr){
    return GLGr_to_SigmaLRr(U,GLGr);
  };
  
  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> 
  lambda_Gtvr_to_Sigmatvr = [U] (std::vector<std::vector<std::complex<double>>> Gtvr){
    return Gtvr_to_Sigmatvr(U,Gtvr);
  };

  std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> 
  lambda_GMatr_to_SigmaMatr = [U] (std::vector<std::vector<std::complex<double>>> GMatr){
    return GMatr_to_SigmaMatr(U, GMatr);
  };

  if(tstp==-1){
    comm.mpi_get_and_comm(tstp, getsLG, getsMat);
    
    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau(my_Ntau, comm, getsMat.size(), setsMat.size(), lambda_GMatr_to_SigmaMatr);
    }

    comm.mpi_comm_and_set(tstp,setsLG,setsMat);

  } else{
    comm.mpi_get_and_comm(tstp, getsLG, getstv);
    
    int my_Ntp = comm.Ntp_per_tstp_rank[tstp][mpi_rank];
    if(my_Ntp != 0){
      Sigma_t(my_Ntp, comm, getsLG.size(), setsLG.size(), lambda_GLGr_to_SigmaLRr);
    }

    int my_Ntau = comm.Ntau_per_rank[mpi_rank];
    if(my_Ntau != 0){
      Sigma_tau(my_Ntau, comm, getstv.size(), setstv.size(), lambda_Gtvr_to_Sigmatvr);
    }
    comm.mpi_comm_and_set(tstp,setsLG,setstv);
  }
}
