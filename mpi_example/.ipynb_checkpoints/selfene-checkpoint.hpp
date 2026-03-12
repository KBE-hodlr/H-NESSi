#pragma once

#include <iostream>
#include <fftw3.h>
#include "mpi_comm.hpp"

class born_approx_se{
  public:
  born_approx_se();
  ~born_approx_se();

  born_approx_se( int L, int Nk, int max_component_size);
  void cleanup();

  std::vector<std::complex<double>> fft_Green(std::vector<std::complex<double>> &Gk, const fftw_plan &Gk_to_Gr);
  std::vector<std::complex<double>> fft_Sigma(std::vector<std::complex<double>> &Sigmar,const fftw_plan &Sigmar_to_Sigmak);

  void Sigma_tau
  ( int my_Ntau,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  void Sigma_t
  ( int my_Ntp,
    mpi_comm &comm,
    int getsSize,
    int setsSize,
    std::function<std::vector<std::vector<std::complex<double>>> (std::vector<std::vector<std::complex<double>>>)> &Gr_to_Sigmar
  );

  void Sigma
  ( int tstp,
    double U,
    mpi_comm &comm,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsMat,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getstv,
    std::vector<std::function<std::complex<double>(int, int, int)>> &getsLG,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsMat,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setstv,
    std::vector<std::function<void(int, int, std::vector<std::complex<double>>&)>> &setsLG
  );


  int L_;
  int Nk_;
  int max_component_size_;
  std::complex<double> tmp;
  std::vector<std::vector<std::complex<double>>> Gk;
  std::vector<std::vector<std::complex<double>>> Gr;
  std::vector<std::vector<std::complex<double>>> Sigmar;
  std::vector<std::vector<std::complex<double>>> Sigmak;

  fftw_complex *in_dummy, *out_dummy;
  fftw_plan Gk_to_Gr, Sigmar_to_Sigmak;

};
