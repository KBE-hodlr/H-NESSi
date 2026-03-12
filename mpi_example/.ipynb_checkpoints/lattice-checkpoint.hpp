#pragma once

#include <complex>

#include "hodlr/herm_matrix_hodlr.hpp"

using Vector2D = Eigen::Vector2d;

class lattice_2d_ysymm{

public:
  int Nt_;
  double dt_;
  int L_;
  int Nk_;

  std::vector<Vector2D> kpoints_;

  int size_;
  double mu_;
  double U_;
  double J_;
  std::vector<Vector2D> A_;

  lattice_2d_ysymm(void);
  lattice_2d_ysymm(int L, int Nt, double dt, double J, double Eint, double mu, int size);
  int kflatindex(int kxi, int kyi);
  std::array<int,2> kxikyi(int index);
  void init_kk(int L);
  void hk(hodlr::ZMatrix &hkmatrix,int tstp, Vector2D& kk, double Ainitx);
  void efield_to_afield(int Nt, double dt, double Eint);
  void vk(hodlr::ZMatrix &vxkmatrix, hodlr::ZMatrix &vykmatrix, int tstp, Vector2D& kk);

};

