#include "lattice.hpp"

lattice_2d_ysymm::lattice_2d_ysymm(void){};

lattice_2d_ysymm::lattice_2d_ysymm(int L, int Nt, double dt, double J, double Amax, double mu, int size)
{
    assert(-1<=Nt);
    assert(-1<=Nt);
    Nt_=Nt;
    size_=size;
    mu_=mu;
    dt_=dt;
    J_=J;
    Nk_ = L*(L/2+1);
    h_nessi::ZMatrix tmp(size,size);
    A_.resize(Nt+2);
    efield_to_afield(Nt, dt, Amax);

    init_kk(L);
   
}

//convert electric field E to vector potential A
void lattice_2d_ysymm::efield_to_afield(int Nt, double dt, double Amax)
{
    Vector2D efield(Amax,0);
    A_.resize(Nt + 1);
    for (int tstp = 0; tstp <= Nt; tstp++){
        A_[tstp] = 1.0 * efield;
    }
}


//define bz
void lattice_2d_ysymm::init_kk(int L)
{
    assert(L > 1); 
    L_ = L;
    double dk = 2 * PI / L_;
    kpoints_.resize(L_ * (L_/2+1));
    for(int ix = 0; ix < L_; ix++) {
        for(int iy = 0; iy < (L_/2+1); iy++) {
            int index = ix * (L_/2+1) + iy;
            kpoints_[index].x() = ix * dk;
            kpoints_[index].y() = iy * dk;
        }
    }
}

//define single-particle hamiltonian
void lattice_2d_ysymm::hk(h_nessi::ZMatrix &hkmatrix, int tstp, Vector2D& kk, double Ainitx) 
{
    Vector2D Atstp(0, 0);
    Vector2D Ainit(Ainitx,0);
    if (tstp > -1) { 
    // Check bounds of A_ && tstp < A_.size()
        Atstp = A_[tstp];
    } 
    double epsk = -mu_-2.0 * J_ * (cos(kk.x() - Atstp.x()-Ainit.x()) + cos(kk.y() - Atstp.y()-Ainit.y()));
    hkmatrix.resize(size_, size_);
    hkmatrix.setZero();
    hkmatrix(0, 0) = epsk;
}

int lattice_2d_ysymm::kflatindex(int kxi, int kyi)
{
    int index = kxi *(L_/2+1) + kyi;
    return index;
}

std::array<int,2> lattice_2d_ysymm::kxikyi(int index)
{
    int kxi = index/(L_/2+1);
    int kyi = index%(L_/2+1);

    std::array<int,2> arr = {kxi,kyi};

    return arr;
}


//define velocity
void lattice_2d_ysymm::vk(h_nessi::ZMatrix &vxkmatrix,h_nessi::ZMatrix &vykmatrix,int tstp, Vector2D& kk)
{
  const Vector2D&  Atstp=A_[tstp];
  
  double vkx=2.0*J_*sin(kk.x()-Atstp.x());
  double vky=2.0*J_*sin(kk.y()-Atstp.y());
  
  vxkmatrix.resize(size_,size_);
  vxkmatrix.setZero();
  vxkmatrix(0,0)=vkx;

  vykmatrix.resize(size_,size_);
  vykmatrix.setZero();
  vykmatrix(0,0)=vky;
}



