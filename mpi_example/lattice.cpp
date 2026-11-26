#include "lattice.hpp"

lattice_2d_ysymm::lattice_2d_ysymm(void){};

lattice_2d_ysymm::lattice_2d_ysymm(int L, int Nt, double dt, double J, double Amax, double mu, int size)
{
    assert(-1<=Nt);
    Nt_=Nt;
    size_=size;
    mu_=mu;
    dt_=dt;
    J_=J;
    Nk_ = L*(L/2+1);
    efield_to_afield(Nt, dt, Amax);   // sizes A_ to Nt+1

    init_kk(L);
   
}

//convert electric field E to vector potential A
// Convention: A(t) = +Amax (x direction) for every t >= 0 (A = 0 at t = -1, the equilibrium state), i.e. a step at t = 0+.
// The paper (Eq. 29) uses A(t) = -Amax theta(t), so that E(t) = -dA/dt = +Amax delta(t). Here E(t) = -Amax delta(t), and
// the stored jxtnorm = <j_x>(t)/Amax (Main.cpp) is -sigma(t) of paper Eq. 30. Kept on purpose (decision D-7).
void lattice_2d_ysymm::efield_to_afield(int Nt, double dt, double Amax)
{
    Vector2D efield(Amax,0);
    A_.resize(Nt + 1);
    for (int tstp = 0; tstp <= Nt; tstp++){
        A_[tstp] = 1.0 * efield;
    }
}


// Smooth, fully contained pulse (input __pulse= smooth; see lattice.hpp). Same sign convention as the step: the code's
// A(t) enters as eps_{k - A(t)}, and E(t) = -dA/dt (paper Eq. 29), so the deposited energy is int E j dt (Main.cpp).
void lattice_2d_ysymm::smooth_pulse_afield(int Nt, double dt, double Amax, double t_start, double T_pulse, double omega,
                                           int npow)
{
    A_.resize(Nt + 1);
    for (int tstp = 0; tstp <= Nt; tstp++){
        double x = tstp * dt - t_start;
        double ax = 0.0;
        if (x > 0.0 && x < T_pulse) {
            double env = std::pow(std::sin(PI * x / T_pulse), npow);
            ax = (omega == 0.0) ? Amax * env : Amax * env * std::sin(omega * x);
        }
        A_[tstp] = Vector2D(ax, 0.0);
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
  
  // v = d eps/dk at k - A(t) - Ainit (the argument of hk). Ainitx_ = 0 unless __Ainitx= is set (before WP8b the
  // Ainit shift was missing here although hk has it).
  double vkx=2.0*J_*sin(kk.x()-Atstp.x()-Ainitx_);
  double vky=2.0*J_*sin(kk.y()-Atstp.y());
  
  vxkmatrix.resize(size_,size_);
  vxkmatrix.setZero();
  vxkmatrix(0,0)=vkx;

  vykmatrix.resize(size_,size_);
  vykmatrix.setZero();
  vykmatrix(0,0)=vky;
}

// Band energy eps_{k-A(t)-Ainit} without the chemical potential (A = 0 at tstp = -1, as in hk)
double lattice_2d_ysymm::epsk_band(int tstp, Vector2D& kk)
{
    Vector2D Atstp(0, 0);
    if (tstp > -1) Atstp = A_[tstp];
    return -2.0 * J_ * (cos(kk.x() - Atstp.x() - Ainitx_) + cos(kk.y() - Atstp.y()));
}
