#include "kpoint.hpp"

kpoint::kpoint(void){}
kpoint::~kpoint(void){}

kpoint::kpoint(int nt,
  int r,
  int nlvl,
  double svdtol,
  int size, 
  double beta, 
  double dt, 
  int SolverOrder,
  Vector2D& kk, 
  lattice_2d_ysymm &latt, 
  double mu,
  double Ainitx):
  G_(nt, r, nlvl, svdtol, size, size, -1 ,SolverOrder),
  Sigma_(nt, r, nlvl, svdtol, size, size, -1 ,SolverOrder)
  {

    beta_=beta;
    dt_=dt;
    nt_=nt;
    // ntau_=ntau;
    r_=r;
    svdtol_=svdtol;
    nlvl_=nlvl;
    size_=size;
    mu_=mu;
    kk_=kk;

    hk_ = h_nessi::function(nt_,size_,size_);

    rho_.resize(nt_);
    for (auto& mat : rho_) mat = h_nessi::DMatrix::Zero(size, size);
  
    for (int tstp=-1; tstp<nt_; tstp++) set_hk(tstp,latt,Ainitx);
  }

kpoint::kpoint(int nt,
  int r,
  int nlvl,
  double svdtol,
  int size, 
  double beta, 
  double dt, 
  int SolverOrder,
  Vector2D& kk, 
  lattice_2d_ysymm &latt, 
  double mu,
  double Ainitx,
  h5e::File &checkpoint_file):
  G_(checkpoint_file, "G/"),
  Sigma_(checkpoint_file, "S/")
  {
    beta_=beta;
    dt_=dt;
    nt_=nt;
    // ntau_=ntau;
    r_=r;
    svdtol_=svdtol;
    nlvl_=nlvl;
    size_=size;
    mu_=mu;
    kk_=kk;

    hk_ = h_nessi::function(nt_,size_,size_);

    rho_.resize(nt_);
    for (auto& mat : rho_) mat = h_nessi::DMatrix::Zero(size, size);
  
    for (int tstp=-1; tstp<nt_; tstp++) set_hk(tstp,latt,Ainitx);
  }

void kpoint::set_hk(int tstp,lattice_2d_ysymm &latt,double Ainitx)
{
  assert(-1<=tstp && tstp<nt_);
  h_nessi::ZMatrix hktmp(1,1);
  latt.hk(hktmp,tstp,kk_,Ainitx);
  hk_.set_value(tstp, hktmp);
}

void kpoint::get_Density_matrix(int tstp, h_nessi::dlr_info &dlr)
{
  h_nessi::DMatrix tmp(size_,size_);
  G_.density_matrix(tstp,dlr,tmp);
  rho_[tstp] = tmp;
}

double kpoint::step_dyson(int tstp, int SolverOrder, lattice_2d_ysymm &latt, Integration::Integrator &I, h_nessi::dyson &dyson_sol, h_nessi::dlr_info &dlr)
{
  // Solve Dyson
  if(tstp==-1){
    double mu_temp = 0.0;
    double alpha=1.0;
    double err = dyson_sol.dyson_mat(G_, mu_temp, hk_, Sigma_, false, alpha);
    return std::pow(err,2);
    
  } else if(tstp<=SolverOrder){
    double mu_temp = 0.0;
    double err = dyson_sol.dyson_start_ntti(G_,mu_temp,hk_,Sigma_,I,dt_,false);
    for(int i = 0; i <= SolverOrder; i++) get_Density_matrix(i,dlr);
    return std::pow(err,2);
    
  } else{
    double mu_temp = 0.0;
    double err = dyson_sol.dyson_timestep(tstp,G_,mu_temp,hk_,Sigma_,I,dt_);
    get_Density_matrix(tstp,dlr);
    return std::pow(err,2);
  }
}



