#pragma once

#include "lattice.hpp"
#include "hodlr/dyson.hpp"

class kpoint{
public:
    kpoint();
    ~kpoint();

    kpoint(int nt,
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
        double Ainitx);

    kpoint(int nt,
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
        h5e::File &checkpoint_file);
    
    void set_hk(int tstp, lattice_2d_ysymm &latt,double Ainitx);
    void get_Density_matrix(int tstp, hodlr::dlr_info &dlr);
    double step_dyson(int tstp, int SolverOrder, lattice_2d_ysymm &latt, Integration::Integrator &I, hodlr::dyson &dyson_sol, hodlr::dlr_info &dlr);

    double beta_;
    double dt_;
    double svdtol_;
    int nt_;
    int r_;
    int nlvl_;
    int size_;

    hodlr::function hk_;
    std::vector<hodlr::DMatrix> rho_;
    hodlr::herm_matrix_hodlr G_, Sigma_;
    
    double mu_; 
    Vector2D kk_;
};
