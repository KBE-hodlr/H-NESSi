#include <sys/stat.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>


// contour library headers
#include "h_nessi/read_inputfile.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/integration.hpp"
#include "SC_gf2.hpp"

void fill_t0_mat(h_nessi::function &phi_func, h_nessi::function &t0_func) {
  t0_func.set_zero();
  h_nessi::cplx I = h_nessi::cplx(0.,1.);
  for(int tstp = -1; tstp < t0_func.nt(); tstp++) {
    t0_func(tstp,0,0) = std::exp( I * phi_func(tstp,0,0));
    t0_func(tstp,1,1) = std::exp(-I * phi_func(tstp,0,0));
  }
}


//==============================================================================
//         main program
//==============================================================================
int main(int argc, char *argv[]){

  // Hodlr parameters
  int nlvl;
  double svdtol;
  int nt;
  double h;
  int SolverOrder;

  char* flin;
  flin=argv[1];
  find_param(flin,"__nlevel=",nlvl);
  find_param(flin,"__svdtol=",svdtol);
  find_param(flin,"__nt=",nt);
  find_param(flin,"__h=",h);
  find_param(flin,"__SolverOrder=",SolverOrder);

  // DLR parameters
  double epsdlr, lambda, MatsMaxErr, beta;
  int MatsMaxIter, r;

  find_param(flin,"__epsdlr=",epsdlr);
  find_param(flin,"__lambda=",lambda);
  find_param(flin,"__MatsubaraMaxErr=",MatsMaxErr);
  find_param(flin,"__MatsubaraMaxIter=",MatsMaxIter);
  find_param(flin,"__beta=",beta);
  
  // Timestepping Parameters
  int BootstrapMaxIter, CorrectorSteps;
  double BootstrapMaxErr, CorrectorMaxErr;
  double phi0, U;
  int size = 2, size2 = 4, xi = -1;

  find_param(flin,"__BootstrapMaxIter=",BootstrapMaxIter);
  find_param(flin,"__BootstrapMaxErr=",BootstrapMaxErr);
  find_param(flin,"__CorrectorSteps=",CorrectorSteps);
  find_param(flin,"__CorrectorMaxErr=",CorrectorMaxErr);
  find_param(flin,"__phi0=",phi0);
  find_param(flin,"__U=",U);

  h_nessi::cplx cplxi = h_nessi::cplx(0.,1.);

  // Dyson
  int rho_version = 1;
  h_nessi::dlr_info dlr(r, lambda, epsdlr, beta, size, xi);
  Integration::Integrator I(SolverOrder);
  h_nessi::dyson dyson_sol(nt, size, SolverOrder, dlr, rho_version);


  // Initialize Green's function storage
  h_nessi::herm_matrix_hodlr G(nt, r, nlvl, svdtol, size, size, xi, SolverOrder);
  h_nessi::herm_matrix_hodlr DeltaPlusSigma(nt, r, nlvl, svdtol, size, size, xi, SolverOrder);
  G.set_tstp_zero(-1);
  DeltaPlusSigma.set_tstp_zero(-1);

  // Self Energy evaluator
  h_nessi::SC_gf2 SC_gf2_solver(dlr);

  // Time dependent hamiltonian storage
  double mu = 0;
  h_nessi::function Hmf(nt, size, size);
  h_nessi::function phi(nt, 1, 1);
  h_nessi::function U_func(nt, 1, 1);
  h_nessi::function t0_func(nt, size, size);
  h_nessi::function current(nt, 1, 1);
  h_nessi::function energy_kin(nt, 1, 1);
  h_nessi::function energy_pot(nt, 1, 1);
  h_nessi::ZMatrix Jintres(2,2);

  // Read in phi, U, t0
  h5e::File phi_file(argv[3]);
  h_nessi::DColVector phi_vec = h5e::load<h_nessi::DColVector>(phi_file, "phi");
  for(int t = 0; t < nt; t++) phi(t,0,0) = phi0 * phi_vec(t);
  phi(-1,0,0) = phi(0,0,0);
  fill_t0_mat(phi, t0_func);
  U_func.set_constant(U);

  // Do Matsubara iterations
  bool matsubara_converged = false;
  double MatErr = 0;
  double eta = 0.0001;
  int eta_steps = 5;
  for(int iter = 0; iter < MatsMaxIter; iter++) {
    // Reset Self-Energy
    DeltaPlusSigma.set_tstp_zero(-1);

    // Evaluate Delta
    SC_gf2_solver.solve_Delta(-1, t0_func, G, DeltaPlusSigma);
    
    // Evaluate Sigma
    SC_gf2_solver.solve_Sigma(-1, U_func, G, DeltaPlusSigma);
    
    // Evaluate Hmf
    SC_gf2_solver.solve_Sigma_Fock(-1, U_func, G, Hmf);
    if(iter < eta_steps) {
      Hmf(-1,0,1) += eta;
      Hmf(-1,1,0) += eta;
    }

    // Solve for G
    MatErr = dyson_sol.dyson_mat(G, mu, Hmf, DeltaPlusSigma, false);
    std::cout << "Matsubara Iteration: " << iter << " Error: " << MatErr << std::endl;

    if(MatErr < MatsMaxErr && iter > eta_steps + 5) {
      matsubara_converged = true;
      std::cout << "Matsubara Iteration Converged after " << iter << " iterations.  Error: " << MatErr << std::endl;
      break;
    }
  }
  if(!matsubara_converged) {
    std::cout << "Matsubara Iteration did not converge!  Final error was " << MatErr << std::endl;
    exit(1);
  }

  // Matsubara is converged, we set the convolution tensor
  G.initGMConvTensor(dlr);

  // Set hmf for first k timesteps.  This is done as we assume TTI
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    Hmf.get_map(tstp) = Hmf.get_map(-1);
  }

  // Use hmf to calculate HF GF
  h_nessi::ZMatrix rhoIC(size, size);
  G.density_matrix(-1, dlr, rhoIC);
  dyson_sol.green_from_H_dm(G, mu, Hmf.get_map(-1), rhoIC, h, SolverOrder);

	// Self Consistency for bootstrapping
  bool bootstrap_converged = false;
  for(int iter = 0; iter < BootstrapMaxIter; iter++) {

    // Evaluate Self-Energy 
    // No need to do HF, as we assume TTI
    for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    	DeltaPlusSigma.set_tstp_zero(tstp);
      SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
      SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);
    }

    // Solve Dyson
    double err = dyson_sol.dyson_start(G, mu, Hmf, DeltaPlusSigma, I, h);

    std::cout << "Bootstrap iteration " << iter << " Error: " << err << std::endl;

    if(err < BootstrapMaxErr) {
      std::cout << "Bootstrap Converged" << std::endl;
      bootstrap_converged = true;
      break;
    }
  }

  if(!bootstrap_converged) {
    exit(1);
  }

  // For the current we need to convolve G with Delta_LmR
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta_LmR(tstp, t0_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, G, DeltaPlusSigma, h, I, Jintres.data());
    current(tstp,0,0) = Jintres(0,0) - Jintres(1,1);
  }

  // First evaluate delta then evaluate kinetic energy
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_kin(tstp,0,0) = Jintres(0,0) + Jintres(1,1);
  }

  // Next add in Sigma then evaluate potential energy
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_pot(tstp,0,0) = Jintres(0,0) + Jintres(1,1) - energy_kin(tstp,0,0);
  }


  // Here is timestepping
	for(int tstp = SolverOrder+1; tstp < nt; tstp++) {
		G.update_blocks(I);
    DeltaPlusSigma.update_blocks(I);

    dyson_sol.extrapolate(G, I);

    for(int iter = 0;  iter < CorrectorSteps; iter++) {
      // HF
      SC_gf2_solver.solve_Sigma_Fock(tstp, U_func, G, Hmf);

      // 2b
    	DeltaPlusSigma.set_tstp_zero(tstp);
      SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
      SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);

      // step
      double err = dyson_sol.dyson_timestep(tstp, G, mu, Hmf.ptr(0), DeltaPlusSigma, I, h);
      std::cout << "tstp " << tstp << "  iter " << iter << "  Err = " << err << std::endl;

      if(err < CorrectorMaxErr) break;
    }

    // For the current we need to convolve G with Delta_LmR
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta_LmR(tstp, t0_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, G, DeltaPlusSigma, h, I, Jintres.data());
    current(tstp,0,0) = Jintres(0,0) - Jintres(1,1);

    // First evaluate delta then evaluate kinetic energy
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_kin(tstp,0,0) = Jintres(0,0) + Jintres(1,1);

    // Next add in Sigma then evaluate potential energy
    SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_pot(tstp,0,0) = Jintres(0,0) + Jintres(1,1) - energy_kin(tstp,0,0);

    // Finally reevaluate HF component to be consistent with DeltaPlusSigma
    SC_gf2_solver.solve_Sigma_Fock(tstp, U_func, G, Hmf);
	}

  // Timestepping is done.  Update last of blocks
  for( int t = 0; t <= SolverOrder; t++) {
    G.update_blocks(I);
    DeltaPlusSigma.update_blocks(I);
  }
  
  h5e::File out_file(argv[2], h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);

  dyson_sol.write_timing(out_file);

  G.write_to_hdf5(out_file, "G/");
  G.write_rho_to_hdf5(out_file, "/rho/", dlr);
  G.write_GR0_to_hdf5(out_file, "G");
  G.write_GL0_to_hdf5(out_file, "G");
  G.write_rank_to_hdf5(out_file, "G");

  Hmf.write_to_hdf5(out_file, "hmf");
  current.write_to_hdf5(out_file, "current");
  energy_kin.write_to_hdf5(out_file, "energy_kin");
  energy_pot.write_to_hdf5(out_file, "energy_pot");
  phi.write_to_hdf5(out_file, "phi");
  t0_func.write_to_hdf5(out_file, "t0");

  return 0;
}
