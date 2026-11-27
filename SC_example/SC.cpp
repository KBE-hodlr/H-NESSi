#include <sys/stat.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>


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

  if(argc < 4) {
    std::cerr << "usage: " << argv[0] << " input.inp output.h5 phi_input.h5" << std::endl;
    return 1;
  }
  char* flin;
  flin=argv[1];
  find_param(flin,"__nlevel=",nlvl);
  find_param(flin,"__svdtol=",svdtol);
  find_param(flin,"__nt=",nt);
  find_param(flin,"__h=",h);
  find_param(flin,"__SolverOrder=",SolverOrder);
  // The bootstrap fills t = 0..SolverOrder, and the efield stencil and the Gregory weights of order k read k+1 grid
  // points, so at least SolverOrder+1 time steps are needed.
  if(nt <= SolverOrder) {
    std::cerr << "SC.x: __nt= " << nt << " must be > __SolverOrder= " << SolverOrder << std::endl;
    return 1;
  }

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

  // Read the drive phi (argv[3]) before allocating anything large.
  // B24: phi(t) is read for t = 0..nt-1, so the file must hold at least nt entries
  h5e::File phi_file(argv[3]);
  h_nessi::DColVector phi_vec = h5e::load<h_nessi::DColVector>(phi_file, "phi");
  if(phi_vec.size() < nt) {
    std::cerr << "SC.x: the dataset \"phi\" in " << argv[3] << " has " << phi_vec.size()
              << " entries, but nt = " << nt << " needs at least nt entries (phi(t), t = 0..nt-1). "
              << "Reduce __nt= or provide a longer phi." << std::endl;
    return 1;
  }

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
  // Observables: complex 1x1 functions, t = -1..nt-1 (t = -1 is not computed and stays 0). Nambu traces (= sum over
  // both spins). The conventions were fixed by energy conservation (WP8): with them
  //   energy_total(t) - energy_total(0) = energy_deposited(t)
  // holds to the discretisation error of the solver (O(h^{k+1}); see the ctest ScExampleEnergyBalance).
  //  current(t)          j(t) = Im Tr[sigma_z (Gamma_L - Gamma_R)](t), Gamma_{R/L}(t) = -i[G * Delta_{L/R}]^<(t,t)
  //                      (thesis Eqs. 6.14-6.15 = paper Eqs. 18-19, with their crossed R/L labels, which are intended).
  //                      Delta_R = 1/2 t0 s G s t0^*, Delta_L = 1/2 t0^* s G s t0 (SC_gf2.cpp, paper Eq. 13), so
  //                      current = Tr[sigma_z (-i)[G*(Delta_R - Delta_L)]^<]; physical value = IMAGINARY part.
  //  efield(t)           E_P(t) = d phi(t)/dt (pump field; phi = A is the Peierls phase of t0, Eqs. 14-15; same sign as the
  //                      dataset "E" = d phi/dt in phi_input.h5). Degree-k polynomial derivative on the time grid.
  //  energy_deposited(t) E^j(t) = int_0^t E_P(s) j(s) ds (thesis Eq. 6.17), Gregory quadrature of order k.
  //  energy_hf(t)        (1/2) Tr[rho(t) Sigma^HF(t)] = (1/2) Tr[rho(t) Hmf(t)] (Galitskii-Migdal HF part, thesis Eq. 4.29
  //                      with h = 0 in the impurity; the diagonal Hartree term is cancelled by mu at half filling and
  //                      omitted, as in Hmf).
  //  energy_corr(t)      (1/2) Im Tr[Sigma_2B * G]^<(t,t) (Galitskii-Migdal correlation part, thesis Eq. 4.29 with
  //                      Sigma~ = Sigma_2B).
  //  energy_hyb(t)       Im Tr[Delta * G]^<(t,t) WITHOUT a factor 1/2: Delta is the lattice hopping (one-body), whose
  //                      energy Tr[rho h_kin] enters Eq. 4.29 fully (1/2 Tr[rho(h+eps)]). Paper Eq. 20 prints 1/2.
  //  energy_total(t)     energy_hf + energy_corr + energy_hyb.
  // For the energies the physical value is the REAL part (Re Tr[-i X]^< = Im Tr[X]^<); efield, energy_deposited are real.
  h_nessi::function current(nt, 1, 1);
  h_nessi::function efield(nt, 1, 1);
  h_nessi::function energy_deposited(nt, 1, 1);
  h_nessi::function energy_hf(nt, 1, 1);
  h_nessi::function energy_corr(nt, 1, 1);
  h_nessi::function energy_hyb(nt, 1, 1);
  h_nessi::function energy_total(nt, 1, 1);
  h_nessi::ZMatrix Jintres(2,2);

  // Fill phi (phi0 * phi_vec), U, t0
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
  double boot_last_err = 0;
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
    boot_last_err = err;

    if(err < BootstrapMaxErr) {
      std::cout << "Bootstrap Converged" << std::endl;
      bootstrap_converged = true;
      break;
    }
  }

  if(!bootstrap_converged) {
    std::cout << "Bootstrap did not converge in BootstrapMaxIter = " << BootstrapMaxIter << " iterations!  Final error was "
              << boot_last_err << " (BootstrapMaxErr = " << BootstrapMaxErr << ")" << std::endl;
    exit(1);
  }

  // Current (see the declaration of `current`): J = -i[G*(Delta_L - Delta_R)]^<, j = Im Tr[sigma_z (-J)]
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta_LmR(tstp, t0_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, G, DeltaPlusSigma, h, I, Jintres.data());
    current(tstp,0,0) = Jintres(1,1) - Jintres(0,0);
  }

  // Hybridization energy: Delta only, Tr[-i(Delta*G)^<] (no factor 1/2)
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_hyb(tstp,0,0) = Jintres(0,0) + Jintres(1,1);
  }

  // Next add in Sigma_2B: correlation energy (1/2) Tr[-i(Sigma_2B*G)^<] = (1/2)(Tr[-i((Delta+Sigma_2B)*G)^<] - energy_hyb)
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);
  }
  for(int tstp = 0; tstp <= SolverOrder; tstp++) {
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_corr(tstp,0,0) = 0.5 * (Jintres(0,0) + Jintres(1,1) - energy_hyb(tstp,0,0));
  }


  // Here is timestepping
	for(int tstp = SolverOrder+1; tstp < nt; tstp++) {
		G.update_blocks(I);
    DeltaPlusSigma.update_blocks(I);

    dyson_sol.extrapolate(G, I);

    bool corrector_converged = false;
    double corrector_last_err = 0;
    for(int iter = 0;  iter < CorrectorSteps; iter++) {
      // HF
      SC_gf2_solver.solve_Sigma_Fock(tstp, U_func, G, Hmf);

      // 2b
    	DeltaPlusSigma.set_tstp_zero(tstp);
      SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
      SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);

      // step
      double err = dyson_sol.dyson_timestep(tstp, G, mu, Hmf.ptr(0), DeltaPlusSigma, I, h);
      std::cout << "tstp " << tstp << "  iter " << iter << "  Err = " << err << '\n';   // P1: no flush per iteration
      corrector_last_err = err;

      if(err < CorrectorMaxErr) { corrector_converged = true; break; }
    }
    // E11 (WP9): the run continues with the last iterate, but say so
    if(!corrector_converged)
      std::cerr << "WARNING: corrector loop at tstp = " << tstp << " reached CorrectorSteps = " << CorrectorSteps
                << " without converging: last Err = " << corrector_last_err << " >= CorrectorMaxErr = " << CorrectorMaxErr
                << std::endl;

    // Current: J = -i[G*(Delta_L - Delta_R)]^<, j = Im Tr[sigma_z (-J)] (see the declaration of `current`)
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta_LmR(tstp, t0_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, G, DeltaPlusSigma, h, I, Jintres.data());
    current(tstp,0,0) = Jintres(1,1) - Jintres(0,0);

    // Hybridization energy Tr[-i(Delta*G)^<]
    DeltaPlusSigma.set_tstp_zero(tstp);
    SC_gf2_solver.solve_Delta(tstp, t0_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_hyb(tstp,0,0) = Jintres(0,0) + Jintres(1,1);

    // Next add in Sigma_2B: correlation energy (1/2) Tr[-i(Sigma_2B*G)^<]
    SC_gf2_solver.solve_Sigma(tstp, U_func, G, DeltaPlusSigma);
    dyson_sol.gamma_integral(tstp, DeltaPlusSigma, G, h, I, Jintres.data());
    energy_corr(tstp,0,0) = 0.5 * (Jintres(0,0) + Jintres(1,1) - energy_hyb(tstp,0,0));

    // Finally reevaluate HF component to be consistent with DeltaPlusSigma
    SC_gf2_solver.solve_Sigma_Fock(tstp, U_func, G, Hmf);
	}

  // Timestepping is done.  Update last of blocks
  for( int t = 0; t <= SolverOrder; t++) {
    G.update_blocks(I);
    DeltaPlusSigma.update_blocks(I);
  }

  // HF energy (1/2) Tr[rho(t) Hmf(t)] and the total energy
  {
    h_nessi::ZMatrix rho_t(size, size);
    for(int tstp = 0; tstp < nt; tstp++) {
      G.density_matrix(tstp, dlr, rho_t);
      energy_hf(tstp,0,0) = 0.5 * (rho_t * Hmf.get_map(tstp)).trace();
      energy_total(tstp,0,0) = energy_hf(tstp,0,0) + energy_corr(tstp,0,0) + energy_hyb(tstp,0,0);
    }
  }
  // Pump field E_P = d phi/dt: derivative of the degree-k interpolating polynomial through k+1 grid points around t
  // (centred where possible). Deposited energy E^j(t) = int_0^t E_P j: Gregory weights of order k.
  {
    const int k = SolverOrder;
    for(int tstp = 0; tstp < nt; tstp++) {
      int start = std::min(std::max(tstp - k/2, 0), nt - 1 - k);
      double d = 0;
      for(int l = 0; l <= k; l++) d += I.poly_diff(tstp - start, l) * phi(start + l, 0, 0).real();
      efield(tstp,0,0) = d / h;
    }
    for(int tstp = 0; tstp < nt; tstp++) {
      double e = 0;
      for(int j = 0; j <= std::max(tstp, k); j++)
        e += I.gregory_weights(tstp, j) * efield(j,0,0).real() * current(j,0,0).imag();
      energy_deposited(tstp,0,0) = h * e;
    }
  }
  
  h5e::File out_file(argv[2], h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);

  dyson_sol.write_timing(out_file);

  G.write_to_hdf5(out_file, "G/");
  G.write_rho_to_hdf5(out_file, "/rho/", dlr);
  G.write_GR0_to_hdf5(out_file, "G");
  G.write_GL0_to_hdf5(out_file, "G");
  G.write_rank_to_hdf5(out_file, "G");

  Hmf.write_to_hdf5(out_file, "hmf");
  // observables, each with a "description" attribute giving its formula
  auto write_obs = [&](h_nessi::function &f, const std::string &name, const std::string &desc) {
    f.write_to_hdf5(out_file, name);
    out_file.getGroup(name).createAttribute<std::string>("description", desc);
  };
  write_obs(current, "current",
    "j(t) = Im Tr[sigma_z (Gamma_L - Gamma_R)](t), Gamma_{R/L} = -i[G*Delta_{L/R}]^<(t,t) (thesis Eqs. 6.14-6.15, paper "
    "Eqs. 18-19, crossed labels intended); stored Tr[sigma_z (-i)[G*(Delta_R-Delta_L)]^<], physical value = imaginary part");
  write_obs(efield, "efield",
    "pump field E_P(t) = d phi/dt (phi = Peierls phase A of t0, paper Eqs. 14-15); degree-k polynomial derivative; real");
  write_obs(energy_deposited, "energy_deposited",
    "E^j(t) = int_0^t E_P j (thesis Eq. 6.17), j = Im(current); Gregory quadrature of order k; real");
  write_obs(energy_hf, "energy_hf",
    "(1/2) Tr[rho(t) Sigma^HF(t)] = (1/2) Tr[rho Hmf] (Galitskii-Migdal, thesis Eq. 4.29; Hartree diagonal omitted, "
    "cancelled by mu at half filling); Nambu trace; physical value = real part");
  write_obs(energy_corr, "energy_corr",
    "(1/2) Im Tr[Sigma_2B*G]^<(t,t) = (1/2) Re Tr[(-i)[Sigma_2B*G]^<] (Galitskii-Migdal, thesis Eq. 4.29); Nambu trace; "
    "physical value = real part");
  write_obs(energy_hyb, "energy_hyb",
    "Im Tr[Delta*G]^<(t,t) = Re Tr[(-i)[Delta*G]^<] (lattice hopping energy, no factor 1/2; paper Eq. 20 prints 1/2); "
    "Nambu trace; physical value = real part");
  write_obs(energy_total, "energy_total",
    "energy_hf + energy_corr + energy_hyb; energy_total(t) - energy_total(0) = energy_deposited(t) up to the "
    "discretisation error; physical value = real part");
  phi.write_to_hdf5(out_file, "phi");
  t0_func.write_to_hdf5(out_file, "t0");

  return 0;
}
