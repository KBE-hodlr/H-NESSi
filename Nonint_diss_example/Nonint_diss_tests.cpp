#include <sys/stat.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>
#include <random>

// contour library headers
#include "h_nessi/read_inputfile.hpp"
#include "h_nessi/herm_matrix_hodlr.hpp"
#include "h_nessi/dyson.hpp"
#include "h_nessi/integration.hpp"

using namespace h_nessi;

void fillRandomHermitian(ZMatrix& mat, double min_eig, double max_eig, unsigned int seed) {
    // 1. Validate the matrix and bounds
    if (mat.rows() != mat.cols()) {
        throw std::invalid_argument("Input ZMatrix must be square.");
    }
    if (min_eig > max_eig) {
        throw std::invalid_argument("min_eig cannot be greater than max_eig.");
    }
    
    int N = mat.rows();
    if (N == 0) return; 

    // 2. Initialize Random Number Generators
    std::mt19937 gen(seed);
    
    // Distribution for eigenvalues uses the custom bounds
    std::uniform_real_distribution<double> eig_dist(min_eig, max_eig);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    // 3. Generate random real eigenvalues within [min_eig, max_eig]
    Eigen::VectorXcd D(N);
    for (int i = 0; i < N; ++i) {
        D(i) = std::complex<double>(eig_dist(gen), 0.0);
    }

    // 4. Generate a random unitary matrix Q via QR decomposition
    ZMatrix X(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            X(i, j) = std::complex<double>(norm_dist(gen), norm_dist(gen));
        }
    }
    
    Eigen::HouseholderQR<ZMatrix> qr(X);
    ZMatrix Q = qr.householderQ();

    // 5. Construct the final Hermitian matrix: H = Q * D * Q^dagger
    mat = Q * D.asDiagonal() * Q.adjoint();
    
    // Force perfect Hermiticity to eliminate microscopic floating-point errors
    mat = 0.5 * (mat + mat.adjoint()).eval();
}



int main(int argc, char *argv[]){

  // Hodlr parameters
  int t_final = std::stoi(argv[1]);

  // Timestepping Parameters
  int size = 5, size2 = size*size, xi = -1;

  h_nessi::cplx cplxi = h_nessi::cplx(0.,1.);

  ZMatrix H_mat(size,size);
  ZMatrix ellG_mat(size,size);
  ZMatrix ellL_mat(size,size);
  ZMatrix rho0_mat(size,size);

  fillRandomHermitian(H_mat, -2, 2, 0);
  fillRandomHermitian(ellG_mat, 0, 3, 1);
  fillRandomHermitian(ellL_mat, 0, 3, 2);
  fillRandomHermitian(rho0_mat, 0, 1, 3);

  int r;
  dlr_info dlr(r, 1, 0.1, 1, size, xi);
  std::cout << "r" << r << std::endl;
  double mu = 0;

  h5e::File out_file(argv[2], h5e::File::Overwrite | h5e::File::ReadWrite | h5e::File::Create);
  h5e::dump<ZMatrix>(out_file, "H_mat", H_mat);
  h5e::dump<ZMatrix>(out_file, "ellG_mat", ellG_mat);
  h5e::dump<ZMatrix>(out_file, "ellL_mat", ellL_mat);
  h5e::dump<ZMatrix>(out_file, "rho0_mat", rho0_mat);

  for(int k = 1; k <= 5; k++) {
    for(int hinv = 10; hinv <= 160; hinv*=2) {
      for(int rv = 0; rv <= 1; rv++) {
        for(int Lb = 0; Lb <= 1; Lb++) {
          for(int Gb = 0; Gb <= 1; Gb++) {

            int nt = t_final * hinv + 1;
            int nlvl = std::max(0, static_cast<int>(std::round(std::log2(nt) - 3.0)));

            Integration::Integrator I(k);
            dyson dyson_sol(nt, size, k, dlr, rv);

            function H(nt, size, size);
            function ellG(nt, size, size);
            function ellL(nt, size, size);
            function rho(nt, size, size);

            H.set_constant(H_mat);
            ellG.set_constant(Gb*ellG_mat);
            ellL.set_constant(Lb*ellL_mat);

            herm_matrix_hodlr G(nt, r, nlvl, 1e-12, size, size, xi, k);
            herm_matrix_hodlr Sigma(nt, r, nlvl, 1e-12, size, size, xi, k);

            // Initial condition
            // \rho = \xi i G^<
            G.set_les_curr(0,0,xi * -1. * cplxi * rho0_mat);

            for(int tstp = 0; tstp <= k; tstp++) Sigma.set_tstp_zero(tstp);

            dyson_sol.dyson_start_2leg_diss(G, mu, H, ellL, ellG, Sigma, I, 1./hinv);

            for(int tstp = k+1; tstp < nt; tstp++) {
              std::cout << k << " " << hinv << " " << rv << " " << Lb << " " << Gb << " " << tstp << std::endl;
              G.update_blocks(I);
              Sigma.update_blocks(I);
              Sigma.set_tstp_zero(tstp);
              dyson_sol.extrapolate_2leg(G,I);
              dyson_sol.dyson_timestep_2leg_diss(tstp, G, mu, H, ellL, ellG, Sigma, I, 1./hinv);
            }

            for(int i = 0; i <= k; i++) {
              std::cout << "update " << i << std::endl;
              G.update_blocks(I);
              Sigma.update_blocks(I);
            }

            ZMatrix tmp_rho(size, size);
            for(int i = 0; i < nt; i++) {
              std::cout << "dm " << i << std::endl;
              G.density_matrix(i, dlr, tmp_rho);
              rho.get_map(i) = tmp_rho;
            }

            std::string prelabel = "/k" + std::to_string(k) + "_h" + std::to_string(hinv) + "_rv" + std::to_string(rv) + "_L" + std::to_string(Lb) + "_G" + std::to_string(Gb) + "/";
            rho.write_to_hdf5(out_file, prelabel + "/rho");
            G.write_to_hdf5(out_file, prelabel + "/G");

          }
        }
      }
    }
  }
}
