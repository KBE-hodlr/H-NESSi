To run this example, the program should be called as
/path/to/SC.x input.inp /path/to/output.h5 phi_input.h5

phi_input.h5 must contain a dataset "phi" with at least nt entries (the shipped file has 32768); the drive is
phi(t) = __phi0 * phi[t], t = 0..nt-1. The program stops with an error message if the file is too short.

Observables in output.h5 (complex 1x1 functions "<name>/data", shape {nt+1,1,1}, t = -1..nt-1; t = -1 is not computed).
All traces are Nambu traces (sum over both spins). The energy conventions are fixed by energy conservation:
energy_total(t) - energy_total(0) = energy_deposited(t) up to the discretisation error (checked by the ctest
ScExampleEnergyBalance).
  current           j(t) = Im Tr[sigma_z (Gamma_L - Gamma_R)](t), Gamma_{R/L} = -i[G*Delta_{L/R}]^<(t,t) (paper Eqs. 18-19;
                    the crossed R/L labels are intended). Physical value = IMAGINARY part.
  efield            pump field E_P(t) = d phi/dt (real; phi = __phi0 * phi[t] is the Peierls phase A(t)).
  energy_deposited  E^j(t) = int_0^t E_P(s) j(s) ds (real).
  energy_hf         (1/2) Tr[rho(t) Sigma^HF(t)]                    } Galitskii-Migdal interaction energy
  energy_corr       (1/2) Im Tr[Sigma_2B * G]^<(t,t)                 }
  energy_hyb        Im Tr[Delta * G]^<(t,t) (no factor 1/2: lattice hopping energy; paper Eq. 20 prints a 1/2)
  energy_total      energy_hf + energy_corr + energy_hyb (the constant Hartree/mu term at half filling is omitted)
For the energies the physical value is the REAL part. Each group carries a "description" attribute with its formula.
(Earlier versions wrote "current" with the opposite sign, "energy_kin" = energy_hyb and "energy_pot" = 2*energy_corr.)
