To run:
export OMP_NUM_THREADS=<nthreads>
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
srun -n <n_mpi> -c <nthreads> --cpu-bind=cores ./your_executable ./your_executable <input_file> <output_dir> <checkpoint_dir>

input_file (argv[1]): Path to your configuration file containing the simulation parameters.
output_dir (argv[2]): The directory where the simulation will write its output files (e.g., ground state data, tops). Must exist prior to execution.
checkpoint_dir (argv[3]): The directory used to read/write checkpoint data. Must exist prior to execution.

Observables (output_dir/obs.h5, Nt doubles each; jxt_iter has max(BootMaxIter,StepMaxIter)+1 rows):
  jxtnorm, jytnorm  <j_x>(t)/Amax, <j_y>(t)/Amax (spin summed, Eq. 26). The vector potential is A(t) = +Amax (x) for
                    t >= 0 (lattice.cpp), whereas paper Eq. 29 uses A(t) = -Amax theta(t); hence jxtnorm = -sigma(t) of
                    paper Eq. 30.
  Ekint             kinetic energy with the bare eps_k, spin summed (Eq. 27).
  Navgt             density per spin (Eq. 28).
  jxt_iter          jxtnorm after each bootstrap (t <= 5) or corrector (t > 5) iteration.
Energy bookkeeping (WP8b; Galitskii-Migdal conventions of D-7, as in SC_example; per site, spin summed; each dataset
has a "description" attribute):
  current_x         <j_x>(t) (Eq. 26, = Amax * jxtnorm), v = 2J sin(k_x - A_x(t) - Ainitx).
  energy_band       (2/N_k) sum_k eps_{k-A(t)} rho_k(t), eps_k = -2J(cos kx + cos ky), no mu (one-body: factor 1).
                    Differs from Ekint, which uses the bare eps_k (no A(t)) and includes -mu.
  energy_hf         (1/2) Tr[rho Sigma^H] = U Navgt^2 (Hartree shift absorbed in __MuChem = mu - U<n_s>, paper Eq. 25).
  energy_corr       (1/2) (2/N_k) sum_k Im[Sigma_2B,k * G_k]^<(t,t) (dyson::gamma_integral per k-point).
  energy_total      energy_band + energy_hf + energy_corr (= <H> + mu N of paper Eq. 25; mu N is conserved).
  efield            E_x(t) = -dA_x/dt (Eq. 29).
  energy_deposited  int_0^t efield * current_x dt'. energy_total(t) - energy_total(0) = energy_deposited(t) up to the
                    discretisation error (ctest MpiExampleEnergyBalance). For the default step the kick lies between
                    t = -1 (equilibrium) and t = 0, so efield = energy_deposited = 0 and energy_total is constant.

Vector potential (optional input keys; without them the default step is used):
  __pulse=  step     A_x(t) = +Amax for t >= 0 (default).
  __pulse=  smooth   A_x(t) = Amax sin^2(pi (t - t0)/T) sin(omega (t - t0)) for t0 < t < t0 + T, 0 otherwise, with
  __pulse_tstart= t0, __pulse_T= T, __pulse_omega= omega (omega = 0: A_x = Amax sin^2). Choose omega*T a multiple of
                     pi so that A is smooth (C^2) at the pulse end; jxtnorm is still <j_x>/Amax.

Checkpoint/restart: when __time_limit= (seconds) or __memory_limit= is reached, the run writes
checkpoint_dir/GSigma<k>.h5 and output_dir/obs_checkpoint.h5 and stops. Rerun with __checkpoint= true and the same
output_dir and checkpoint_dir to continue from the next time step.
