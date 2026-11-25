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

Checkpoint/restart: when __time_limit= (seconds) or __memory_limit= is reached, the run writes
checkpoint_dir/GSigma<k>.h5 and output_dir/obs_checkpoint.h5 and stops. Rerun with __checkpoint= true and the same
output_dir and checkpoint_dir to continue from the next time step.
