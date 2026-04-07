To run:
export OMP_NUM_THREADS=<nthreads>
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
srun -n <n_mpi> -c <nthreads> --cpu-bind=cores ./your_executable ./your_executable <input_file> <output_dir> <checkpoint_dir>

input_file (argv[1]): Path to your configuration file containing the simulation parameters.
output_dir (argv[2]): The directory where the simulation will write its output files (e.g., ground state data, tops). Must exist prior to execution.
checkpoint_dir (argv[3]): The directory used to read/write checkpoint data. Must exist prior to execution.