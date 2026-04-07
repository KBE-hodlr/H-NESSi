#!/bin/bash
#SBATCH --job-name=mpi_hodlr
#SBATCH --account=m5202
#SBATCH -q premium                       # perlmutter regular queue
#SBATCH -C cpu                # perlmutter CPU partition
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128            # use a full CPU socket/thread budget
#SBATCH --time=00:05:00                # default (can be overridden via sbatch CLI)
#SBATCH --output=perlmutter_%j.out
#SBATCH --error=perlmutter_%j.err

# ----------------------------------------
# 1) Threads tuning (Eigen + OpenMP)
# ----------------------------------------
export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

srun -n 2 -c 128 --cpu-bind=cores /global/homes/t/tblommel/Libraries/H-NESSi/build/mpi_example/Main.x /global/homes/t/tblommel/Libraries/H-NESSi/mpi_example/input.inp /global/homes/t/tblommel/Libraries/H-NESSi/mpi_example/ /global/homes/t/tblommel/Libraries/H-NESSi/mpi_example/
