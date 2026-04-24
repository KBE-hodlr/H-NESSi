#!/bin/bash
#SBATCH --job-name=dissKBEtest
#SBATCH --account=m5202
#SBATCH -q premium                       # perlmutter regular queue
#SBATCH -C cpu                # perlmutter CPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --time=08:00:00                # default (can be overridden via sbatch CLI)
#SBATCH --output=perlmutter_%j.out
#SBATCH --error=perlmutter_%j.err

export SLURM_CPU_BIND="cores"
srun /global/u1/t/tblommel/Libraries/H-NESSi/build/Nonint_diss_example/Nonint_diss_tests.x 2 /global/cfs/projectdirs/m5202/Thomas_nonint_diss_test/output.h5

