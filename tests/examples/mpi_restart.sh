#!/bin/bash
# B16 (fixed in WP8): mpi_example must resume from a checkpoint and reproduce the uninterrupted run bitwise.
# A run with __time_limit=0 stops after its first time step and writes the checkpoint; the restart run
# (__checkpoint=true, same output/checkpoint dirs) continues to Nt. obs.h5 must equal the uninterrupted run exactly
# (since WP8b also the energy datasets, which obs_checkpoint.h5 carries as current_x, energy_band, energy_corr).
# usage: mpi_restart.sh MAIN_X H5FLAT WORKDIR INPUT
set -u
MAIN=$1; H5FLAT=$2; W=$3; IN=$4
rm -rf "$W"; mkdir -p "$W"/full/out "$W"/full/chk "$W"/rst/out "$W"/rst/chk
cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
sed -e 's/^__checkpoint=.*/__checkpoint=  false/' -e 's/^__time_limit=.*/__time_limit=  3600/' "$IN" > full.inp
sed -e 's/^__checkpoint=.*/__checkpoint=  false/' -e 's/^__time_limit=.*/__time_limit=  0/' "$IN" > stop.inp
sed -e 's/^__checkpoint=.*/__checkpoint=  true/' -e 's/^__time_limit=.*/__time_limit=  3600/' "$IN" > restart.inp
"$MAIN" full.inp full/out/ full/chk/ > full.log 2>&1 || { echo "uninterrupted run failed"; tail -5 full.log; exit 3; }
"$MAIN" stop.inp rst/out/ rst/chk/ > stop.log 2>&1 || { echo "stop run failed"; tail -5 stop.log; exit 3; }
ls rst/chk/GSigma0.h5 > /dev/null 2>&1 || { echo "no checkpoint written by the stop run"; exit 3; }
grep -q "REACHED THE TIME LIMIT" stop.log || { echo "stop run did not stop at the time limit"; exit 3; }
"$MAIN" restart.inp rst/out/ rst/chk/ > restart.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "restart run failed with exit code $rc"; tail -3 restart.log; exit 1; fi
if grep -q "Sanitizer" full.log stop.log restart.log; then echo "sanitizer report"; grep -h -A3 "ERROR: .*Sanitizer" *.log | head -20; exit 1; fi
"$H5FLAT" cmp full/out/obs.h5 rst/out/obs.h5 0 /jxtnorm /jytnorm /Ekint /Navgt /jxt_iter \
  /current_x /energy_band /energy_hf /energy_corr /energy_total /efield /energy_deposited
