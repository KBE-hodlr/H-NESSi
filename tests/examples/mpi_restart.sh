#!/bin/bash
# B16 (example): mpi_example must resume from a checkpoint and reproduce the uninterrupted run.
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
"$MAIN" restart.inp rst/out/ rst/chk/ > restart.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "restart run failed with exit code $rc"; tail -3 restart.log; exit 1; fi
"$H5FLAT" cmp full/out/obs.h5 rst/out/obs.h5 1e-10 /jxtnorm /jytnorm /Ekint /Navgt
