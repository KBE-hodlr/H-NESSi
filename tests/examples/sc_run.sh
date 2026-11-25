#!/bin/bash
# SC_example smoke run on a tiny input (used as a sanitizer run in the ASan build): exit 0, no sanitizer report,
# observables written with their description attributes.
# usage: sc_run.sh SC_X H5FLAT WORKDIR INPUT PHI_H5
set -u
SC=$1; H5FLAT=$2; W=$3; IN=$4; PHI=$5
rm -rf "$W"; mkdir -p "$W"; cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$SC" "$IN" out.h5 "$PHI" > run.log 2>&1
rc=$?
if grep -q "Sanitizer" run.log; then echo "sanitizer report"; grep -A6 "ERROR: .*Sanitizer" run.log | head -20; exit 1; fi
[ $rc -eq 0 ] || { echo "SC.x failed with exit code $rc"; tail -5 run.log; exit 1; }
"$H5FLAT" dump out.h5 | grep -q "^DATASET /energy_total/data " || { echo "energy_total/data missing"; exit 1; }
