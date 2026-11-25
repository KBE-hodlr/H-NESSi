#!/bin/bash
# B24 (fixed in WP8): SC.x must refuse nt larger than the length of "phi" in phi_input.h5 (32768 in the shipped file)
# with a clear message, instead of reading past the end of phi.
# usage: sc_phi_length.sh SC_X WORKDIR INPUT PHI_H5
set -u
SC=$1; W=$2; IN=$3; PHI=$4
rm -rf "$W"; mkdir -p "$W"; cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
sed -e 's/^__nt=.*/__nt=  32769/' "$IN" > long.inp
"$SC" long.inp out.h5 "$PHI" > run.log 2>&1
rc=$?
cat run.log | grep -v "^GTL_DEBUG"
if grep -q "Sanitizer" run.log; then echo "sanitizer report"; exit 1; fi
[ $rc -ne 0 ] || { echo "SC.x accepted nt > length of phi"; exit 1; }
grep -q 'has 32768 entries, but nt = 32769' run.log || { echo "missing or wrong error message"; exit 1; }
[ ! -e out.h5 ] || { echo "output file written"; exit 1; }
