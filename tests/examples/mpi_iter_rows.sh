#!/bin/bash
# B17 (fixed in WP8): jxt_iter must have max(BootMaxIter,StepMaxIter)+1 rows. The input forces the corrector to run all
# StepMaxIter+1 iterations (StepMaxErr=0) with StepMaxIter > BootMaxIter; before the fix this wrote past jxt_iter_local
# (AddressSanitizer: heap-buffer-overflow at Main.cpp, jxt_iter_local[iter][tstp]).
# usage: mpi_iter_rows.sh MAIN_X H5FLAT WORKDIR INPUT
set -u
MAIN=$1; H5FLAT=$2; W=$3; IN=$4
rm -rf "$W"; mkdir -p "$W"/out "$W"/chk
cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
sed -e 's/^__BootMaxIter=.*/__BootMaxIter=  2/' -e 's/^__StepMaxIter=.*/__StepMaxIter=  25/' \
    -e 's/^__StepMaxErr=.*/__StepMaxErr=  0/' -e 's/^__Nt=.*/__Nt=  12/' -e 's/^__write_GS=.*/__write_GS=  false/' \
    -e 's/^__write_tops=.*/__write_tops=  false/' "$IN" > iter.inp
"$MAIN" iter.inp out/ chk/ > run.log 2>&1
rc=$?
if grep -q "Sanitizer" run.log; then echo "sanitizer report"; grep -A6 "ERROR: .*Sanitizer" run.log | head -20; exit 1; fi
if [ $rc -ne 0 ]; then echo "run failed with exit code $rc"; tail -5 run.log; exit 1; fi
hdr=$("$H5FLAT" dump out/obs.h5 | grep '^DATASET /jxt_iter ')
echo "$hdr"
[ "$(echo "$hdr" | awk '{print $5}')" = "26x12" ] || { echo "jxt_iter has the wrong shape (expected 26x12)"; exit 1; }
