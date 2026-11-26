#!/bin/bash
# mpi_example energy balance (WP8b, conventions of Q10/D-7): for the contained pulse of tests/data/mpi_pulse.inp
# (__pulse= smooth: A_x(t) = 0.8 sin^2(pi(t-0.2)/4) sin(pi(t-0.2)) for 0.2 < t < 4.2; L = 4, h = 0.04, Nt = 151)
#   energy_total(t) - energy_total(0) - energy_deposited(t) = 0,  energy_total = energy_band + energy_hf + energy_corr,
# up to the discretisation error (3.5e-7 here; 9e-9 at h = 0.02; max|E^j| = 0.0285). Any other factor of the band or
# correlation term (or the opposite sign of E.j) gives >= 5e-4. Tolerance 1e-5.
# usage: mpi_energy_balance.sh MAIN_X H5FLAT WORKDIR INPUT
set -u
MAIN=$1; H5FLAT=$2; W=$3; IN=$4
rm -rf "$W"; mkdir -p "$W"/out "$W"/chk; cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$MAIN" "$IN" out/ chk/ > run.log 2>&1
rc=$?
if grep -q "Sanitizer" run.log; then echo "sanitizer report"; grep -A6 "ERROR: .*Sanitizer" run.log | head -20; exit 1; fi
[ $rc -eq 0 ] || { echo "Main.x failed with exit code $rc"; tail -5 run.log; exit 1; }
grep -q "pulse = smooth" run.log || { echo "input did not select the smooth pulse"; exit 1; }
"$H5FLAT" dump out/obs.h5 > dump.txt
# values of a real 1D dataset of obs.h5
vals() { awk -v ds="$1" '$1=="DATASET"{on=($2==ds); next} on{print $1}' dump.txt; }
vals /energy_total > et.txt; vals /energy_deposited > ej.txt
[ -s et.txt ] && [ -s ej.txt ] || { echo "energy datasets missing"; exit 1; }
paste et.txt ej.txt | awk 'NR==1{e0=$1} {r=$1-e0-$2; if(r<0)r=-r; if(r>m)m=r; a=($2<0)?-$2:$2; if(a>ejm)ejm=a}
  END{printf "max|E_total(t)-E_total(0)-E^j(t)| = %.3e   max|E^j| = %.6f\n", m, ejm;
      if(!(m < 1e-5)) {print "energy balance violated"; exit 1}
      if(!(ejm > 0.01)) {print "deposited energy too small: pulse not seen"; exit 1}}'
