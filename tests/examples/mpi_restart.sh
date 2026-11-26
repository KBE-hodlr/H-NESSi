#!/bin/bash
# B16 (fixed in WP8): mpi_example must resume from a checkpoint and reproduce the uninterrupted run bitwise.
# The run is split at the given stop points: "first" = __time_limit=0 (stops after its first time step), a number N =
# __stop_tstp= N (WP9: checkpoint and stop after time step N, deterministic). The first piece starts fresh
# (__checkpoint=false); every later piece restarts from the checkpoint of the previous one (__checkpoint=true, same
# output/checkpoint dirs), so several stop points test chained restarts. The last piece runs to Nt.
# obs.h5 must equal the uninterrupted run exactly (since WP8b also the energy datasets, which obs_checkpoint.h5 carries
# as current_x, energy_band, energy_corr). Since WP9 timings_file.h5 is also restored for t < t0: every per-step timing
# row t > SolverOrder must be nonzero after the last restart.
# B35 (WP9): with MPI_LAUNCH set (e.g. "srun -n 6") all runs use that launcher, so a lattice with fewer k-points than
# ranks tests restarts on ranks without k-points.
# usage: mpi_restart.sh MAIN_X H5FLAT WORKDIR INPUT [STOP ...]     (default STOP: first)
set -u
MAIN=$1; H5FLAT=$2; W=$3; IN=$4; shift 4
STOPS=("$@"); [ ${#STOPS[@]} -eq 0 ] && STOPS=(first)
LAUNCH=${MPI_LAUNCH:-}
rm -rf "$W"; mkdir -p "$W"/full/out "$W"/full/chk "$W"/rst/out "$W"/rst/chk
cd "$W"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# mkinp CHECKPOINT TIME_LIMIT STOP_TSTP OUT
mkinp() {
  grep -v '^__stop_tstp=' "$IN" | sed -e "s/^__checkpoint=.*/__checkpoint=  $1/" -e "s/^__time_limit=.*/__time_limit=  $2/" > "$4"
  [ "$3" != "-1" ] && echo "__stop_tstp=  $3" >> "$4"
  return 0
}
mkinp false 3600 -1 full.inp
$LAUNCH "$MAIN" full.inp full/out/ full/chk/ > full.log 2>&1 || { echo "uninterrupted run failed"; tail -5 full.log; exit 3; }
logs="full.log"
ck=false
i=0
for s in "${STOPS[@]}"; do
  if [ "$s" = first ]; then mkinp $ck 0 -1 piece$i.inp; msg="REACHED THE TIME LIMIT"
  else mkinp $ck 3600 "$s" piece$i.inp; msg="REACHED THE STOP TIME STEP"; fi
  $LAUNCH "$MAIN" piece$i.inp rst/out/ rst/chk/ > piece$i.log 2>&1 || { echo "piece $i (stop $s) failed"; tail -5 piece$i.log; exit 3; }
  logs="$logs piece$i.log"
  ls rst/chk/GSigma0.h5 > /dev/null 2>&1 || { echo "no checkpoint written by piece $i"; exit 3; }
  grep -q "$msg" piece$i.log || { echo "piece $i did not stop ($msg)"; exit 3; }
  ck=true; i=$((i+1))
done
mkinp true 3600 -1 restart.inp
$LAUNCH "$MAIN" restart.inp rst/out/ rst/chk/ > restart.log 2>&1
rc=$?
logs="$logs restart.log"
if [ $rc -ne 0 ]; then echo "restart run failed with exit code $rc"; tail -3 restart.log; exit 1; fi
if grep -q "Sanitizer" $logs; then echo "sanitizer report"; grep -h -A3 "ERROR: .*Sanitizer" $logs | head -20; exit 1; fi
if grep -q "WARNING: obs_checkpoint.h5" $logs; then echo "unexpected checkpoint warning"; grep -h "WARNING: obs_checkpoint" $logs; exit 1; fi
"$H5FLAT" cmp full/out/obs.h5 rst/out/obs.h5 0 /jxtnorm /jytnorm /Ekint /Navgt /jxt_iter \
  /current_x /energy_band /energy_hf /energy_corr /energy_total /efield /energy_deposited || exit 1
# timings restored: no zero row for t > SolverOrder (= 5) in the restarted timings_file.h5
"$H5FLAT" dump rst/out/timings_file.h5 > timings.txt
awk '$1=="DATASET"{ds=$2; t=-1; next} {t++; if(t>5 && $1+0==0){print "zero timing " ds " t=" t; z++}}
     END{if(z){exit 1}}' timings.txt > timings_check.txt
rc=$?
head -5 timings_check.txt
[ $rc -eq 0 ] || { echo "timings_file.h5 not restored for t < t0"; exit 1; }
echo "restart (stops: ${STOPS[*]}${LAUNCH:+; launcher: $LAUNCH}) reproduces the uninterrupted run bitwise"
