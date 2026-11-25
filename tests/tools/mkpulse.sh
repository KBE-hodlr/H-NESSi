#!/bin/bash
# Generates tests/data/sc_pulse.h5: bash mkpulse.sh sc_pulse.h5 0.03 300   (needs h5import)
# usage: mkpulse.sh OUT.h5 H N [A0 OMEGA T1 T]   phi[n] = A0 sin^2(pi (t-T1)/T) sin(OMEGA (t-T1)) for T1<t<T1+T, t=n*H
OUT=$1; H=$2; N=$3; A0=${4:-0.5}; OM=${5:-3.141592653589793}; T1=${6:-0.6}; TT=${7:-6.0}
tmp=$(mktemp -d)
awk -v h=$H -v n=$N -v a=$A0 -v om=$OM -v t1=$T1 -v tt=$TT 'BEGIN{pi=atan2(0,-1); for(i=0;i<n;i++){t=i*h; v=0; if(t>t1 && t<t1+tt){s=sin(pi*(t-t1)/tt); v=a*s*s*sin(om*(t-t1))}; printf "%.17g\n", v}}' > $tmp/phi.txt
cat > $tmp/phi.cfg <<CFG
PATH phi
INPUT-CLASS TEXTFP
INPUT-SIZE 64
RANK 1
DIMENSION-SIZES $N
OUTPUT-CLASS FP
OUTPUT-SIZE 64
OUTPUT-ARCHITECTURE IEEE
OUTPUT-BYTE-ORDER LE
CFG
rm -f "$OUT"; h5import $tmp/phi.txt -c $tmp/phi.cfg -o "$OUT"; rm -rf $tmp
