#!/bin/bash
cd ~/veins_workspace/cmdp_pkg_v2/simulations/manhattan
export PATH=$HOME/omnetpp-5.6.2/bin:$PATH
export LD_LIBRARY_PATH=$HOME/veins_workspace/cmdp_pkg_v2:$HOME/omnetpp-5.6.2/lib:$HOME/veins-veins-5.2/out/gcc-release/src:$LD_LIBRARY_PATH
ulimit -n 8192

run_sim() {
    echo "=== $1 START $(date) ==="
    fuser -k 9999/tcp 2>/dev/null; killall -9 sumo 2>/dev/null; sleep 2
    rm -f cmdp_log.csv cmdp_log_delivery.csv
    python3 ~/veins-veins-5.2/bin/veins_launchd -vv -c sumo &
    sleep 3
    opp_run -u Cmdenv -c "$1" -r 0 \
      -n "../../src:.:/home/wei/veins_ned_root" \
      -l ../../libcmdp_release.so \
      -l /home/wei/veins-veins-5.2/out/gcc-release/src/libveins.so \
      --sim-time-limit=50s omnetpp.ini 2>&1 | tail -3
    cp cmdp_log.csv ~/E3_results/E3_${1}.csv 2>/dev/null
    python3 -c "
import csv
rows=[r for r in csv.DictReader(open('$HOME/E3_results/E3_${1}.csv')) if r.get('simTime','')!='simTime']
t=len(rows); r=sum(1 for x in rows if x.get('reachable','0')=='1')
fe=sum(1 for x in rows if x.get('feasible','0')=='1')
print(f'  {t}q Reach={100*r/t:.1f}% Feas={100*fe/t:.1f}%')
" 2>/dev/null || echo "  ERROR"
    echo "=== $1 DONE $(date) ==="
}

echo "===== E3-SLIM START $(date) ====="

# Strict: p005 p020 p030 (p003已有)
for LBL in p005 p020 p030; do
    run_sim "E3-Strict-${LBL}"
done

# PPOPIP: p020 p030 (p003 p005已有)
for LBL in p020 p030; do
    run_sim "E3-PPOPIP-${LBL}"
done

# 杀 GNN/PPO, 跑 LS
fuser -k 5555/tcp 2>/dev/null; fuser -k 5556/tcp 2>/dev/null
for LBL in p003 p005 p020 p030; do
    run_sim "E3-LS-${LBL}"
done

echo "===== E3-SLIM DONE $(date) ====="
