#!/bin/bash
cd ~/veins_workspace/cmdp_pkg_v2/simulations/manhattan
ulimit -n 8192
export PATH="/home/wei/omnetpp-5.6.2/bin:$PATH"
export LD_LIBRARY_PATH="/home/wei/omnetpp-5.6.2/lib:/home/wei/veins-veins-5.2/out/gcc-release/src:$LD_LIBRARY_PATH"

for SEED in 0 1 2; do
    echo ">>> LearnedMask seed${SEED} started $(date)"
    
    # 清理旧进程
    fuser -k 9999/tcp 2>/dev/null
    killall -9 opp_run sumo 2>/dev/null
    sleep 2
    
    # 启动veins_launchd
    python3 ~/veins-veins-5.2/bin/veins_launchd -vv -c sumo &
    sleep 3
    
    # 清理旧csv
    rm -f cmdp_log.csv cmdp_log_delivery.csv
    
    # 跑仿真
    opp_run -u Cmdenv -c Ch6-OV-LearnedMask-Full -r 0 \
      -n "../../src:.:$HOME/veins_ned_root" \
      -l ../../libcmdp_release.so \
      -l $HOME/veins-veins-5.2/out/gcc-release/src/libveins.so \
      --sim-time-limit=50s --seed-set=$SEED \
      omnetpp.ini 2>&1 | tail -5
    
    # 保存到永久位置
    cp cmdp_log.csv cmdp_log_LearnedMask_s${SEED}.csv
    cp cmdp_log_delivery.csv cmdp_log_delivery_LearnedMask_s${SEED}.csv 2>/dev/null
    
    echo ">>> LearnedMask seed${SEED} done $(date)"
    echo "rows: $(wc -l < cmdp_log_LearnedMask_s${SEED}.csv)"
done

echo "=== ALL DONE $(date) ==="

# 立刻统计
python3 -c "
import csv
for s in [0,1,2]:
    f=f'cmdp_log_LearnedMask_s{s}.csv'
    try:
        rows=[r for r in csv.DictReader(open(f)) if r.get('simTime','')!='simTime']
        n=len(rows)
        feas=sum(1 for r in rows if r.get('feasible','0')=='1')
        reach=sum(1 for r in rows if r.get('reachable','0')=='1')
        hops_f=[float(r['hops']) for r in rows if r.get('feasible','0')=='1' and 'hops' in r]
        h=f'avgHops={sum(hops_f)/len(hops_f):.1f}' if hops_f else 'noFeasHops'
        print(f'LM s{s}: n={n} feas={100*feas/n:.1f}% reach={100*reach/n:.1f}% {h}')
    except Exception as e:
        print(f'LM s{s}: {e}')
"
