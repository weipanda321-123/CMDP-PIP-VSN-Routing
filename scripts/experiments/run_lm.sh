#!/bin/bash
cd ~/veins_workspace/cmdp_pkg_v2/simulations/manhattan
ulimit -n 8192
export PATH="/home/wei/omnetpp-5.6.2/bin:$PATH"
export LD_LIBRARY_PATH="/home/wei/omnetpp-5.6.2/lib:/home/wei/veins-veins-5.2/out/gcc-release/src:$LD_LIBRARY_PATH"

LOG=/tmp/learnedmask_overnight.log
echo "=== Started $(date) ===" | tee $LOG

for SEED in 1 2 3; do
    echo ">>> Seed $SEED started $(date)" | tee -a $LOG
    killall -9 opp_run sumo 2>/dev/null; sleep 2
    opp_run -u Cmdenv -c Ch6-OV-LearnedMask-Full -r 0 \
      -n "../../src:.:$HOME/veins_ned_root" \
      -l ../../libcmdp_release.so \
      -l $HOME/veins-veins-5.2/out/gcc-release/src/libveins.so \
      --sim-time-limit=200s --seed-set=$SEED \
      omnetpp.ini >> $LOG 2>&1
    cp results/Ch6-OV-LearnedMask-Full-#0.sca results/Ch6-OV-LearnedMask-Full-seed${SEED}.sca 2>/dev/null
    cp cmdp_log.csv /tmp/LearnedMask-Full-seed${SEED}.csv 2>/dev/null
    echo ">>> Seed $SEED done $(date)" | tee -a $LOG
done
echo "=== ALL DONE $(date) ===" | tee -a $LOG
