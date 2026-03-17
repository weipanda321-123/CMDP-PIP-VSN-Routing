#!/bin/bash
# fix_router_lambdas.sh — 修复 CmdpRouter.h 中 baseline 分支的 lambda
set -e
ROUTER_H="$HOME/veins_workspace/cmdp_pkg_v2/src/CmdpRouter.h"

echo "=== 1. 替换新版 CmdpBaselineAlgorithms.h ==="
cp ~/veins_workspace/cmdp_pkg_v2/src/CmdpBaselineAlgorithms.h \
   ~/veins_workspace/cmdp_pkg_v2/src/CmdpBaselineAlgorithms.h.old 2>/dev/null || true
# 用户需要先把新文件复制到 src/

echo "=== 2. 修复 lambda: getTrustCost → computeTrustCost ==="
sed -i '/OJCOMS Baseline/,/gaLambdaPen_/{
  s/trustMgr\.getTrustCost/trustMgr.computeTrustCost/g
  s/delayEst\.estimate(graph, u, v)/delayEst.predictDelayByDistance(baselineNodeDist(graph, u, v))/g
}' "$ROUTER_H"

echo "=== 3. 验证 ==="
echo "--- lambda 内容: ---"
grep -A3 'DPBaseline\|KSPBaseline\|BranchBound\|GABaseline' "$ROUTER_H" | head -40

echo ""
echo "Done! 现在编译:"
echo "  cd ~/veins_workspace/cmdp_pkg_v2 && make -j\$(nproc) MODE=debug 2>&1 | tail -30"
