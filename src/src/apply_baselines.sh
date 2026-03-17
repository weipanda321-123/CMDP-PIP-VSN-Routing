#!/bin/bash
# apply_baselines.sh — 在 VM 上运行，自动修改 CmdpRouter.h 和 omnetpp.ini
# 用法: cd ~/veins_workspace/cmdp_pkg_v2 && bash src/apply_baselines.sh
set -e

ROUTER_H="$HOME/veins_workspace/cmdp_pkg_v2/src/CmdpRouter.h"
INI_FILE="$HOME/veins_workspace/cmdp_pkg_v2/simulations/manhattan/omnetpp.ini"

echo "=== Step 1: 备份原文件 ==="
cp "$ROUTER_H" "${ROUTER_H}.bak.$(date +%s)"
cp "$INI_FILE" "${INI_FILE}.bak.$(date +%s)"
echo "  ✅ 备份完成"

echo ""
echo "=== Step 2: 查看 CmdpRouter.h 当前 route() 结构 ==="
echo "--- 现有 policyMode 分支: ---"
grep -n 'policyMode_\|solveL\|solveE\|solveR\|route(' "$ROUTER_H" | head -20
echo ""
echo "--- 现有 #include: ---"
grep -n '#include' "$ROUTER_H" | head -10
echo ""

echo "=== Step 3: 修改 CmdpRouter.h ==="

# 3a. 添加 #include（在文件最后一个已有 #include 之后）
if grep -q 'CmdpBaselineAlgorithms.h' "$ROUTER_H"; then
    echo "  SKIP - CmdpBaselineAlgorithms.h 已经 included"
else
    # 找到最后一个 #include 行号
    LAST_INCLUDE=$(grep -n '#include' "$ROUTER_H" | tail -1 | cut -d: -f1)
    if [ -n "$LAST_INCLUDE" ]; then
        sed -i "${LAST_INCLUDE}a #include \"CmdpBaselineAlgorithms.h\"" "$ROUTER_H"
        echo "  ✅ 在第 ${LAST_INCLUDE} 行后添加 #include"
    else
        echo "  ❌ 找不到 #include 行，请手动添加"
    fi
fi

# 3b. 添加成员变量（在 private: 区域）
if grep -q 'dpBbins_' "$ROUTER_H"; then
    echo "  SKIP - baseline 成员变量已存在"
else
    # 找到 "private:" 后的第一个空行或变量声明
    # 策略：在 policyMode_ 声明附近插入
    POLICY_LINE=$(grep -n 'policyMode_' "$ROUTER_H" | head -1 | cut -d: -f1)
    if [ -n "$POLICY_LINE" ]; then
        sed -i "${POLICY_LINE}a\\
\\
    // ---- Baseline hyperparameters (OJCOMS Table 2a) ----\\
    int    dpBbins_       = 200;     // DP: budget discretization bins\\
    int    kspK_          = 20;      // KSP: number of candidate paths\\
    int    bbNodeLimit_   = 100000;  // B\&B: max nodes to explore\\
    double bbTimeLimitMs_ = 1000.0;  // B\&B: time limit per query (ms)\\
    int    gaPopSize_     = 100;     // GA: population size\\
    int    gaGenerations_ = 200;     // GA: max generations\\
    double gaMutationRate_= 0.1;    // GA: mutation probability\\
    double gaLambdaPen_   = 10.0;   // GA: penalty weight" "$ROUTER_H"
        echo "  ✅ 在 policyMode_ (第 ${POLICY_LINE} 行) 后添加成员变量"
    else
        echo "  ⚠️  找不到 policyMode_，尝试在文件末尾 }; 前插入"
        # fallback: 在最后的 }; 前插入
        sed -i '/^};/i \
    // ---- Baseline hyperparameters (OJCOMS Table 2a) ----\
    int    dpBbins_       = 200;\
    int    kspK_          = 20;\
    int    bbNodeLimit_   = 100000;\
    double bbTimeLimitMs_ = 1000.0;\
    int    gaPopSize_     = 100;\
    int    gaGenerations_ = 200;\
    double gaMutationRate_= 0.1;\
    double gaLambdaPen_   = 10.0;' "$ROUTER_H"
        echo "  ✅ 在文件末尾 }; 前添加成员变量"
    fi
fi

# 3c. 添加 route() 分支（用 Python 更精确）
if grep -q 'DPBaseline\|KSPBaseline\|BranchBound\|GABaseline' "$ROUTER_H"; then
    echo "  SKIP - baseline route 分支已存在"
else
    python3 << 'PYEOF'
import re

path = "$ROUTER_H"
# Shell variable won't expand inside Python heredoc, use os
import os
path = os.path.expanduser("~/veins_workspace/cmdp_pkg_v2/src/CmdpRouter.h")

with open(path) as f:
    content = f.read()

# 找到最后一个 solveRLPolicy 或 solveExactLabelSetting 的调用位置
# 然后在其对应的 } 后面插入新分支

# 策略：找 "RLPolicy" 那个 else if 块的结尾 }
# 一般形式：} else if (policyMode_ == "RLPolicy") { ... return solveRLPolicy(...); }
# 我们在这个 } 后面加新代码

# 更安全的策略：找包含 "RLPolicy" 的行，然后往下找到对应的 } 
# 或者更简单：找 route() 函数的最后一个 return 前插入

# 最安全：找 "solveRLPolicy" 这行，然后在后面几行的 } 后插入
new_branches = '''
        // ---- OJCOMS Baseline Algorithms (density sweep) ----
        else if (policyMode_ == "DPBaseline") {
            auto getTrust = [this](int v) { return trustMgr_.getTrustCost(v); };
            auto getDelay = [this, &graph](int u, int v) {
                return delayEst_.estimate(graph, u, v);
            };
            return solveDPResource(graph, src, dst, budget, getTrust, getDelay, dpBbins_);
        }
        else if (policyMode_ == "KSPBaseline") {
            auto getTrust = [this](int v) { return trustMgr_.getTrustCost(v); };
            auto getDelay = [this, &graph](int u, int v) {
                return delayEst_.estimate(graph, u, v);
            };
            return solveKShortestPaths(graph, src, dst, budget, getTrust, getDelay, kspK_);
        }
        else if (policyMode_ == "BranchBound") {
            auto getTrust = [this](int v) { return trustMgr_.getTrustCost(v); };
            auto getDelay = [this, &graph](int u, int v) {
                return delayEst_.estimate(graph, u, v);
            };
            return solveBranchAndBound(graph, src, dst, budget, getTrust, getDelay, bbNodeLimit_, bbTimeLimitMs_);
        }
        else if (policyMode_ == "GABaseline") {
            auto getTrust = [this](int v) { return trustMgr_.getTrustCost(v); };
            auto getDelay = [this, &graph](int u, int v) {
                return delayEst_.estimate(graph, u, v);
            };
            return solveGeneticAlgorithm(graph, src, dst, budget, getTrust, getDelay, gaPopSize_, gaGenerations_, gaMutationRate_, gaLambdaPen_);
        }'''

# 查找插入点：在 "RLPolicy" 分支之后
# 搜索模式：找到包含 "RLPolicy" 的 else if 行
lines = content.split('\n')
insert_after = -1

for i, line in enumerate(lines):
    if 'RLPolicy' in line and 'policyMode_' in line:
        # 找到 RLPolicy 分支，现在往下找这个分支的结尾 }
        brace_count = 0
        started = False
        for j in range(i, min(i + 30, len(lines))):
            if '{' in lines[j]:
                brace_count += lines[j].count('{')
                started = True
            if '}' in lines[j]:
                brace_count -= lines[j].count('}')
            if started and brace_count <= 0:
                insert_after = j
                break
        break

if insert_after < 0:
    # Fallback: 找 solveExactLabelSetting 分支
    for i, line in enumerate(lines):
        if 'ExactBaseline' in line and 'policyMode_' in line:
            brace_count = 0
            started = False
            for j in range(i, min(i + 30, len(lines))):
                if '{' in lines[j]:
                    brace_count += lines[j].count('{')
                    started = True
                if '}' in lines[j]:
                    brace_count -= lines[j].count('}')
                if started and brace_count <= 0:
                    insert_after = j
                    break
            break

if insert_after >= 0:
    lines.insert(insert_after + 1, new_branches)
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ 在第 {insert_after + 1} 行后插入 4 个 baseline 分支")
else:
    print("  ❌ 找不到 RLPolicy/ExactBaseline 分支，请手动插入")
    print("     在 route() 函数的最后一个 else if 后面加入以下代码:")
    print(new_branches[:200] + "...")
PYEOF
fi

echo ""
echo "=== Step 4: 修改 omnetpp.ini ==="

if grep -q 'CMDP-DP' "$INI_FILE"; then
    echo "  SKIP - baseline configs 已存在"
else
    cat >> "$INI_FILE" << 'INICFG'

# ================================================================
#  OJCOMS Baseline Configs (density sweep, added by apply_baselines.sh)
# ================================================================

[Config CMDP-DP]
description = "DP (resource-discretized) baseline"
extends = General
*.node[*].appl.cmdpPolicyMode = "DPBaseline"
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpUseGNN = false

[Config CMDP-KSP]
description = "K-Shortest Paths (Yen, K=20) baseline"
extends = General
*.node[*].appl.cmdpPolicyMode = "KSPBaseline"
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpUseGNN = false

[Config CMDP-BB]
description = "Branch and Bound with Lagrangian relaxation"
extends = General
*.node[*].appl.cmdpPolicyMode = "BranchBound"
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpUseGNN = false

[Config CMDP-GA]
description = "Genetic Algorithm (P=100, G=200)"
extends = General
*.node[*].appl.cmdpPolicyMode = "GABaseline"
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpUseGNN = false
INICFG
    echo "  ✅ 追加 4 个 Config 到 omnetpp.ini"
fi

echo ""
echo "=== Step 5: 验证修改 ==="
echo "--- CmdpRouter.h 新增内容: ---"
grep -n 'DPBaseline\|KSPBaseline\|BranchBound\|GABaseline\|CmdpBaselineAlgorithms\|dpBbins_' "$ROUTER_H"
echo ""
echo "--- omnetpp.ini 新增 Config: ---"
grep -n 'CMDP-DP\|CMDP-KSP\|CMDP-BB\|CMDP-GA' "$INI_FILE"

echo ""
echo "=== Step 6: 编译 ==="
echo "现在运行以下命令编译:"
echo ""
echo "  cd ~/veins_workspace/cmdp_pkg_v2"
echo "  make -j\$(nproc) MODE=debug"
echo ""
echo "如果编译报错，检查:"
echo "  1. delayEst_.estimate(graph, u, v) 的签名是否匹配"
echo "     → 可能需要改为 delayEst_.predictDelay(dist, snir) 或其他"
echo "  2. trustMgr_.getTrustCost(v) 是否存在"
echo "  3. GraphSnapshot 是否有 adj (map<int,vector<int>>)"
echo ""
echo "Done!"
