#!/usr/bin/env python3
"""
SAF Cost Shaping Patch — 注入LagrangianGreedy + 填充safScores
================================================================
修复Bug 2: SAF cost shaping只在ExactLabel有，LagGreedy完全缺失
三处修改:
  1. CMDPApp.cc: buildFilteredGraphSnapshot()中填充graph.safScores
  2. CmdpRouter.h: LagGreedy评分循环加SAF shaping (DN之后)
  3. CmdpRouter.h: ExactLabel已有SAF代码，无需改动(确认safScores有数据后自动生效)
"""
import sys

BASE = "/home/wei/veins_workspace/cmdp_pkg_v2/src"

def patch_file(filepath, old, new, label):
    with open(filepath) as f:
        content = f.read()
    if old not in content:
        print(f"FAIL {label}: pattern not found in {filepath}")
        print(f"  Looking for: {repr(old[:80])}...")
        return False
    count = content.count(old)
    if count > 1:
        print(f"WARN {label}: pattern found {count} times, replacing first only")
    content = content.replace(old, new, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"OK   {label}")
    return True

# ============================================================
# PATCH 1: CMDPApp.cc — 填充 graph.safScores
# ============================================================
# 替换636行的空注释为实际SAF评分代码

old_1 = '    // SAF: no graph modification here. Local Repair handles SAF in forwardDataPacket()\n    return graph;'

new_1 = '''    // SAF: compute preference scores and inject into graph for route planning
    if (enableSAF_) {
        veins::Coord myPos = getMyPosition();
        veins::Coord mySpd = getMySpeed();

        for (const auto& kv : neighborTable_.getAll()) {
            int nbrId = kv.first;
            const NeighborEntry& e = kv.second;

            veins::Coord nbrPos(e.posX, e.posY, 0);
            veins::Coord nbrSpd(e.speedX, e.speedY, 0);

            // Link lifetime prediction: T_hat = (R - d) / |v_rel|
            double relSpeed = (mySpd - nbrSpd).length();
            double dist = myPos.distance(nbrPos);
            double commR = 300.0;
            double linkLife = (relSpeed > 0.1) ? (commR - dist) / relSpeed : 30.0;
            linkLife = std::max(0.0, std::min(30.0, linkLife));

            // Social metric from SocialGraphManager
            double sm = socialGraphMgr_.getSocialMetric(myVehicleId_, nbrId);

            // Fuzzy inference -> score in [0, 1]
            double score = fuzzyEngine_.evaluate(linkLife, sm);
            graph.safScores[nbrId] = score;
        }

        // Debug: confirm safScores are populated
        static int safDbgCnt = 0;
        if (safDbgCnt++ < 20 && !graph.safScores.empty()) {
            fprintf(stderr, "[SAF-GRAPH] node=%d filled %zu safScores, sample=%.3f\\n",
                    myVehicleId_, graph.safScores.size(),
                    graph.safScores.begin()->second);
        }
    }
    return graph;'''

# ============================================================
# PATCH 2: CmdpRouter.h — LagGreedy评分循环加SAF shaping
# ============================================================
# 在DN shaping三层}闭合后、double rj之前插入SAF block
# 精确匹配: DN块末尾三个} + rj计算行

old_2 = '''                        }
                    }
                }
                double rj = (j == dst) ? 0.0 : trustMgr.computeTrustCost(j);
                double score = cij + lambda_ * rj;
                if (score < bestScore) { bestScore = score; bestJ = j; }'''

new_2 = '''                        }
                    }
                }
                // SAF cost shaping (after DN, density-adaptive)
                if (graph.enableSAF && !graph.safScores.empty()) {
                    int nNbr = graph.adjacency.count(current) ? graph.adjacency.at(current).size() : 0;
                    double safAlpha = std::min(0.5, std::max(0.0, (nNbr - 4) * 0.08));
                    // 4 nbr->0(OFF), 6->0.16, 10+->0.48(cap)
                    if (safAlpha > 0.01) {
                        auto it = graph.safScores.find(j);
                        if (it != graph.safScores.end() && it->second > 0.5) {
                            cij *= (1.0 - safAlpha * (it->second - 0.5) * 2.0);
                            // score=0.5->1.0, score=0.95->1-0.9*alpha
                        }
                    }
                }
                double rj = (j == dst) ? 0.0 : trustMgr.computeTrustCost(j);
                double score = cij + lambda_ * rj;
                if (score < bestScore) { bestScore = score; bestJ = j; }'''

# ============================================================
# Apply patches
# ============================================================
print("=" * 60)
print("SAF Cost Shaping Patch")
print("=" * 60)

ok1 = patch_file(f"{BASE}/CMDPApp.cc", old_1, new_1,
                 "Patch 1: CMDPApp.cc safScores filling")

ok2 = patch_file(f"{BASE}/CmdpRouter.h", old_2, new_2,
                 "Patch 2: CmdpRouter.h LagGreedy SAF shaping")

print("=" * 60)
if ok1 and ok2:
    print("ALL PATCHES APPLIED SUCCESSFULLY")
    print()
    print("Next steps:")
    print("  1. cd ~/omnetpp-5.6.2 && . setenv")
    print("  2. export LD_LIBRARY_PATH=$HOME/veins-veins-5.2/out/gcc-debug/src:$LD_LIBRARY_PATH")
    print("  3. cd ~/veins_workspace/cmdp_pkg_v2 && make 2>&1 | tail -5")
    print("  4. Look for [SAF-GRAPH] in stderr to confirm safScores populated")
    print("  5. Run d200 quick test: G5(SAF-only) and G7(Full)")
else:
    print("SOME PATCHES FAILED — check output above")
    print("Run these to debug:")
    print(f"  sed -n '633,640p' {BASE}/CMDPApp.cc")
    print(f"  sed -n '248,256p' {BASE}/CmdpRouter.h")
    sys.exit(1)
