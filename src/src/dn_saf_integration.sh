#!/bin/bash
# ============================================================
# DN/SAF Integration — Final Patches (ALL APIs CONFIRMED)
# 2026-02-24
#
# 执行方式: 把这个文件传到 VM, 然后逐段执行
# 或者直接按顺序复制粘贴每个 STEP 的命令
#
# 严格按 Step 1→2→3→4→5 顺序执行，每步验证
# ============================================================

set -e  # 出错即停

PROJ=~/veins_workspace/cmdp_pkg_v2
SRC=$PROJ/src

echo "=========================================="
echo " Step 1: Fix Makefile (2 changes)"
echo "=========================================="

cd $PROJ

# 1a. 加 vsn_veins_algos include 路径
sed -i '/^CFLAGS += -I\$(SRC_DIR)/ s|$| -I$(HOME)/veins_workspace/vsn_veins_algos/src|' Makefile
echo "[1a] Include path added:"
grep "CFLAGS.*vsn" Makefile && echo "  ✓" || echo "  ✗ FAILED"

# 1b. mkdir 支持子目录
sed -i 's|@mkdir -p \$(OUT_DIR)|@mkdir -p $(dir $@)|' Makefile
echo "[1b] mkdir fixed:"
grep -n 'mkdir.*dir.*@' Makefile && echo "  ✓" || echo "  ✗ FAILED"

# 1c. 编译测试
echo ""
echo ">>> Compiling (Step 1 verification)..."
cd ~/omnetpp-5.6.2 && . setenv 2>/dev/null
cd $PROJ
make clean && make 2>&1 | tail -5
echo ""
echo "DN symbols: $(nm libcmdp_debug.so 2>/dev/null | grep -c DNBackbone)"
echo "SAF symbols: $(nm libcmdp_debug.so 2>/dev/null | grep -c 'FuzzyEngine\|SocialGraph')"

echo ""
echo ">>> Step 1 DONE. Check above: DN>0, SAF>0 means success."
echo ">>> Press Enter to continue to Step 2, or Ctrl+C to stop."
read

echo "=========================================="
echo " Step 2: CMDPApp.h — add members + decls"
echo "=========================================="

cd $SRC

# 2a. 加 #include (在 NeighborTable.h 之后)
sed -i '/#include "NeighborTable.h"/a\#include "NeighborAdapter.h"\n#include "saf/SAFRelaySelector.h"\n#include "saf/SocialGraphManager.h"\n#include "saf/FuzzyEngine.h"' CMDPApp.h
echo "[2a] Includes added:"
grep -c "NeighborAdapter\|SAFRelaySelector\|SocialGraphManager\|FuzzyEngine" CMDPApp.h

# 2b. 加成员变量 (在 NeighborTable neighborTable_; 之后)
sed -i '/    NeighborTable neighborTable_;/a\    // DN subsystem\n    veins_dn2::DNBackboneManager dnManager_;\n    veins_dn2::NeighborTable dnNeighborTable_;\n    cMessage* dnUpdateTimer_ = nullptr;\n    // SAF subsystem\n    saf::SAFRelaySelector safSelector_;\n    saf::SocialGraphManager socialGraphMgr_;\n    saf::NeighborTable safNeighborTable_;\n    saf::FuzzyEngine fuzzyEngine_;' CMDPApp.h
echo "[2b] Members added:"
grep -c "dnManager_\|safSelector_\|socialGraphMgr_\|fuzzyEngine_" CMDPApp.h

# 2c. 加函数声明 (在 buildGraphSnapshot 之后)
sed -i '/    GraphSnapshot buildGraphSnapshot();/a\    GraphSnapshot buildFilteredGraphSnapshot();\n    void onBeaconUpdateDNSAF(int senderId, double t);\n    void dnPeriodicUpdate();' CMDPApp.h
echo "[2c] Function declarations added:"
grep -c "buildFilteredGraph\|onBeaconUpdateDNSAF\|dnPeriodicUpdate" CMDPApp.h

echo ""
echo ">>> Step 2 DONE."
echo ">>> Press Enter to continue to Step 3, or Ctrl+C to stop."
read

echo "=========================================="
echo " Step 3: CMDPApp.cc — wire up DN/SAF"
echo "=========================================="

# ----------------------------------------------------------
# 3a. initialize() — DN init after enableSAF_ line
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

# Try both variants
for old in [
    '        enableSAF_       = par("enableSAF").boolValue();',
    '        enableSAF_       = par("enableSAF");',
]:
    if old in content:
        new = old + '''

        // --- DN initialization ---
        if (enableDN_) {
            dnManager_.enableDN = true;
            dnManager_.dnPa = 0.5;       // preferential attachment probability
            dnManager_.dnKmax = 20;      // max logical degree
            dnManager_.dnRoutingMode = "overlayPreferred";
            dnUpdateTimer_ = new cMessage("dnUpdateTimer");
            scheduleAt(simTime() + 2.0, dnUpdateTimer_);
            EV_INFO << "[CMDP] DN backbone enabled: Pa=0.5 Kmax=20" << endl;
        }
        // --- SAF initialization ---
        if (enableSAF_) {
            safSelector_.setCommRange(commRange_);
            safSelector_.setTMax(60.0);
            EV_INFO << "[CMDP] SAF relay selection enabled" << endl;
        }'''
        content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        print("Step 3a: initialize() DN/SAF init — Done")
        break
else:
    print("FAIL 3a — enableSAF_ line not found!")
    import subprocess
    subprocess.run(["grep", "-n", "enableSAF", path])
PYEOF

# ----------------------------------------------------------
# 3b. onBeacon() — add DN/SAF sync call at end
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = '''            logGNNTrainingRecord(myVehicleId_, senderId, dist, snir,
                                  modelDelay, simTime().dbl());
        }
    }
}'''

new = '''            logGNNTrainingRecord(myVehicleId_, senderId, dist, snir,
                                  modelDelay, simTime().dbl());
        }
    }

    // Sync to DN/SAF subsystems
    onBeaconUpdateDNSAF(senderId, simTime().dbl());
}'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3b: onBeacon DN/SAF sync — Done")
else:
    print("FAIL 3b — onBeacon tail not found!")
PYEOF

# ----------------------------------------------------------
# 3c. Insert new functions before onDataPacket
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = 'void CMDPApp::onDataPacket(CMDPDataPacket* pkt) {'

new_funcs = '''// ============================================================
//  DN/SAF subsystem sync (called after each beacon)
// ============================================================
void CMDPApp::onBeaconUpdateDNSAF(int senderId, double t) {
    if (enableDN_) {
        DNTableAdapter::syncToDN(neighborTable_, dnNeighborTable_, t);
    }
    if (enableSAF_) {
        SAFTableAdapter::syncToSAF(neighborTable_, safNeighborTable_, t);
        socialGraphMgr_.registerNode(senderId);
        socialGraphMgr_.registerNode(myVehicleId_);
        socialGraphMgr_.recordContact(myVehicleId_, senderId, t);
    }
}

// ============================================================
//  DN periodic update (called every 2s by timer)
// ============================================================
void CMDPApp::dnPeriodicUpdate() {
    if (!enableDN_) return;
    double t = simTime().dbl();
    DNTableAdapter::syncToDN(neighborTable_, dnNeighborTable_, t);
    dnManager_.dnUpdate(dnNeighborTable_, getRNG(0), t);
    EV_DEBUG << "[DN] periodic update: logicalNeighbors="
             << dnManager_.LNeighbors.size() << endl;
}

'''

if old in content:
    content = content.replace(old, new_funcs + old, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3c: new functions (onBeaconUpdateDNSAF + dnPeriodicUpdate) — Done")
else:
    print("FAIL 3c — onDataPacket not found!")
PYEOF

# ----------------------------------------------------------
# 3d. handleSelfMsg — add dnUpdateTimer_ dispatch
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = '''void CMDPApp::handleSelfMsg(cMessage* msg) {
    if (msg == beaconTimer_) {'''

new = '''void CMDPApp::handleSelfMsg(cMessage* msg) {
    // DN periodic update timer (every 2s)
    if (msg == dnUpdateTimer_) {
        dnPeriodicUpdate();
        scheduleAt(simTime() + 2.0, dnUpdateTimer_);
        return;
    }
    if (msg == beaconTimer_) {'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3d: handleSelfMsg dnUpdateTimer — Done")
else:
    print("FAIL 3d — handleSelfMsg structure not matched!")
PYEOF

# ----------------------------------------------------------
# 3e. generateDataPacket — graph build dispatch
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = '    GraphSnapshot graph = buildGraphSnapshot();'
new = '    GraphSnapshot graph = (enableDN_ || enableSAF_) ? buildFilteredGraphSnapshot() : buildGraphSnapshot();'

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3e: graph build dispatch — Done")
else:
    print("FAIL 3e — buildGraphSnapshot line not found!")
PYEOF

# ----------------------------------------------------------
# 3f. Add buildFilteredGraphSnapshot() after buildGraphSnapshot()
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = '''    lastGraphTime_ = now;
    return graph;
}'''

new_func = r'''    lastGraphTime_ = now;
    return graph;
}

// ============================================================
//  DN-filtered + SAF-scored graph (Chapter 6 ablation)
// ============================================================
GraphSnapshot CMDPApp::buildFilteredGraphSnapshot() {
    // Start with full global topology graph
    GraphSnapshot graph = buildGraphSnapshot();

    // --- DN filtering: prune non-backbone edges from my node ---
    if (enableDN_) {
        auto filtered = DNFilter::getFilteredCandidates(neighborTable_, dnManager_);
        if (!filtered.empty()) {
            std::set<int> kept(filtered.begin(), filtered.end());
            auto it = graph.adjacency.find(myVehicleId_);
            if (it != graph.adjacency.end()) {
                std::vector<int> newAdj;
                for (int nid : it->second) {
                    if (kept.count(nid)) newAdj.push_back(nid);
                }
                it->second = newAdj;
                EV_DEBUG << "[DN] filter: " << neighborTable_.size()
                         << " nbrs -> " << newAdj.size() << " backbone" << endl;
            }
        }
    }

    // --- SAF scoring: compute fuzzy relay score for each neighbor ---
    if (enableSAF_) {
        veins::Coord myPos = getMyPosition();
        veins::Coord mySpd = getMySpeed();
        auto entries = neighborTable_.getAllEntries();

        for (const auto& e : entries) {
            veins::Coord nbrPos(e.posX, e.posY, 0);
            veins::Coord nbrSpd(e.speedX, e.speedY, 0);

            // Link lifetime prediction: T_hat = (R - d) / |v_rel|
            double relSpeed = (mySpd - nbrSpd).length();
            double dist = myPos.distance(nbrPos);
            double linkLife = (relSpeed > 0.1) ?
                (commRange_ - dist) / relSpeed : 60.0;
            linkLife = std::max(0.0, std::min(60.0, linkLife));

            // Social metric from encounter history
            double socialMetric = socialGraphMgr_.getSocialMetric(
                myVehicleId_, e.vehicleId);

            // Fuzzy inference: 25 Mamdani rules -> score in [0,1]
            double score = fuzzyEngine_.evaluate(linkLife, socialMetric);

            graph.safScores[e.vehicleId] = score;
        }

        // Prune very low SAF candidates from my edges (score < 0.15)
        auto it = graph.adjacency.find(myVehicleId_);
        if (it != graph.adjacency.end() && !graph.safScores.empty()) {
            std::vector<int> newAdj;
            for (int nid : it->second) {
                auto sit = graph.safScores.find(nid);
                // Keep if: no score (remote node) OR score >= 0.15
                if (sit == graph.safScores.end() || sit->second >= 0.15) {
                    newAdj.push_back(nid);
                }
            }
            if (!newAdj.empty()) {
                it->second = newAdj;
            }
            // else keep all — don't isolate ourselves
        }
    }

    return graph;
}'''

if old in content:
    content = content.replace(old, new_func, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3f: buildFilteredGraphSnapshot — Done")
else:
    print("FAIL 3f — buildGraphSnapshot tail not matched!")
    import subprocess
    subprocess.run(["grep", "-n", "lastGraphTime_", path])
PYEOF

# ----------------------------------------------------------
# 3g. Destructor — clean up dnUpdateTimer_
# ----------------------------------------------------------
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CMDPApp.cc"
with open(path) as f:
    content = f.read()

old = 'CMDPApp::~CMDPApp() {'
new = 'CMDPApp::~CMDPApp() {\n    cancelAndDelete(dnUpdateTimer_);'

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3g: destructor cleanup — Done")
else:
    print("WARN 3g — destructor not found, check manually")
PYEOF

echo ""
echo ">>> Step 3 DONE. Now need to add safScores to GraphSnapshot."
echo ">>> Press Enter to continue to Step 3-extra, or Ctrl+C to stop."
read

echo "=========================================="
echo " Step 3-extra: Add safScores to GraphSnapshot"
echo "=========================================="

# GraphSnapshot 在 CmdpRouter.h 第38行开始
# 需要在结构体 }; 之前加 safScores 字段
python3 - << 'PYEOF'
path = "/home/wei/veins_workspace/cmdp_pkg_v2/src/CmdpRouter.h"
with open(path) as f:
    content = f.read()

# 在 getNeighbors 函数后面、结构体闭合 }; 之前加 safScores
old = '''    std::vector<int> getNeighbors(int node) const {
        auto it = adjacency.find(node);
        if (it != adjacency.end()) return it->second;
        return {};
    }
};'''

new = '''    std::vector<int> getNeighbors(int node) const {
        auto it = adjacency.find(node);
        if (it != adjacency.end()) return it->second;
        return {};
    }

    // SAF fuzzy relay scores (Chapter 4 integration)
    std::map<int, double> safScores;

    double getSAFScore(int node) const {
        auto it = safScores.find(node);
        return (it != safScores.end()) ? it->second : 0.5;  // neutral default
    }
};'''

if old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Step 3-extra: GraphSnapshot.safScores — Done")
else:
    print("FAIL — GraphSnapshot tail not matched!")
PYEOF

echo ""
echo ">>> Step 3-extra DONE."
echo ">>> Press Enter to continue to Step 4, or Ctrl+C to stop."
read

echo "=========================================="
echo " Step 4: omnetpp.ini — add G5/G6 configs"
echo "=========================================="

cat >> $PROJ/simulations/manhattan/omnetpp.ini << 'EOF'

# ============================================================
# G5: SAF + PIP/CMDP (ablation: no DN)
# ============================================================
[Config CMDP-SAF-PIP]
extends = CMDP-PIP
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = true

# ============================================================
# G6: DN + PIP/CMDP (ablation: no SAF)
# ============================================================
[Config CMDP-DN-PIP]
extends = CMDP-PIP
*.node[*].appl.enableDN = true
*.node[*].appl.enableSAF = false
EOF

echo "[4] New configs:"
grep "CMDP-SAF-PIP\|CMDP-DN-PIP" $PROJ/simulations/manhattan/omnetpp.ini

echo ""
echo ">>> Step 4 DONE."
echo ">>> Press Enter to compile and test, or Ctrl+C to stop."
read

echo "=========================================="
echo " Step 5: Compile + Verify"
echo "=========================================="

cd ~/omnetpp-5.6.2 && . setenv 2>/dev/null
cd $PROJ
make clean && make 2>&1 | tail -30

echo ""
echo "=========================================="
if [ -f libcmdp_debug.so ]; then
    echo "✅ BUILD SUCCESSFUL"
    echo "  DN symbols:       $(nm libcmdp_debug.so | grep -c DNBackbone)"
    echo "  SAF symbols:      $(nm libcmdp_debug.so | grep -c 'FuzzyEngine\|SocialGraph')"
    echo "  FilteredGraph:    $(nm libcmdp_debug.so | grep -c buildFilteredGraph)"
    echo "  onBeaconDNSAF:    $(nm libcmdp_debug.so | grep -c onBeaconUpdateDNSAF)"
    echo "  dnPeriodicUpdate: $(nm libcmdp_debug.so | grep -c dnPeriodicUpdate)"
else
    echo "❌ BUILD FAILED — check errors above"
fi
echo "=========================================="
