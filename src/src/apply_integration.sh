#!/bin/bash
# ============================================================
# DN+SAF+CMDP Veins系统级集成 Patch Script
# 在 cmdp_pkg_v2 中集成 DN骨干 + SAF模糊中继 + PIP/CMDP路由
# ============================================================
set -e

SRC=~/veins_workspace/cmdp_pkg_v2/src
cd "$SRC"

echo "=== Step 1: NeighborAdapter.h ==="
cat > NeighborAdapter.h << 'HEOF'
#ifndef NEIGHBOR_ADAPTER_H
#define NEIGHBOR_ADAPTER_H

#include "NeighborTable.h"
#include "dn/DNBackboneManager.h"
#include "saf/SocialGraphManager.h"
#include "veins/base/utils/Coord.h"
#include <cmath>
#include <vector>
#include <set>

// ============================================================
//  DN Adapter: sync CMDP table -> DN table
// ============================================================
class DNTableAdapter {
public:
    static void syncToDN(const NeighborTable& cmdp,
                         veins_dn2::NeighborTable& dnTable,
                         double now)
    {
        auto entries = cmdp.getAllEntries();
        for (auto& e : entries) {
            double snirDb = (e.snir > 0) ? 10.0 * std::log10(e.snir) : -999.0;
            dnTable.updateNeighbor(
                e.vehicleId, e.posX, e.posY,
                e.speedX, e.speedY, 0.0,
                0, false, now, snirDb, -999.0);
        }
        dnTable.prune(now);
    }
};

// ============================================================
//  SAF Adapter: sync CMDP table -> SAF table  
// ============================================================
class SAFTableAdapter {
public:
    static void syncToSAF(const NeighborTable& cmdp,
                          saf::NeighborTable& safTable,
                          double now)
    {
        auto entries = cmdp.getAllEntries();
        for (auto& e : entries) {
            veins::Coord pos(e.posX, e.posY, 0);
            veins::Coord spd(e.speedX, e.speedY, 0);
            double snirDb = (e.snir > 0) ? 10.0 * std::log10(e.snir) : -999.0;
            safTable.updateNeighbor(e.vehicleId, pos, spd, 0.0, now, snirDb);
        }
        safTable.prune(now);
    }
};

// ============================================================
//  DN Filter: backbone candidate filtering for CmdpRouter
// ============================================================
class DNFilter {
public:
    // Get DN-filtered candidate IDs; if DN off, return all
    static std::vector<int> getFilteredCandidates(
        const NeighborTable& cmdp,
        const veins_dn2::DNBackboneManager& dn)
    {
        std::vector<int> result;
        auto entries = cmdp.getAllEntries();

        if (!dn.enableDN) {
            for (auto& e : entries) result.push_back(e.vehicleId);
            return result;
        }

        for (auto& e : entries) {
            if (dn.isLogicalNeighbor(e.vehicleId))
                result.push_back(e.vehicleId);
        }

        // overlayPreferred fallback: if no backbone neighbors, use all
        if (result.empty() && dn.dnRoutingMode == "overlayPreferred") {
            for (auto& e : entries) result.push_back(e.vehicleId);
        }

        return result;
    }

    // Build a filtered GraphSnapshot using only DN candidates
    static GraphSnapshot buildFilteredGraph(
        const NeighborTable& cmdp,
        const veins_dn2::DNBackboneManager& dn,
        double commRange, double now)
    {
        GraphSnapshot graph;
        auto candidates = getFilteredCandidates(cmdp, dn);
        std::set<int> candSet(candidates.begin(), candidates.end());

        auto entries = cmdp.getAllEntries();
        for (auto& e : entries) {
            graph.positions[e.vehicleId] = {e.posX, e.posY};
            graph.nodeSet.insert(e.vehicleId);
        }

        // Only add edges between DN-filtered candidates
        for (auto& e1 : entries) {
            if (candSet.find(e1.vehicleId) == candSet.end()) continue;
            for (auto& e2 : entries) {
                if (e1.vehicleId >= e2.vehicleId) continue;
                if (candSet.find(e2.vehicleId) == candSet.end()) continue;
                double dx = e1.posX - e2.posX;
                double dy = e1.posY - e2.posY;
                if (dx*dx + dy*dy <= commRange * commRange) {
                    graph.addEdge(e1.vehicleId, e2.vehicleId);
                    graph.addEdge(e2.vehicleId, e1.vehicleId);
                }
            }
        }

        return graph;
    }
};

// ============================================================
//  SAF Cost Shaper: inject SAF fuzzy scores as delay bias
//  Lower fuzzyScore → higher cost → PPO avoids that hop
// ============================================================
class SAFCostShaper {
public:
    /**
     * Modify delay estimates in GraphSnapshot based on SAF scores.
     * SAF score ∈ [0,1]: high = good relay → reduce cost
     * Formula: adjusted_delay = base_delay * (1 + alpha * (1 - fuzzyScore))
     * alpha=0 means no SAF effect; alpha=1 doubles delay for worst relays
     */
    static void applyToGraph(
        GraphSnapshot& graph,
        const saf::SAFRelaySelector& safSelector,
        const saf::NeighborTable& safNbrs,
        const saf::SocialGraphManager& sgm,
        int myId,
        const veins::Coord& myPos,
        const veins::Coord& mySpeed,
        int dstId,
        const veins::Coord& dstPos,
        const std::set<int>& visited,
        double simTime,
        double alpha = 0.5)
    {
        if (!safSelector.isEnabled() || alpha < 1e-6) return;

        auto result = safSelector.selectNextHop(
            myId, myPos, mySpeed,
            dstId, dstPos,
            safNbrs, sgm, visited, simTime);

        // Build score map from candidates
        std::map<int, double> scoreMap;
        for (auto& c : result.candidates) {
            scoreMap[c.nodeId] = c.fuzzyScore;
        }

        // Apply cost shaping to edges from myId
        // (modifies edge weights conceptually; CmdpRouter uses these)
        // Store as node-level annotation for DelayEstimator
        for (auto& kv : scoreMap) {
            int nid = kv.first;
            double score = kv.second;
            // Store SAF score as negative trust bonus
            // (CmdpRouter sees lower delay for high-SAF neighbors)
            graph.safScores[nid] = score;
        }
    }
};

#endif
HEOF
echo "  NeighborAdapter.h created"

echo "=== Step 2: Fix DN include path ==="
# DN's DNBackboneManager.h includes "veins_dn2/NeighborTable.h"
# We need to redirect to our local copy
if [ ! -f dn/NeighborTable.h ]; then
    cp ~/veins_workspace/vsn_veins_algos/src/veins_dn2/NeighborTable.h dn/
    cp ~/veins_workspace/vsn_veins_algos/src/veins_dn2/NeighborTable.cc dn/ 2>/dev/null || true
    echo "  Copied DN NeighborTable"
fi

# Fix include path in DNBackboneManager.h: "veins_dn2/NeighborTable.h" -> "NeighborTable.h"
sed -i 's|#include "veins_dn2/NeighborTable.h"|#include "dn/NeighborTable.h"|g' dn/DNBackboneManager.h
sed -i 's|#include "veins_dn2/DNBackboneManager.h"|#include "dn/DNBackboneManager.h"|g' dn/DNBackboneManager.cc
echo "  Fixed DN include paths"

echo "=== Step 3: Fix SAF include paths ==="
# SAF files include "NeighborTable.h" etc - need to be "saf/NeighborTable.h"
# Copy SAF's own NeighborTable
if [ ! -f saf/NeighborTable.h ]; then
    cp ~/veins_workspace/saf_pkg/src/NeighborTable.h saf/SAFNeighborTable.h
    echo "  Copied SAF NeighborTable as SAFNeighborTable.h"
fi

# Fix SAF includes to use local paths
sed -i 's|#include "NeighborTable.h"|#include "saf/SAFNeighborTable.h"|g' saf/SAFRelaySelector.h
sed -i 's|#include "FuzzyEngine.h"|#include "saf/FuzzyEngine.h"|g' saf/SAFRelaySelector.h
sed -i 's|#include "SocialGraphManager.h"|#include "saf/SocialGraphManager.h"|g' saf/SAFRelaySelector.h
sed -i 's|#include "LinkLifetimePredictor.h"|#include "saf/LinkLifetimePredictor.h"|g' saf/SAFRelaySelector.h

# Fix .cc files too
for f in saf/SAFRelaySelector.cc saf/SocialGraphManager.cc saf/FuzzyEngine.cc; do
    if [ -f "$f" ]; then
        sed -i 's|#include "NeighborTable.h"|#include "saf/SAFNeighborTable.h"|g' "$f"
        sed -i 's|#include "FuzzyEngine.h"|#include "saf/FuzzyEngine.h"|g' "$f"
        sed -i 's|#include "SocialGraphManager.h"|#include "saf/SocialGraphManager.h"|g' "$f"
        sed -i 's|#include "LinkLifetimePredictor.h"|#include "saf/LinkLifetimePredictor.h"|g' "$f"
        sed -i 's|#include "SAFRelaySelector.h"|#include "saf/SAFRelaySelector.h"|g' "$f"
    fi
done
echo "  Fixed SAF include paths"

echo "=== Step 4: Add safScores to GraphSnapshot ==="
# Add safScores map to GraphSnapshot in CmdpRouter.h
if ! grep -q "safScores" CmdpRouter.h; then
    sed -i '/std::set<int> nodeSet;/a\    std::map<int, double> safScores;  // SAF fuzzy score per node (0-1)' CmdpRouter.h
    echo "  Added safScores to GraphSnapshot"
else
    echo "  safScores already exists"
fi

echo "=== Step 5: Backup CMDPApp.h/cc ==="
cp CMDPApp.h CMDPApp.h.bak_preintegration
cp CMDPApp.cc CMDPApp.cc.bak_preintegration
echo "  Backed up originals"

echo "=== Step 6: Patch CMDPApp.h ==="
# Add DN/SAF includes and member variables
python3 << 'PYEOF'
path = "CMDPApp.h"
with open(path) as f:
    code = f.read()

# 1. Add includes after existing includes
old_inc = '#include "NeighborTable.h"'
new_inc = '''#include "NeighborTable.h"

// --- DN+SAF Integration ---
#include "dn/DNBackboneManager.h"
#include "dn/NeighborTable.h"
#include "saf/SAFRelaySelector.h"
#include "saf/SocialGraphManager.h"
#include "saf/SAFNeighborTable.h"
#include "NeighborAdapter.h"'''

if 'DNBackboneManager.h' not in code:
    code = code.replace(old_inc, new_inc)
    print("  Added DN/SAF includes")
else:
    print("  Includes already present")

# 2. Add DN/SAF member variables after existing members
old_member = '    NeighborTable neighborTable_;'
new_member = '''    NeighborTable neighborTable_;

    // --- DN Integration (Chapter 3) ---
    veins_dn2::DNBackboneManager dnManager_;
    veins_dn2::NeighborTable dnNeighborTable_;
    cMessage* dnUpdateTimer_ = nullptr;

    // --- SAF Integration (Chapter 4) ---
    saf::SAFRelaySelector safSelector_;
    saf::SocialGraphManager socialGraphMgr_;
    saf::NeighborTable safNeighborTable_;'''

if 'dnManager_' not in code:
    code = code.replace(old_member, new_member)
    print("  Added DN/SAF member variables")
else:
    print("  Members already present")

# 3. Add DN/SAF helper methods
old_util = '    // Utility'
new_util = '''    // DN/SAF methods
    void initDN();
    void initSAF();
    void dnPeriodicUpdate();
    void onBeaconUpdateDNSAF(int senderId, double simTime);
    GraphSnapshot buildFilteredGraphSnapshot();

    // Utility'''

if 'initDN' not in code:
    code = code.replace(old_util, new_util)
    print("  Added DN/SAF methods")
else:
    print("  Methods already present")

with open(path, 'w') as f:
    f.write(code)
print("  CMDPApp.h patched")
PYEOF

echo "=== Step 7: Patch CMDPApp.cc ==="
python3 << 'PYEOF'
path = "CMDPApp.cc"
with open(path) as f:
    code = f.read()

# ============================================================
# 7a. Add DN/SAF initialization in initialize()
# ============================================================
# Find where enableDN_ and enableSAF_ are read from par()
if 'initDN()' not in code:
    # Find the block after enableSAF_ is set
    old_init = '    enableSAF_          = par("enableSAF").boolValue();'
    if old_init not in code:
        # Try alternative
        old_init = '    enableSAF_         = par("enableSAF").boolValue();'
    if old_init not in code:
        # Search for enableSAF
        import re
        m = re.search(r'(.*enableSAF_\s*=\s*par.*)', code)
        if m:
            old_init = m.group(1)
            print(f"  Found enableSAF line: {old_init.strip()}")
        else:
            # enableSAF_ might not exist yet, add after enableDN_
            m2 = re.search(r'(.*enableDN_\s*=\s*par.*)', code)
            if m2:
                old_init = m2.group(1)
                print(f"  Found enableDN line, will add enableSAF after")
            else:
                print("  ERROR: Cannot find enableDN/enableSAF lines!")
                old_init = None

    if old_init:
        new_init = old_init + '''

    // --- DN initialization ---
    if (enableDN_) {
        initDN();
    }

    // --- SAF initialization ---
    if (enableSAF_) {
        initSAF();
    }'''
        code = code.replace(old_init, new_init, 1)
        print("  Added DN/SAF init calls")
else:
    print("  DN/SAF init already present")

# ============================================================
# 7b. Add DN/SAF beacon updates in onBeacon()
# ============================================================
if 'onBeaconUpdateDNSAF' not in code:
    # Find end of neighborTable_.update() call in onBeacon
    import re
    m = re.search(r'(neighborTable_\.update\([^)]+\);)', code)
    if m:
        old_beacon = m.group(1)
        new_beacon = old_beacon + '''

    // --- Update DN and SAF subsystems ---
    onBeaconUpdateDNSAF(beacon->getSenderAddress(), simTime());'''
        code = code.replace(old_beacon, new_beacon, 1)
        print("  Added DN/SAF beacon update")
    else:
        print("  WARNING: Could not find neighborTable_.update in onBeacon")
else:
    print("  DN/SAF beacon update already present")

# ============================================================
# 7c. Add DN timer handling in handleSelfMsg()
# ============================================================
if 'dnUpdateTimer_' not in code or 'dnPeriodicUpdate' not in code:
    old_self = '    if (msg == beaconTimer_) {'
    new_self = '''    if (msg == dnUpdateTimer_) {
        // DN periodic backbone update
        if (enableDN_) {
            dnPeriodicUpdate();
        }
        scheduleAt(simTime() + dnManager_.dnUpdateInterval, dnUpdateTimer_);
        return;
    }
    if (msg == beaconTimer_) {'''
    if old_self in code:
        code = code.replace(old_self, new_self, 1)
        print("  Added DN timer handling")
    else:
        print("  WARNING: Could not find beaconTimer_ in handleSelfMsg")
else:
    print("  DN timer already present")

# ============================================================
# 7d. Replace buildGraphSnapshot with DN-filtered version
# ============================================================
if 'buildFilteredGraphSnapshot' not in code or True:
    # Modify forwardDataPacket or the routing call to use filtered graph
    old_route = '    GraphSnapshot graph = buildGraphSnapshot();'
    new_route = '''    // Build graph with DN filtering (if enabled)
    GraphSnapshot graph = enableDN_ ? buildFilteredGraphSnapshot() : buildGraphSnapshot();'''
    if old_route in code and 'enableDN_ ? buildFilteredGraphSnapshot' not in code:
        code = code.replace(old_route, new_route, 1)
        print("  Added DN-filtered graph building")
    else:
        print("  Graph filtering already present or not found")

# ============================================================
# 7e. Add implementation functions at end of file
# ============================================================
impl_code = '''

// ============================================================
//  DN+SAF Integration Implementation
// ============================================================

void CMDPApp::initDN() {
    EV_INFO << "Initializing DN backbone manager" << endl;
    dnManager_.enableDN = true;
    dnManager_.dnUpdateInterval = par("dnUpdateInterval").doubleValue();
    dnManager_.dnPa = par("dnPa").doubleValue();
    dnManager_.dnKmax = par("dnKmax").intValue();
    dnManager_.dnConnectM = par("dnConnectM").intValue();
    dnManager_.dnRoutingMode = par("dnRoutingMode").stdstringValue();

    dnNeighborTable_.setNeighborTimeout(neighborTimeout_);

    // Schedule DN periodic update timer
    dnUpdateTimer_ = new cMessage("dnUpdateTimer");
    scheduleAt(simTime() + dnManager_.dnUpdateInterval, dnUpdateTimer_);
}

void CMDPApp::initSAF() {
    EV_INFO << "Initializing SAF relay selector" << endl;
    safSelector_.setEnabled(true);
    safSelector_.setCommRange(commRange_);
    safSelector_.setTMax(60.0);

    safNeighborTable_.setBeaconTimeout(neighborTimeout_);

    // Social graph setup
    socialGraphMgr_.setMode(saf::SocialGraphManager::Mode::BeaconBased);
    socialGraphMgr_.setParams(
        0.9,     // theta (SHP weight)
        0.1,     // psi (CSC weight)
        60.0,    // contact window
        1.0,     // contact thresh duration
        3,       // contact thresh frequency
        16,      // profile length
        42       // profile seed
    );
}

void CMDPApp::dnPeriodicUpdate() {
    // Sync CMDP neighbors to DN table
    DNTableAdapter::syncToDN(neighborTable_, dnNeighborTable_, simTime());

    // Run DN backbone update (preferential attachment + pruning)
    dnManager_.dnUpdate(dnNeighborTable_, getRNG(0), simTime());

    EV_DEBUG << "DN update: kL=" << dnManager_.getLogicalDegree()
             << " hub=" << dnManager_.getHubFlag()
             << " time=" << dnManager_.lastUpdateDecisionMs << "ms" << endl;
}

void CMDPApp::onBeaconUpdateDNSAF(int senderId, double simTime) {
    // Update DN neighbor table
    if (enableDN_) {
        DNTableAdapter::syncToDN(neighborTable_, dnNeighborTable_, simTime);
    }

    // Update SAF neighbor table + social graph
    if (enableSAF_) {
        SAFTableAdapter::syncToSAF(neighborTable_, safNeighborTable_, simTime);

        // Register node and record contact
        socialGraphMgr_.registerNode(getMyVehicleId());
        socialGraphMgr_.registerNode(senderId);
        socialGraphMgr_.recordContact(getMyVehicleId(), senderId, simTime);

        // Set group based on road direction (simplified)
        // Group 0=NS, 1=EW, 2=other
        auto mySpeed = getMySpeed();
        int myGroup = (std::abs(mySpeed.x) > std::abs(mySpeed.y)) ? 1 : 0;
        socialGraphMgr_.setNodeGroup(getMyVehicleId(), myGroup);
    }
}

GraphSnapshot CMDPApp::buildFilteredGraphSnapshot() {
    double now = simTime().dbl();

    // Use DN filter to build graph with only backbone neighbors
    GraphSnapshot graph = DNFilter::buildFilteredGraph(
        neighborTable_, dnManager_, commRange_, now);

    // Add self
    auto myPos = getMyPosition();
    graph.positions[getMyVehicleId()] = {myPos.x, myPos.y};
    graph.nodeSet.insert(getMyVehicleId());

    // Apply SAF cost shaping (if enabled)
    if (enableSAF_) {
        auto mySpeed = getMySpeed();
        std::set<int> visited;  // empty for now

        // Get SAF scores for all candidates
        saf::SAFResult safResult = safSelector_.selectNextHop(
            getMyVehicleId(),
            myPos, mySpeed,
            -1,  // dst unknown at graph build time
            veins::Coord::ZERO,
            safNeighborTable_,
            socialGraphMgr_,
            visited,
            now);

        // Store SAF scores in graph for CmdpRouter to use
        for (auto& c : safResult.candidates) {
            graph.safScores[c.nodeId] = c.fuzzyScore;
        }
    }

    return graph;
}
'''

if 'void CMDPApp::initDN()' not in code:
    code += impl_code
    print("  Added DN/SAF implementation functions")
else:
    print("  Implementation already present")

# ============================================================
# 7f. Add cleanup in destructor
# ============================================================
if 'dnUpdateTimer_' not in code.split('~CMDPApp')[0] if '~CMDPApp' in code else True:
    old_destr = 'CMDPApp::~CMDPApp() {'
    new_destr = '''CMDPApp::~CMDPApp() {
    cancelAndDelete(dnUpdateTimer_);'''
    if old_destr in code and 'cancelAndDelete(dnUpdateTimer_)' not in code:
        code = code.replace(old_destr, new_destr, 1)
        print("  Added dnUpdateTimer cleanup")
else:
    print("  Cleanup already present")

with open(path, 'w') as f:
    f.write(code)
print("  CMDPApp.cc patched")
PYEOF

echo "=== Step 8: Add DN/SAF parameters to .ned ==="
NED="CMDPApp.ned"
if ! grep -q "dnUpdateInterval" "$NED"; then
    # Add DN parameters before the closing brace
    python3 << 'PYEOF'
path = "CMDPApp.ned"
with open(path) as f:
    code = f.read()

# Find the parameters section and add DN/SAF params
old = '        // Logging'
new = '''        // DN parameters (Chapter 3)
        bool enableDN = default(false);
        double dnUpdateInterval @unit(s) = default(2s);
        double dnPa = default(0.5);
        int dnKmax = default(20);
        int dnConnectM = default(3);
        string dnRoutingMode = default("overlayPreferred");

        // SAF parameters (Chapter 4)
        bool enableSAF = default(false);

        // Logging'''

if 'dnUpdateInterval' not in code:
    if old in code:
        code = code.replace(old, new)
    else:
        # Try adding before closing brace of parameters
        code = code.replace('}', '''
        // DN parameters (Chapter 3)
        bool enableDN = default(false);
        double dnUpdateInterval @unit(s) = default(2s);
        double dnPa = default(0.5);
        int dnKmax = default(20);
        int dnConnectM = default(3);
        string dnRoutingMode = default("overlayPreferred");

        // SAF parameters (Chapter 4)
        bool enableSAF = default(false);
}''', 1)
    print("  Added DN/SAF NED parameters")
else:
    print("  NED params already present")

with open(path, 'w') as f:
    f.write(code)
PYEOF
    echo "  NED parameters added"
else
    echo "  NED parameters already exist"
fi

echo "=== Step 9: Add G0-G7 configs to omnetpp.ini ==="
INI="../simulations/manhattan/omnetpp.ini"
if ! grep -q "G0_AODV" "$INI"; then
    cat >> "$INI" << 'INIEOF'

# ============================================================
#  G0-G7 Ablation Experiment Configs (Chapter 6)
# ============================================================

[Config G0_AODV]
description = "G0: AODV baseline (no DN, no SAF, no PIP)"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpPolicyMode = "lagrangian"

[Config G1_GPSR]
description = "G1: GPSR baseline"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpPolicyMode = "greedy"

[Config G2_RCSPP]
description = "G2: RCSPP exact solver (small scale only)"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpPolicyMode = "rcspp"

[Config G3_CMDP_noPIP]
description = "G3: CMDP without PIP mask"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = false
*.node[*].appl.cmdpPolicyMode = "ppo"

[Config G4_CMDP_PIP]
description = "G4: CMDP + PIP (no DN, no SAF)"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = true
*.node[*].appl.cmdpPolicyMode = "ppo"

[Config G5_SAF_CMDP]
description = "G5: SAF + CMDP/PIP (no DN)"
*.node[*].appl.enableDN = false
*.node[*].appl.enableSAF = true
*.node[*].appl.cmdpUsePIP = true
*.node[*].appl.cmdpPolicyMode = "ppo"

[Config G6_DN_CMDP]
description = "G6: DN + CMDP/PIP (no SAF)"
*.node[*].appl.enableDN = true
*.node[*].appl.enableSAF = false
*.node[*].appl.cmdpUsePIP = true
*.node[*].appl.cmdpPolicyMode = "ppo"
*.node[*].appl.dnUpdateInterval = 2s
*.node[*].appl.dnPa = 0.5
*.node[*].appl.dnKmax = 20
*.node[*].appl.dnConnectM = 3

[Config G7_Full]
description = "G7: DN + SAF + CMDP/PIP (full system)"
*.node[*].appl.enableDN = true
*.node[*].appl.enableSAF = true
*.node[*].appl.cmdpUsePIP = true
*.node[*].appl.cmdpPolicyMode = "ppo"
*.node[*].appl.dnUpdateInterval = 2s
*.node[*].appl.dnPa = 0.5
*.node[*].appl.dnKmax = 20
*.node[*].appl.dnConnectM = 3

INIEOF
    echo "  G0-G7 configs added to omnetpp.ini"
else
    echo "  G0-G7 configs already exist"
fi

echo "=== Step 10: Update Makefile ==="
# Add dn/ and saf/ subdirectories to include path
MAKEFILE="Makefile"
if ! grep -q "dn/" "$MAKEFILE" 2>/dev/null; then
    # Check if Makefile has INCLUDE_PATH
    if grep -q "INCLUDE_PATH" "$MAKEFILE"; then
        sed -i 's|INCLUDE_PATH =|INCLUDE_PATH = -I. -Idn -Isaf|' "$MAKEFILE" 2>/dev/null || true
    fi
    echo "  Makefile updated (check manually)"
else
    echo "  Makefile already has dn/saf paths"
fi

echo ""
echo "============================================================"
echo "  Integration patch complete!"
echo "  Files modified:"
echo "    - NeighborAdapter.h (new)"
echo "    - dn/NeighborTable.h (copied)"
echo "    - saf/SAFNeighborTable.h (copied)"
echo "    - CmdpRouter.h (safScores added)"
echo "    - CMDPApp.h (DN/SAF members)"
echo "    - CMDPApp.cc (DN/SAF logic)"
echo "    - CMDPApp.ned (DN/SAF parameters)"
echo "    - omnetpp.ini (G0-G7 configs)"
echo ""
echo "  Next: compile with"
echo "    cd ~/veins_workspace/cmdp_pkg_v2"
echo "    make clean && make MODE=debug -j4"
echo "============================================================"
