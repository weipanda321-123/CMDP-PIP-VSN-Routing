#include "veins_dn2/DNBackboneManager.h"
#include <algorithm>
#include <numeric>
#include <chrono>

namespace veins_dn2 {

void DNBackboneManager::onVehicleJoin(NeighborTable& nt, omnetpp::cRNG* rng, double now)
{
    if (!enableDN) return;
    // On first join, do one DN update
    dnUpdate(nt, rng, now);
}

void DNBackboneManager::dnUpdate(NeighborTable& nt, omnetpp::cRNG* rng, double now)
{
    auto tStart = std::chrono::high_resolution_clock::now();

    if (!enableDN) {
        // DN disabled: LNeighbors = all physical candidates (passthrough)
        LNeighbors.clear();
        auto cand = nt.getPhysicalCandidates(now);
        for (auto& c : cand) LNeighbors.insert(c.id);
        kL = (int)LNeighbors.size();
        hubFlag = false;
        auto tEnd = std::chrono::high_resolution_clock::now();
        lastUpdateDecisionMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return;
    }

    // Step 1: prune logical neighbors that are no longer physically reachable
    pruneLogicalNeighbors(nt, now);

    // Step 2: get physically feasible candidates
    auto cand = nt.getPhysicalCandidates(now);
    if (cand.empty()) {
        kL = (int)LNeighbors.size();
        computeHubFlag(nt, now);
        auto tEnd = std::chrono::high_resolution_clock::now();
        lastUpdateDecisionMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return;
    }

    // Step 3: try to fill up to dnConnectM logical edges
    int needed = dnConnectM - (int)LNeighbors.size();
    if (needed <= 0) {
        // Already at or above target
        kL = (int)LNeighbors.size();
        computeHubFlag(nt, now);
        auto tEnd = std::chrono::high_resolution_clock::now();
        lastUpdateDecisionMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return;
    }

    // Respect kmax
    int room = dnKmax - (int)LNeighbors.size();
    needed = std::min(needed, room);
    if (needed <= 0) {
        kL = (int)LNeighbors.size();
        computeHubFlag(nt, now);
        auto tEnd = std::chrono::high_resolution_clock::now();
        lastUpdateDecisionMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return;
    }

    for (int attempt = 0; attempt < needed; ++attempt) {
        double toss = omnetpp::uniform(rng, 0.0, 1.0);
        int selected = -1;
        int retries = 0;

        while (retries < retryMax) {
            if (toss < dnPa) {
                selected = selectPreferential(cand, LNeighbors, rng);
            } else {
                selected = selectRandom(cand, LNeighbors, rng);
            }
            if (selected < 0) break; // no candidates left

            // Check if target has reached its kmax (we use advertised kL as proxy)
            const NeighborEntry* ne = nt.getEntry(selected);
            if (ne && ne->logicalDegree >= dnKmax) {
                retries++;
                selected = -1;
                continue;
            }
            break;
        }

        if (selected >= 0) {
            LNeighbors.insert(selected);
            logicalEdgeCreationTime[selected] = now;
        }
    }

    kL = (int)LNeighbors.size();
    computeHubFlag(nt, now);

    auto tEnd = std::chrono::high_resolution_clock::now();
    lastUpdateDecisionMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
}

void DNBackboneManager::computeHubFlag(const NeighborTable& nt, double now)
{
    if (!enableDN) { hubFlag = false; return; }

    if (dnHubMode == "kth") {
        hubFlag = (kL >= dnHubKth);
    } else if (dnHubMode == "topAlpha") {
        // Collect all neighbor kL values + own kL
        std::vector<int> allKL;
        allKL.push_back(kL);
        auto cand = nt.getPhysicalCandidates(now);
        for (auto& c : cand) {
            allKL.push_back(c.logicalDegree);
        }
        std::sort(allKL.begin(), allKL.end(), std::greater<int>());
        int topIdx = std::max(1, (int)(dnHubAlpha * allKL.size()));
        int threshold = (topIdx <= (int)allKL.size()) ? allKL[topIdx - 1] : 0;
        hubFlag = (kL >= threshold);
    } else {
        hubFlag = (kL >= dnHubKth); // fallback
    }
}

void DNBackboneManager::removeLogicalNeighbor(int id)
{
    LNeighbors.erase(id);
    logicalEdgeCreationTime.erase(id);
    kL = (int)LNeighbors.size();
}

void DNBackboneManager::pruneLogicalNeighbors(const NeighborTable& nt, double now)
{
    std::vector<int> toRemove;
    for (int nid : LNeighbors) {
        if (!nt.isAlive(nid, now)) {
            toRemove.push_back(nid);
        }
    }
    for (int nid : toRemove) {
        removeLogicalNeighbor(nid);
    }
}

std::vector<int> DNBackboneManager::getCandidateNextHops(const NeighborTable& nt, double now) const
{
    if (!enableDN || dnRoutingMode == "overlayOnly") {
        // Only logical neighbors
        return std::vector<int>(LNeighbors.begin(), LNeighbors.end());
    }
    // overlayPreferred: return all physical candidates, but caller should
    // weight LNeighbors / hubs higher
    auto cand = nt.getPhysicalCandidates(now);
    std::vector<int> result;
    // Logical neighbors first (preferred)
    for (int nid : LNeighbors) {
        if (nt.isAlive(nid, now)) result.push_back(nid);
    }
    // Then remaining physical neighbors
    for (auto& c : cand) {
        if (LNeighbors.count(c.id) == 0) {
            result.push_back(c.id);
        }
    }
    return result;
}

int DNBackboneManager::selectPreferential(const std::vector<NeighborEntry>& cand,
                                           const std::set<int>& exclude,
                                           omnetpp::cRNG* rng) const
{
    // Build weight vector: w_i = kL_i + k0
    std::vector<double> weights;
    std::vector<int> ids;

    for (auto& c : cand) {
        if (exclude.count(c.id)) continue;
        
        // Base weight: preferential attachment (degree)
        double w = c.logicalDegree + k0;
        
        // Quality factor 1: link stability (penalize high relative speed)
        double relVx = c.vx - mySpeedX;
        double relVy = c.vy - mySpeedY;
        double relSpeed = std::sqrt(relVx*relVx + relVy*relVy);
        // stability: 1.0 when relSpeed=0, decays to 0.2 at dnSpeedDiffMax
        double stability = std::max(0.2, 1.0 - 0.8 * relSpeed / std::max(1.0, dnSpeedDiffMax));
        
        // Quality factor 2: trust/social forwarding willingness
        double trustFactor = std::max(0.3, c.trustScore);  // floor at 0.3
        
        // Combined quality-aware weight
        w *= stability * trustFactor;
        
        // Hard filter: skip very unstable or untrustworthy
        if (relSpeed > dnSpeedDiffMax * 1.5 || c.trustScore < dnTrustMin) continue;
        
        weights.push_back(w);
        ids.push_back(c.id);
    }

    if (ids.empty()) return -1;

    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total <= 0) {
        // Fallback: uniform
        int idx = (int)omnetpp::intuniform(rng, 0, (int)ids.size() - 1);
        return ids[idx];
    }

    double r = omnetpp::uniform(rng, 0.0, total);
    double cum = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        cum += weights[i];
        if (r <= cum) return ids[i];
    }
    return ids.back();
}

int DNBackboneManager::selectRandom(const std::vector<NeighborEntry>& cand,
                                     const std::set<int>& exclude,
                                     omnetpp::cRNG* rng) const
{
    std::vector<int> ids;
    for (auto& c : cand) {
        if (exclude.count(c.id)) continue;
        ids.push_back(c.id);
    }
    if (ids.empty()) return -1;
    int idx = (int)omnetpp::intuniform(rng, 0, (int)ids.size() - 1);
    return ids[idx];
}

} // namespace veins_dn2
