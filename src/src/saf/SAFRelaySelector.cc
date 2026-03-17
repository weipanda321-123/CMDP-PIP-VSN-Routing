#include "SAFRelaySelector.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace saf {

SAFRelaySelector::SAFRelaySelector() {}

SAFResult SAFRelaySelector::selectNextHop(
    int myId, const Coord& myPos, const Coord& mySpeed,
    int dstId, const Coord& dstPos,
    const NeighborTable& neighbors,
    const SocialGraphManager& socialMgr,
    const std::set<int>& visitedSet,
    double simTime) const
{
    SAFResult result;
    auto nbrIds = neighbors.getNeighborIds();

    // Check if destination is a direct neighbor
    if (neighbors.has(dstId)) {
        result.nextHopId = dstId;
        result.directDst = true;
        result.score = 1.0;
        return result;
    }

    // Build candidate list: exclude only recently visited (last 3 hops)
    // TTL handles loop termination; visited only prevents immediate backtrack
    std::vector<int> visitedVec(visitedSet.begin(), visitedSet.end());
    std::set<int> recentVisited;
    int recentN = std::min((int)visitedVec.size(), 3);
    for (int i = (int)visitedVec.size() - recentN; i < (int)visitedVec.size(); i++)
        recentVisited.insert(visitedVec[i]);

    for (int nid : nbrIds) {
        if (nid == myId) continue;
        if (recentVisited.count(nid)) continue;

        // SINR threshold filter (Algorithm 1 line 9: SINR >= gamma_0)
        auto* ne_check = neighbors.get(nid);
        if (ne_check && ne_check->snir > 0.01) {
            double snir_dB = 10.0 * std::log10(ne_check->snir);
            if (snir_dB < -13.0) continue;  // gamma_0 = -13 dB
        }

        const NeighborEntry* ne = neighbors.get(nid);
        if (!ne) continue;

        RelayCandidate rc;
        rc.nodeId = nid;

        // T_hat: predicted link lifetime
        rc.linkLifetime = predictor_.predict(myPos, mySpeed,
                                              veins::Coord(ne->posX, ne->posY, 0), veins::Coord(ne->speedX, ne->speedY, 0));

        // SM: social relationship metric
        rc.socialMetric = socialMgr.getSocialMetric(myId, nid);

        // Fuzzy score
        rc.fuzzyScore = fuzzy_.evaluate(rc.linkLifetime, rc.socialMetric);

        rc.snir = ne->snir;

        // Distance to destination (tie-breaking only, per Algorithm 1)
        if (dstPos.x != 0.0 || dstPos.y != 0.0) {
            double dx = ne->posX - dstPos.x;
            double dy = ne->posY - dstPos.y;
            rc.distToDst = std::sqrt(dx * dx + dy * dy);
        } else {
            rc.distToDst = std::numeric_limits<double>::max();
        }
        result.candidates.push_back(rc);
    }

    if (result.candidates.empty()) {
        result.failed = true;
        return result;
    }

    // Path-product approximation (paper Section V-F: max product of PV)
    double myDistToDst = std::numeric_limits<double>::max();
    if (dstPos.x != 0.0 || dstPos.y != 0.0) {
        double dx = myPos.x - dstPos.x;
        double dy = myPos.y - dstPos.y;
        myDistToDst = std::sqrt(dx * dx + dy * dy);
    }
    for (auto& rc : result.candidates) {
        double gp = 0.5;
        if (myDistToDst > 1.0 && rc.distToDst < std::numeric_limits<double>::max()) {
            double progressRatio = (myDistToDst - rc.distToDst) / myDistToDst;
            gp = std::max(0.1, std::min(1.0, 0.5 + 0.5 * progressRatio));
        }
        rc.fuzzyScore = rc.fuzzyScore * gp;
    }

    // Sort by pathScore DESC
    std::sort(result.candidates.begin(), result.candidates.end(),
        [](const RelayCandidate& a, const RelayCandidate& b) {
            if (std::abs(a.fuzzyScore - b.fuzzyScore) > 1e-6)
                return a.fuzzyScore > b.fuzzyScore;
            if (std::abs(a.snir - b.snir) > 0.1)
                return a.snir > b.snir;
            return a.distToDst < b.distToDst;
        });

    result.nextHopId = result.candidates[0].nodeId;
    result.score = result.candidates[0].fuzzyScore;
    return result;
}

SAFResult SAFRelaySelector::selectBaselineNextHop(
    int myId, const Coord& myPos,
    int dstId, const Coord& dstPos,
    const NeighborTable& neighbors,
    const std::set<int>& visitedSet,
    double simTime) const
{
    SAFResult result;
    auto nbrIds = neighbors.getNeighborIds();

    // Direct delivery check
    if (neighbors.has(dstId)) {
        result.nextHopId = dstId;
        result.directDst = true;
        result.score = 1.0;
        return result;
    }

    // Baseline: greedy geographic forwarding (closest to destination)
    double bestDist = std::numeric_limits<double>::max();
    int bestId = -1;

    std::vector<int> visitedVec2(visitedSet.begin(), visitedSet.end());
    std::set<int> recentVisited2;
    int recentN2 = std::min((int)visitedVec2.size(), 3);
    for (int i = (int)visitedVec2.size() - recentN2; i < (int)visitedVec2.size(); i++)
        recentVisited2.insert(visitedVec2[i]);

    for (int nid : nbrIds) {
        if (nid == myId) continue;
        if (recentVisited2.count(nid)) continue;

        // SINR threshold filter (same as SAF path for fair comparison)
        auto* ne_chk2 = neighbors.get(nid);
        if (ne_chk2 && ne_chk2->snir > 0.01) {
            double snir_dB2 = 10.0 * std::log10(ne_chk2->snir);
            if (snir_dB2 < -13.0) continue;
        }

        const NeighborEntry* ne = neighbors.get(nid);
        if (!ne) continue;

        double dx = ne->posX - dstPos.x;
        double dy = ne->posY - dstPos.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        // Geographic progress filter
        if (dstPos.x != 0.0 || dstPos.y != 0.0) {
            double myDx = myPos.x - dstPos.x;
            double myDy = myPos.y - dstPos.y;
            double myDist = std::sqrt(myDx * myDx + myDy * myDy);
            if (dist >= myDist) continue;
        }

        if (dist < bestDist) {
            bestDist = dist;
            bestId = nid;
        }
    }

    if (bestId < 0) {
        result.failed = true;
    } else {
        result.nextHopId = bestId;
        result.score = 0.0; // no fuzzy score in baseline
    }
    return result;
}

} // namespace saf
