#ifndef SAF_RELAY_SELECTOR_H
#define SAF_RELAY_SELECTOR_H

#include "FuzzyEngine.h"
#include "SocialGraphManager.h"
#include "LinkLifetimePredictor.h"
#include "NeighborTable.h"
#include "veins/base/utils/Coord.h"
#include <vector>
#include <set>

namespace saf {

struct RelayCandidate {
    int    nodeId;
    double linkLifetime;  // T_hat
    double socialMetric;  // SM
    double fuzzyScore;    // output of fuzzy engine
    double snir;          // for tie-breaking
    double distToDst;     // for tie-breaking
};

struct SAFResult {
    int    nextHopId  = -1;
    double score      = -1.0;
    bool   directDst  = false;  // true if dst is direct neighbor
    bool   failed     = false;  // true if no candidate found
    std::vector<RelayCandidate> candidates; // for logging
};

class SAFRelaySelector {
public:
    SAFRelaySelector();

    void setEnabled(bool e) { enabled_ = e; }
    bool isEnabled() const { return enabled_; }

    void setCommRange(double r) { predictor_.setCommRange(r); }
    void setTMax(double t) { predictor_.setTMax(t); }

    /**
     * Select next hop.
     *
     * @param myId          Current node ID
     * @param myPos         Current node position
     * @param mySpeed       Current node speed
     * @param dstId         Destination node ID
     * @param dstPos        Destination position (if known; else Coord::ZERO)
     * @param neighbors     Neighbor table
     * @param socialMgr     Social graph manager
     * @param visitedSet    Set of already-visited node IDs
     * @param simTime       Current simulation time
     * @return SAFResult with selected next hop
     */
    SAFResult selectNextHop(
        int myId, const Coord& myPos, const Coord& mySpeed,
        int dstId, const Coord& dstPos,
        const NeighborTable& neighbors,
        const SocialGraphManager& socialMgr,
        const std::set<int>& visitedSet,
        double simTime) const;

    /**
     * Baseline fallback: select by minimum distance to destination
     * (used when enableSAF=false)
     */
    SAFResult selectBaselineNextHop(
        int myId, const Coord& myPos,
        int dstId, const Coord& dstPos,
        const NeighborTable& neighbors,
        const std::set<int>& visitedSet,
        double simTime) const;

    const FuzzyEngine& getFuzzyEngine() const { return fuzzy_; }

private:
    bool enabled_ = true;
    FuzzyEngine fuzzy_;
    LinkLifetimePredictor predictor_;
};

} // namespace saf

#endif // SAF_RELAY_SELECTOR_H
