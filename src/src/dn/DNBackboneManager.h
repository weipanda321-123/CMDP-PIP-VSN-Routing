#ifndef VEINS_DN2_DNBACKBONEMANAGER_H_
#define VEINS_DN2_DNBACKBONEMANAGER_H_

#include <set>
#include <vector>
#include <string>
#include <omnetpp.h>
#include "../NeighborTable.h"

namespace veins_dn2 {

/**
 * DNBackboneManager: one instance per vehicle.
 * Maintains logical overlay neighbors (LNeighbors), computes hub flag,
 * and provides candidate next-hop API for upper-layer routing.
 */
class DNBackboneManager {
public:
    // ---- configuration (set by DNApp from ini) ----
    bool   enableDN         = true;
    double dnUpdateInterval = 2.0;   // ΔT seconds
    double dnPa             = 0.5;   // preferential attachment probability
    int    dnKmax           = 20;    // max logical degree per vehicle
    int    dnConnectM       = 3;     // target logical edges per update
    double dnNeighborTimeout = 3.0;
    bool   dnUseSnirGate    = false;
    double dnSnirThreshold  = 5.0;   // dB
    std::string dnHubMode   = "kth"; // "kth" or "topAlpha"
    int    dnHubKth         = 8;
    double dnHubAlpha       = 0.1;   // top 10%
    double k0               = 1.0;   // additive constant for preferential attachment
    int    retryMax          = 5;

    // ---- quality-aware DN (Chapter 6 enhancement) ----
    double mySpeedX = 0, mySpeedY = 0;  // set by CMDPApp before dnUpdate
    double dnSpeedDiffMax = 15.0;       // m/s, penalize relative speed above this
    double dnTrustMin = 0.3;            // minimum trust for candidate selection

    // ---- routing mode ----
    std::string dnRoutingMode = "overlayPreferred"; // "overlayOnly" or "overlayPreferred"

    // ---- state ----
    std::set<int> LNeighbors;   // logical neighbor ids
    int           kL = 0;       // logical degree
    bool          hubFlag = false;

    // ---- statistics ----
    double lastUpdateDecisionMs = 0; // decision computation time (ms)

    // ---- methods ----
    DNBackboneManager() {}

    /** Called when vehicle first joins the network. */
    void onVehicleJoin(NeighborTable& nt, omnetpp::cRNG* rng, double now);

    /** Periodic DN update (called every ΔT). */
    void dnUpdate(NeighborTable& nt, omnetpp::cRNG* rng, double now);

    /** Compute hub flag based on current kL and neighbor info. */
    void computeHubFlag(const NeighborTable& nt, double now);

    /** Remove a specific logical neighbor (e.g., on timeout or departure). */
    void removeLogicalNeighbor(int id);

    /** Clean logical neighbors that are no longer physically reachable. */
    void pruneLogicalNeighbors(const NeighborTable& nt, double now);

    /** Get candidate next hops for routing based on dnRoutingMode. */
    std::vector<int> getCandidateNextHops(const NeighborTable& nt, double now) const;

    /** Check if a specific id is a logical neighbor. */
    bool isLogicalNeighbor(int id) const { return LNeighbors.count(id) > 0; }

    int getLogicalDegree() const { return kL; }
    bool getHubFlag() const { return hubFlag; }

    // Track logical edge creation times for lifetime stats
    std::map<int, double> logicalEdgeCreationTime;

private:
    /** Select a neighbor by preferential attachment. */
    int selectPreferential(const std::vector<NeighborEntry>& cand,
                           const std::set<int>& exclude,
                           omnetpp::cRNG* rng) const;

    /** Select a neighbor uniformly at random. */
    int selectRandom(const std::vector<NeighborEntry>& cand,
                     const std::set<int>& exclude,
                     omnetpp::cRNG* rng) const;
};

} // namespace veins_dn2

#endif
