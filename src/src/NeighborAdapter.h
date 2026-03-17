#ifndef NEIGHBOR_ADAPTER_H
#define NEIGHBOR_ADAPTER_H

/**
 * NeighborAdapter: bridges CMDP NeighborTable with DN and SAF subsystems.
 */

#include "NeighborTable.h"
#include "NeighborTable.h"
#include "dn/DNBackboneManager.h"
#include "veins/base/utils/Coord.h"
#include <cmath>
#include <map>
#include <vector>

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
            // Pass trust score to DN entry
            auto* dnEntry = const_cast<veins_dn2::NeighborEntry*>(dnTable.getEntry(e.vehicleId));
            if (dnEntry) dnEntry->trustScore = e.trustScore;
        }
        dnTable.prune(now);
    }
};



// ============================================================
//  DN Filter: backbone candidate filtering for CmdpRouter
// ============================================================
class DNFilter {
public:
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

        // overlayPreferred fallback
        if (result.empty() && dn.dnRoutingMode == "overlayPreferred") {
            for (auto& e : entries) result.push_back(e.vehicleId);
        }

        return result;
    }

    // Pareto DN: compute multi-dimensional quality score for ALL neighbors
    static std::map<int, double> computeDNQuality(
        const NeighborTable& cmdp,
        const veins_dn2::DNBackboneManager& dn,
        double mySpeedX, double mySpeedY)
    {
        std::map<int, double> scores;
        auto entries = cmdp.getAllEntries();
        if (entries.empty()) return scores;
        for (auto& e : entries) {
            double hubScore = dn.isLogicalNeighbor(e.vehicleId) ? 0.9 : 0.4;
            double relVx = e.speedX - mySpeedX;
            double relVy = e.speedY - mySpeedY;
            double relSpeed = std::sqrt(relVx*relVx + relVy*relVy);
            double stabilityScore = std::max(0.1, 1.0 - relSpeed / 30.0);
            double trustScore = std::max(0.1, std::min(1.0, e.trustScore));
            double quality = std::pow(hubScore, 0.4)
                           * std::pow(stabilityScore, 0.3)
                           * std::pow(trustScore, 0.3);
            scores[e.vehicleId] = quality;
        }
        return scores;
    }
};

#endif
