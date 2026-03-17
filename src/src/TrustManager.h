#ifndef TRUST_MANAGER_H
#define TRUST_MANAGER_H

#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>

/**
 * TrustManager: per-vehicle social trustworthiness (OJCOMS Eq.11-14)
 *
 *   p_i^his  = (1 + h1) / (2 + h1 + h2)            -- Eq.11
 *   p_i^cur  = mean{ p_trust_{i,k}(t) : k in N(i) } -- Eq.12
 *   p_i^trust = alpha1 * p_i^his + alpha2 * p_i^cur  -- Eq.14
 *   r_i       = -log(p_i^trust)                      -- trust cost
 *   B         = -log(P_th)                            -- budget
 *
 * Cold-start fix (SAF lesson): pre-inject trust for group vehicles
 * to avoid all vehicles having identical trust = 0.5 at t=0.
 */

struct RelayRecord {
    int successCount = 0;   // h1
    int refusalCount = 0;   // h2
};

struct PairwiseTrust {
    int h1 = 0;
    int h2 = 0;
};

class TrustManager {
public:
    TrustManager() : alpha1_(0.5), alpha2_(0.5), pThreshold_(0.1) {}

    void initialize(double alpha1, double alpha2, double pThreshold) {
        alpha1_ = alpha1;
        alpha2_ = alpha2;
        double sum = alpha1_ + alpha2_;
        if (sum > 0) { alpha1_ /= sum; alpha2_ /= sum; }
        pThreshold_ = pThreshold;
    }

    void registerVehicle(int vid) {
        if (relayRecords_.find(vid) == relayRecords_.end())
            relayRecords_[vid] = RelayRecord();
    }

    void unregisterVehicle(int vid) {
        relayRecords_.erase(vid);
        pairwiseTrust_.erase(vid);
        for (auto& kv : pairwiseTrust_) kv.second.erase(vid);
        trustOverrides_.erase(vid);
    }

    void recordRelaySuccess(int vid) {
        registerVehicle(vid);
        relayRecords_[vid].successCount++;
    }

    void recordRelayFailure(int vid) {
        registerVehicle(vid);
        relayRecords_[vid].refusalCount++;
    }

    void recordPairwiseSuccess(int fromV, int forV) {
        pairwiseTrust_[forV][fromV].h1++;
    }

    void recordPairwiseFailure(int fromV, int forV) {
        pairwiseTrust_[forV][fromV].h2++;
    }

    /** p_i^his (Eq.11) */
    double computeHistoricalTrust(int vid) const {
        auto it = relayRecords_.find(vid);
        if (it == relayRecords_.end()) return 0.5;
        double h1 = it->second.successCount;
        double h2 = it->second.refusalCount;
        return (1.0 + h1) / (2.0 + h1 + h2);
    }

    /** p_i^cur (Eq.12-13) */
    double computeCurrentTrust(int vid) const {
        auto itI = pairwiseTrust_.find(vid);
        if (itI == pairwiseTrust_.end() || itI->second.empty()) return 0.5;
        double sum = 0.0;
        int count = 0;
        for (const auto& kv : itI->second) {
            double h1k = kv.second.h1;
            double h2k = kv.second.h2;
            sum += (1.0 + h1k) / (2.0 + h1k + h2k);
            count++;
        }
        return (count > 0) ? (sum / count) : 0.5;
    }

    /** p_i^trust (Eq.14) */
    double computeTrust(int vid) const {
        auto ov = trustOverrides_.find(vid);
        if (ov != trustOverrides_.end()) return std::max(ov->second, 1e-6);
        double pHis = computeHistoricalTrust(vid);
        double pCur = computeCurrentTrust(vid);
        return std::max(alpha1_ * pHis + alpha2_ * pCur, 1e-6);
    }

    /** r_i = -log(p_i^trust) */
    double computeTrustCost(int vid) const {
        return -std::log(computeTrust(vid));
    }

    /** B = -log(P_th) */
    double getBudget() const {
        return -std::log(std::max(pThreshold_, 1e-12));
    }

    double getThreshold() const { return pThreshold_; }

    std::vector<int> getRegisteredVehicles() const {
        std::vector<int> ids;
        for (const auto& kv : relayRecords_) ids.push_back(kv.first);
        return ids;
    }

    /**
     * Pre-inject trust values to avoid cold-start problem.
     * (SAF lesson: "cold start must be designed upfront")
     *
     * Same-group vehicles: high trust (h1=5, h2=0 -> p_his=0.857)
     * Cross-group vehicles: low trust  (h1=1, h2=3 -> p_his=0.400)
     * Unknown vehicles: default        (h1=0, h2=0 -> p_his=0.500)
     */
    void preInjectTrust(int myId, int totalVehicles, int numGroups) {
        if (totalVehicles <= 0 || numGroups <= 0) return;
        int groupSize = totalVehicles / numGroups;
        int myGroup = myId / groupSize;

        int injected = 0;
        // Same-group: high trust
        int myGrpStart = myGroup * groupSize;
        int myGrpEnd = std::min(myGrpStart + groupSize, totalVehicles);
        for (int j = myGrpStart; j < myGrpEnd && injected < 30; j++) {
            if (j == myId) continue;
            registerVehicle(j);
            for (int k = 0; k < 5; k++) recordRelaySuccess(j);
            injected++;
        }

        // Cross-group: low trust (10% sample)
        for (int j = 0; j < totalVehicles; j++) {
            int jGroup = j / groupSize;
            if (jGroup == myGroup || j == myId) continue;
            if (j % 10 == 0) {
                registerVehicle(j);
                recordRelaySuccess(j);
                for (int k = 0; k < 3; k++) recordRelayFailure(j);
            }
        }
    }

    void setTrustOverride(int vid, double val) { trustOverrides_[vid] = val; }

private:
    double alpha1_, alpha2_, pThreshold_;
    std::map<int, RelayRecord> relayRecords_;
    std::map<int, std::map<int, PairwiseTrust>> pairwiseTrust_;
    std::map<int, double> trustOverrides_;
};

#endif
