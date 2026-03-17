#ifndef PIP_MASKER_H
#define PIP_MASKER_H

#include <map>
#include <set>
#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <functional>
#include "TrustManager.h"
#include "LearnedMaskWeights.h"
#include <cmath>
#include <functional>

/**
 * PIPMasker: Proactive Infeasibility Prevention (OJCOMS Eq.25-27)
 *
 * h*(v) = min trust-cost path from v to destination     (Eq.25)
 * A_feas(S_t) = { vj : rj + h*(vj) <= bt }             (Eq.27)
 *
 * Computed via reverse Dijkstra on trust-cost graph.
 * Refreshed periodically or on trust score change.
 */

struct PIPStats {
    int maskedActions = 0;
    int totalCandidates = 0;
    double dijkstraTimeMs = 0;
    int nodesExpanded = 0;
};

class PIPMasker {
public:
    PIPMasker() : destination_(-1) {}

    /**
     * Compute h*(v) for all nodes via reverse Dijkstra.
     * Edge cost: forward edge v->j costs r_j (trust cost of entering relay j).
     * Destination node costs 0.
     */
    PIPStats computeTrustToGo(
        int destination,
        const std::map<int, std::vector<int>>& adjacency,
        const TrustManager& trustMgr)
    {
        PIPStats stats;
        destination_ = destination;
        hStar_.clear();

        auto tStart = std::chrono::high_resolution_clock::now();

        // Build reverse adjacency: for forward edge v->j with cost r_j,
        // reverse edge is j->v with cost r_v (entering v in reverse direction)
        // But we want h*(v) = min sum of r for path v->dest.
        // Reverse Dijkstra from dest: edge from j to v in reverse has cost r_j
        // (the cost to enter j in the forward graph).
        // Actually simpler: treat forward cost of edge(v,j) = r_j.
        // Reverse that: revEdge(j,v) has cost r_j.
        // Then Dijkstra from dest in reverse graph gives h*(v).
        std::map<int, std::vector<std::pair<int, double>>> revAdj;
        std::set<int> allNodes;

        for (const auto& kv : adjacency) {
            int v = kv.first;
            allNodes.insert(v);
            for (int j : kv.second) {
                allNodes.insert(j);
                double costJ = (j == destination) ? 0.0 : trustMgr.computeTrustCost(j);
                revAdj[j].push_back({v, costJ});
            }
        }

        using PQEntry = std::pair<double, int>;
        std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;

        for (int n : allNodes) hStar_[n] = std::numeric_limits<double>::infinity();
        hStar_[destination] = 0.0;
        pq.push({0.0, destination});

        { FILE* _dbg = fopen("/tmp/pip_debug.txt", "a"); if(_dbg) { fprintf(_dbg, "[PIP] dst=%d adjSize=%zu revAdjSize=%zu allNodes=%zu\n", destination, adjacency.size(), revAdj.size(), allNodes.size()); fclose(_dbg); } }
        int expanded = 0;
        while (!pq.empty()) {
            auto top = pq.top(); pq.pop();
            double dist = top.first;
            int u = top.second;

            if (dist > hStar_[u]) continue;
            expanded++;

            auto rit = revAdj.find(u);
            if (rit == revAdj.end()) continue;

            for (const auto& edge : rit->second) {
                int v = edge.first;
                double w = edge.second;
                double newDist = dist + w;
                if (newDist < hStar_[v]) {
                    hStar_[v] = newDist;
                    pq.push({newDist, v});
                }
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        stats.dijkstraTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        stats.nodesExpanded = expanded;
        return stats;
    }

    void setPipAlpha(double a) { pipAlpha_ = a; }
    void setLearnedMask(bool on) { useLearnedMask_ = on; }
    void setStrictSafe(bool v) { strictSafe_ = v; }
    void setMaxDegree(int d) { maxDegree_ = std::max(1, d); }
    void setTotalBudget(double b) { totalBudget_ = b; }
    void setMaxHops(int h) { maxHops_ = h; }
    void setCurrentHop(int h) { currentHop_ = h; }
    void setAdjacency(const std::map<int, std::vector<int>>* a) { adjacency_ = a; }
    void setDistFunc(std::function<double(int,int)> f) { distFunc_ = f; }

    /** MLP inference: 7 features -> sigmoid probability */
    float mlpPredict(float input[7]) const {
        float h1[64], h2[64];
        for (int i = 0; i < 64; i++) {
            float s = fc1_b[i];
            for (int k = 0; k < 7; k++) s += fc1_w[i][k] * input[k];
            h1[i] = s > 0 ? s : 0;  // ReLU
        }
        for (int i = 0; i < 64; i++) {
            float s = fc2_b[i];
            for (int k = 0; k < 64; k++) s += fc2_w[i][k] * h1[k];
            h2[i] = s > 0 ? s : 0;
        }
        float s = fc3_b[0];
        for (int k = 0; k < 64; k++) s += fc3_w[0][k] * h2[k];
        return 1.0f / (1.0f + expf(-s));  // sigmoid
    }

    /**
     * Get feasible action set (Eq.27):
     * A_feas = { j : r_j + h*(j) <= remainingBudget }
     */
    std::vector<int> getFeasibleActions(
        int currentNode,
        double remainingBudget,
        const std::vector<int>& candidates,
        const TrustManager& trustMgr,
        PIPStats& stats) const
    {
        std::vector<int> feasible;
        stats.totalCandidates = (int)candidates.size();
        stats.maskedActions = 0;

        if (useLearnedMask_ && adjacency_) {
            // === LearnedMask mode: MLP predicts feasibility ===
            int degCur = 0;
            auto itC = adjacency_->find(currentNode);
            if (itC != adjacency_->end()) degCur = (int)itC->second.size();

            for (int j : candidates) {
                double rj = (j == destination_) ? 0.0 : trustMgr.computeTrustCost(j);
                int degJ = 0;
                double avgNbTrust = 0.5;
                auto itJ = adjacency_->find(j);
                if (itJ != adjacency_->end()) {
                    degJ = (int)itJ->second.size();
                    if (degJ > 0) {
                        double sum = 0;
                        for (int nb : itJ->second)
                            sum += trustMgr.computeTrustCost(nb);
                        avgNbTrust = sum / degJ;
                    }
                }
                double distToDst = 2000.0;  // default
                if (distFunc_) distToDst = distFunc_(j, destination_);

                float feat[7] = {
                    (float)rj,
                    (float)degJ / (float)maxDegree_,
                    (float)(distToDst / 2000.0),
                    (float)(remainingBudget / std::max(totalBudget_, 0.01)),
                    (float)currentHop_ / (float)std::max(maxHops_, 1),
                    (float)degCur / (float)maxDegree_,
                    (float)avgNbTrust
                };
                float prob = mlpPredict(feat);
                if (prob > 0.15f) {
                    feasible.push_back(j);
                } else {
                    stats.maskedActions++;
                }
            }
        } else {
            // === Exact PIP mode: h* + budget check ===
            for (int j : candidates) {
                double rj = (j == destination_) ? 0.0 : trustMgr.computeTrustCost(j);
                double hj = getHStar(j);
                if (rj + hj * pipAlpha_ <= remainingBudget + 1e-9) {
                    feasible.push_back(j);
                } else {
                    stats.maskedActions++;
                }
            }
        }

        // Repair: if mask empties feasible set, unmask best candidate
        if (feasible.empty() && !candidates.empty() && !strictSafe_) {
            if (useLearnedMask_ && adjacency_) {
                // LearnedMask repair: pick candidate with highest MLP score
                float bestProb = -1;
                int bestJ = candidates[0];
                int degCur = 0;
                auto itC = adjacency_->find(currentNode);
                if (itC != adjacency_->end()) degCur = (int)itC->second.size();
                for (int j : candidates) {
                    double rj = (j == destination_) ? 0.0 : trustMgr.computeTrustCost(j);
                    int degJ = 0; double avgNbTrust = 0.5;
                    auto itJ = adjacency_->find(j);
                    if (itJ != adjacency_->end()) {
                        degJ = (int)itJ->second.size();
                        if (degJ > 0) { double sum=0; for(int nb:itJ->second) sum+=trustMgr.computeTrustCost(nb); avgNbTrust=sum/degJ; }
                    }
                    double distToDst = distFunc_ ? distFunc_(j, destination_) : 2000.0;
                    float feat[7] = { (float)rj, (float)degJ/(float)maxDegree_,
                        (float)(distToDst/2000.0), (float)(remainingBudget/std::max(totalBudget_,0.01)),
                        (float)currentHop_/(float)std::max(maxHops_,1),
                        (float)degCur/(float)maxDegree_, (float)avgNbTrust };
                    float p = mlpPredict(feat);
                    if (p > bestProb) { bestProb = p; bestJ = j; }
                }
                feasible.push_back(bestJ);
            } else {
                double bestCost = std::numeric_limits<double>::infinity();
                int bestJ = candidates[0];
                for (int j : candidates) {
                    double rj = (j == destination_) ? 0.0 : trustMgr.computeTrustCost(j);
                    double cost = rj + getHStar(j);
                    if (cost < bestCost) { bestCost = cost; bestJ = j; }
                }
                feasible.push_back(bestJ);
            }
            stats.maskedActions--;
        }

        return feasible;
    }

    double getHStar(int nodeId) const {
        auto it = hStar_.find(nodeId);
        if (it != hStar_.end()) return it->second;
        return std::numeric_limits<double>::infinity();
    }

    bool isComputed() const { return destination_ >= 0; }

private:
    double pipAlpha_ = 1.0;
    bool useLearnedMask_ = false;
    bool strictSafe_ = false;
    int maxDegree_ = 1;
    double totalBudget_ = 2.3;
    int maxHops_ = 20;
    int currentHop_ = 0;
    const std::map<int, std::vector<int>>* adjacency_ = nullptr;
    std::function<double(int,int)> distFunc_;
    int destination_;
    std::map<int, double> hStar_;
};

#endif
