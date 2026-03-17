// ===========================================================================
//  CmdpBaselineAlgorithms.h — DP / KSP / B&B / GA for OJCOMS density-sweep
//  Matched to actual project types: GraphSnapshot, RouteResult
//  API: TrustManager::computeTrustCost(vid)
//       DelayEstimator::predictDelayByDistance(dist)
//       GraphSnapshot::adjacency, nodeSet, positions
//       RouteResult::path (vector<int>)
// ===========================================================================
#ifndef CMDP_BASELINE_ALGORITHMS_H
#define CMDP_BASELINE_ALGORITHMS_H

#include <vector>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <functional>
#include <numeric>

// Helper: Euclidean distance between two nodes using GraphSnapshot positions
inline double baselineNodeDist(const GraphSnapshot& g, int u, int v) {
    auto iu = g.positions.find(u);
    auto iv = g.positions.find(v);
    if (iu == g.positions.end() || iv == g.positions.end()) return 300.0;
    double dx = iu->second.first - iv->second.first;
    double dy = iu->second.second - iv->second.second;
    return std::sqrt(dx*dx + dy*dy);
}

// =====================================================================
//  1. Dynamic Programming (resource-discretized)
//     O(|E| × Bbins), default Bbins=200
// =====================================================================
inline RouteResult solveDPResource(
    const GraphSnapshot& graph,
    int src, int dst, double budgetB,
    const std::function<double(int)>& getTrustCost,
    const std::function<double(int,int)>& getDelay,
    int Bbins = 200)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RouteResult result;
    result.policyMode = "DPBaseline";

    int N = (int)graph.nodeSet.size();
    if (N > 100) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        return result;
    }

    std::vector<int> nodes(graph.nodeSet.begin(), graph.nodeSet.end());
    std::map<int, int> nodeIdx;
    for (int i = 0; i < N; i++) nodeIdx[nodes[i]] = i;

    if (!nodeIdx.count(src) || !nodeIdx.count(dst)) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        return result;
    }

    int srcI = nodeIdx[src], dstI = nodeIdx[dst];
    const double INF = 1e18;

    auto toBin = [&](double r) -> int {
        return std::max(0, std::min((int)std::round(r / budgetB * Bbins), Bbins));
    };

    // dp[i][b] = min delay from node i to dst with budget bin b remaining
    std::vector<std::vector<double>> dp(N, std::vector<double>(Bbins + 1, INF));
    std::vector<std::vector<int>> pred(N, std::vector<int>(Bbins + 1, -1));
    for (int b = 0; b <= Bbins; b++) dp[dstI][b] = 0.0;

    bool changed = true;
    for (int iter = 0; iter < N && changed; iter++) {
        changed = false;
        for (int ii = 0; ii < N; ii++) {
            if (ii == dstI) continue;
            int u = nodes[ii];
            auto adjIt = graph.adjacency.find(u);
            if (adjIt == graph.adjacency.end()) continue;
            for (int v : adjIt->second) {
                if (!nodeIdx.count(v)) continue;
                int jj = nodeIdx[v];
                double r_v = getTrustCost(v);
                int bCost = toBin(r_v);
                double d_uv = getDelay(u, v);
                for (int b = bCost; b <= Bbins; b++) {
                    double nc = d_uv + dp[jj][b - bCost];
                    if (nc < dp[ii][b] - 1e-12) {
                        dp[ii][b] = nc;
                        pred[ii][b] = jj;
                        changed = true;
                    }
                }
            }
        }
    }

    if (dp[srcI][Bbins] < INF / 2) {
        result.totalDelay = dp[srcI][Bbins];
        result.reachable = true;
        int cur = srcI, b = Bbins;
        result.path.push_back(nodes[cur]);
        double trustAcc = 0.0;
        while (cur != dstI && pred[cur][b] >= 0) {
            int nxt = pred[cur][b];
            double r = getTrustCost(nodes[nxt]);
            trustAcc += r;
            b -= toBin(r);
            cur = nxt;
            result.path.push_back(nodes[cur]);
            if ((int)result.path.size() > N + 1) break;
        }
        result.totalTrustCost = trustAcc;
        result.feasible = (trustAcc <= budgetB + 1e-9) && (cur == dstI);
        result.reachable = (cur == dstI);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
    return result;
}

// =====================================================================
//  2. K-Shortest Paths (Yen's algorithm), K=20
// =====================================================================
inline RouteResult solveKShortestPaths(
    const GraphSnapshot& graph,
    int src, int dst, double budgetB,
    const std::function<double(int)>& getTrustCost,
    const std::function<double(int,int)>& getDelay,
    int K = 20)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RouteResult result;
    result.policyMode = "KSPBaseline";
    const double INF = 1e18;

    auto dijkstra = [&](int s, int d,
                        const std::set<int>& blockedN,
                        const std::set<std::pair<int,int>>& blockedE)
        -> std::pair<double, std::vector<int>>
    {
        std::map<int, double> dist;
        std::map<int, int> prev;
        using PQ = std::priority_queue<std::pair<double,int>,
            std::vector<std::pair<double,int>>, std::greater<std::pair<double,int>>>;
        PQ pq;
        dist[s] = 0; pq.push({0, s});
        while (!pq.empty()) {
            auto top = pq.top(); pq.pop();
            double dd = top.first; int u = top.second;
            if (dd > dist[u] + 1e-12) continue;
            if (u == d) break;
            auto it = graph.adjacency.find(u);
            if (it == graph.adjacency.end()) continue;
            for (int v : it->second) {
                if (blockedN.count(v) || blockedE.count(std::make_pair(u,v))) continue;
                double w = getDelay(u, v);
                double nd = dist[u] + w;
                if (!dist.count(v) || nd < dist[v] - 1e-12) {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.push({nd, v});
                }
            }
        }
        if (!dist.count(d) || dist[d] >= INF/2)
            return std::make_pair(INF, std::vector<int>());
        std::vector<int> path;
        int c = d;
        while (c != s && prev.count(c)) { path.push_back(c); c = prev[c]; }
        path.push_back(s);
        std::reverse(path.begin(), path.end());
        if (path.front() != s) return std::make_pair(INF, std::vector<int>());
        return std::make_pair(dist[d], path);
    };

    std::set<int> emptyN;
    std::set<std::pair<int,int>> emptyE;
    auto first = dijkstra(src, dst, emptyN, emptyE);
    if (first.second.empty()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        return result;
    }

    std::vector<std::pair<double, std::vector<int>>> A;
    A.push_back(first);
    std::set<std::vector<int>> seen;
    seen.insert(first.second);

    auto cmp = [](const std::pair<double,std::vector<int>>& a,
                  const std::pair<double,std::vector<int>>& b) {
        return a.first > b.first;
    };
    std::priority_queue<std::pair<double,std::vector<int>>,
        std::vector<std::pair<double,std::vector<int>>>, decltype(cmp)> B(cmp);

    for (int k = 1; k < K; k++) {
        const std::vector<int>& prevPath = A.back().second;
        for (int i = 0; i < (int)prevPath.size() - 1; i++) {
            int spurNode = prevPath[i];
            std::vector<int> root(prevPath.begin(), prevPath.begin() + i + 1);

            std::set<std::pair<int,int>> bEdges;
            for (size_t ai = 0; ai < A.size(); ai++) {
                const std::vector<int>& p = A[ai].second;
                if ((int)p.size() > i + 1) {
                    bool match = true;
                    for (int j = 0; j <= i && match; j++)
                        if (p[j] != root[j]) match = false;
                    if (match)
                        bEdges.insert(std::make_pair(p[i], p[i+1]));
                }
            }
            std::set<int> bNodes;
            for (int j = 0; j < i; j++) bNodes.insert(root[j]);

            auto spur = dijkstra(spurNode, dst, bNodes, bEdges);
            if (spur.second.empty()) continue;

            std::vector<int> total = root;
            total.insert(total.end(), spur.second.begin()+1, spur.second.end());
            std::set<int> uniq(total.begin(), total.end());
            if ((int)uniq.size() != (int)total.size()) continue;

            if (!seen.count(total)) {
                double td = 0;
                for (int j = 0; j < (int)total.size()-1; j++)
                    td += getDelay(total[j], total[j+1]);
                B.push(std::make_pair(td, total));
                seen.insert(total);
            }
        }
        if (B.empty()) break;
        A.push_back(B.top()); B.pop();
    }

    double bestD = INF;
    for (size_t ai = 0; ai < A.size(); ai++) {
        const std::vector<int>& path = A[ai].second;
        double delay = A[ai].first;
        double trust = 0;
        for (int i = 1; i < (int)path.size(); i++)
            trust += getTrustCost(path[i]);
        if (trust <= budgetB + 1e-9 && delay < bestD) {
            bestD = delay;
            result.path = path;
            result.totalDelay = delay;
            result.totalTrustCost = trust;
            result.feasible = true;
            result.reachable = true;
        }
    }
    if (result.path.empty() && !A.empty()) {
        const std::vector<int>& path = A[0].second;
        double trust = 0;
        for (int i = 1; i < (int)path.size(); i++)
            trust += getTrustCost(path[i]);
        result.path = path;
        result.totalDelay = A[0].first;
        result.totalTrustCost = trust;
        result.reachable = true;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
    return result;
}

// =====================================================================
//  3. Branch & Bound with Lagrangian relaxation
// =====================================================================
inline RouteResult solveBranchAndBound(
    const GraphSnapshot& graph,
    int src, int dst, double budgetB,
    const std::function<double(int)>& getTrustCost,
    const std::function<double(int,int)>& getDelay,
    int nodeLimit = 100000,
    double timeLimitMs = 1000.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RouteResult result;
    result.policyMode = "BranchBound";
    const double INF = 1e18;
    double lambda = 1.0;

    // Reverse Dijkstra for h*(v) lower bounds
    std::map<int, double> hStar;
    {
        std::map<int, std::vector<std::pair<int,double>>> revAdj;
        for (auto it = graph.adjacency.begin(); it != graph.adjacency.end(); ++it) {
            int u = it->first;
            for (int v : it->second) {
                double w = getDelay(u, v) + lambda * getTrustCost(v);
                revAdj[v].push_back(std::make_pair(u, w));
            }
        }
        hStar[dst] = 0;
        using PQ = std::priority_queue<std::pair<double,int>,
            std::vector<std::pair<double,int>>, std::greater<std::pair<double,int>>>;
        PQ pq;
        pq.push(std::make_pair(0.0, dst));
        while (!pq.empty()) {
            double dd = pq.top().first; int u = pq.top().second; pq.pop();
            if (dd > hStar[u] + 1e-12) continue;
            auto rit = revAdj.find(u);
            if (rit == revAdj.end()) continue;
            for (size_t i = 0; i < rit->second.size(); i++) {
                int w_node = rit->second[i].first;
                double w_cost = rit->second[i].second;
                double nh = w_cost + hStar[u];
                if (!hStar.count(w_node) || nh < hStar[w_node] - 1e-12) {
                    hStar[w_node] = nh;
                    pq.push(std::make_pair(nh, w_node));
                }
            }
        }
    }

    if (!hStar.count(src)) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        return result;
    }

    struct BBNode {
        std::vector<int> path;
        double delay, trust, lb;
        bool operator>(const BBNode& o) const { return lb > o.lb; }
    };
    std::priority_queue<BBNode, std::vector<BBNode>, std::greater<BBNode>> pq;
    double bestUB = INF;
    std::vector<int> bestPath;
    BBNode startNode; startNode.path = {src}; startNode.delay = 0; startNode.trust = 0; startNode.lb = hStar[src];
    pq.push(startNode);
    int explored = 0;

    while (!pq.empty() && explored < nodeLimit) {
        double elapsed = std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        if (elapsed > timeLimitMs) break;

        BBNode cur = pq.top(); pq.pop();
        explored++;
        int last = cur.path.back();

        if (last == dst) {
            if (cur.trust <= budgetB + 1e-9 && cur.delay < bestUB) {
                bestUB = cur.delay;
                bestPath = cur.path;
            }
            continue;
        }
        if (cur.lb >= bestUB - 1e-12) continue;

        auto it = graph.adjacency.find(last);
        if (it == graph.adjacency.end()) continue;
        std::set<int> visited(cur.path.begin(), cur.path.end());

        for (int j : it->second) {
            if (visited.count(j)) continue;
            double nd = cur.delay + getDelay(last, j);
            double nt = cur.trust + getTrustCost(j);
            if (nt > budgetB + 1e-9) continue;
            double h = hStar.count(j) ? hStar[j] : INF;
            double lb = nd + (j == dst ? 0 : h);
            if (lb >= bestUB - 1e-12) continue;
            BBNode child;
            child.path = cur.path;
            child.path.push_back(j);
            child.delay = nd; child.trust = nt; child.lb = lb;
            pq.push(child);
        }
    }

    if (!bestPath.empty()) {
        double td = 0, tt = 0;
        for (int i = 0; i < (int)bestPath.size()-1; i++) {
            td += getDelay(bestPath[i], bestPath[i+1]);
            tt += getTrustCost(bestPath[i+1]);
        }
        result.path = bestPath;
        result.totalDelay = td;
        result.totalTrustCost = tt;
        result.feasible = (tt <= budgetB + 1e-9);
        result.reachable = true;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
    return result;
}

// =====================================================================
//  4. Genetic Algorithm — P=100, G=200, mutation=0.1
// =====================================================================
inline RouteResult solveGeneticAlgorithm(
    const GraphSnapshot& graph,
    int src, int dst, double budgetB,
    const std::function<double(int)>& getTrustCost,
    const std::function<double(int,int)>& getDelay,
    int popSize = 100,
    int nGen = 200,
    double mutRate = 0.1,
    double lambdaPen = 10.0)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    RouteResult result;
    result.policyMode = "GABaseline";
    std::mt19937 rng(42);

    auto bfs = [&](int s, int d) -> std::vector<int> {
        std::queue<int> q; std::map<int,int> prev;
        prev[s] = -1; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == d) break;
            auto it = graph.adjacency.find(u);
            if (it == graph.adjacency.end()) continue;
            for (int v : it->second)
                if (!prev.count(v)) { prev[v] = u; q.push(v); }
        }
        if (!prev.count(d)) return {};
        std::vector<int> p;
        for (int c = d; c != -1; c = prev[c]) p.push_back(c);
        std::reverse(p.begin(), p.end());
        return p;
    };

    auto randPath = [&](int s, int d) -> std::vector<int> {
        std::vector<int> path; path.push_back(s);
        std::set<int> vis; vis.insert(s);
        int cur = s;
        int maxSteps = (int)graph.nodeSet.size() * 2;
        for (int step = 0; step < maxSteps; step++) {
            if (cur == d) return path;
            auto it = graph.adjacency.find(cur);
            if (it == graph.adjacency.end()) break;
            std::vector<int> cands;
            for (int j : it->second) if (!vis.count(j)) cands.push_back(j);
            if (cands.empty()) break;
            int nxt = cands[std::uniform_int_distribution<int>(0,(int)cands.size()-1)(rng)];
            path.push_back(nxt); vis.insert(nxt); cur = nxt;
        }
        return {};
    };

    auto fitness = [&](const std::vector<int>& p) -> double {
        if (p.empty() || p.front() != src || p.back() != dst) return 1e18;
        double d = 0, t = 0;
        for (int i = 0; i < (int)p.size()-1; i++) {
            d += getDelay(p[i], p[i+1]);
            t += getTrustCost(p[i+1]);
        }
        return d + lambdaPen * std::max(0.0, t - budgetB);
    };

    std::vector<std::vector<int>> pop;
    auto seed = bfs(src, dst);
    if (!seed.empty()) pop.push_back(seed);
    for (int attempt = 0; attempt < popSize * 5 && (int)pop.size() < popSize; attempt++) {
        auto p = randPath(src, dst);
        if (!p.empty()) pop.push_back(p);
        else if (!seed.empty()) pop.push_back(seed);
    }
    if (pop.empty()) {
        auto t1 = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
        return result;
    }

    for (int gen = 0; gen < nGen; gen++) {
        std::vector<double> fits(pop.size());
        for (int i = 0; i < (int)pop.size(); i++) fits[i] = fitness(pop[i]);

        int elite = std::max(1, (int)(pop.size() * 0.05));
        std::vector<int> idx(pop.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin()+elite, idx.end(),
            [&](int a, int b){ return fits[a] < fits[b]; });

        std::vector<std::vector<int>> newPop;
        for (int i = 0; i < elite; i++) newPop.push_back(pop[idx[i]]);

        while ((int)newPop.size() < popSize) {
            auto tourn = [&]() -> int {
                int best = 0; double bf = 1e18;
                for (int tt = 0; tt < 3; tt++) {
                    int i = std::uniform_int_distribution<int>(0,(int)pop.size()-1)(rng);
                    if (fits[i] < bf) { bf = fits[i]; best = i; }
                }
                return best;
            };
            const std::vector<int>& p1 = pop[tourn()];
            const std::vector<int>& p2 = pop[tourn()];

            std::set<int> s1(p1.begin(), p1.end());
            std::vector<int> common;
            for (int n : p2) if (s1.count(n) && n!=src && n!=dst) common.push_back(n);
            std::vector<int> child = p1;
            if (!common.empty()) {
                int pivot = common[std::uniform_int_distribution<int>(0,(int)common.size()-1)(rng)];
                child.clear();
                for (int n : p1) { child.push_back(n); if (n==pivot) break; }
                bool found = false;
                for (int n : p2) {
                    if (n==pivot) { found=true; continue; }
                    if (found) child.push_back(n);
                }
                std::set<int> seenN; std::vector<int> clean;
                for (int n : child) if(seenN.insert(n).second) clean.push_back(n);
                if (!clean.empty() && clean.front()==src && clean.back()==dst)
                    child = clean;
                else
                    child = p1;
            }

            // Mutation
            if (std::uniform_real_distribution<double>(0,1)(rng) < mutRate && (int)child.size() > 2) {
                int mi = std::uniform_int_distribution<int>(1,(int)child.size()-2)(rng);
                auto adjIt = graph.adjacency.find(child[mi-1]);
                if (adjIt != graph.adjacency.end()) {
                    std::set<int> inPath(child.begin(), child.end());
                    bool mutDone = false;
                    for (int j : adjIt->second) {
                        if (mutDone) break;
                        if (!inPath.count(j)) {
                            auto adjIt2 = graph.adjacency.find(j);
                            if (adjIt2 != graph.adjacency.end()) {
                                for (int k : adjIt2->second) {
                                    if (mi+1 < (int)child.size() && k == child[mi+1]) {
                                        child[mi] = j;
                                        mutDone = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (!child.empty()) newPop.push_back(child);
        }
        pop = newPop;
    }

    double bf = 1e18;
    for (size_t i = 0; i < pop.size(); i++) {
        double f = fitness(pop[i]);
        if (f < bf) { bf = f; result.path = pop[i]; }
    }
    if (!result.path.empty()) {
        double d = 0, t = 0;
        for (int i = 0; i < (int)result.path.size()-1; i++) {
            d += getDelay(result.path[i], result.path[i+1]);
            t += getTrustCost(result.path[i+1]);
        }
        result.totalDelay = d;
        result.totalTrustCost = t;
        result.feasible = (t <= budgetB + 1e-9);
        result.reachable = true;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.computeTimeMs = std::chrono::duration<double,std::milli>(t1-t0).count();
    return result;
}

#endif // CMDP_BASELINE_ALGORITHMS_H
