#ifndef SAF_SOCIAL_GRAPH_MANAGER_H
#define SAF_SOCIAL_GRAPH_MANAGER_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <cstdint>
#include "veins/base/utils/Coord.h"

namespace saf {

// ============================================================
//  Contact record between two vehicles
// ============================================================
struct ContactRecord {
    double totalContactDuration = 0.0;  // seconds within sliding window
    int    contactCount         = 0;
    double lastContactTime      = -1.0;
    bool   currentlyInContact   = false;
    double contactStartTime     = 0.0;
};

// ============================================================
//  Per-node social data (cached)
// ============================================================
struct NodeSocialData {
    std::vector<uint8_t> socialProfile;   // HP_i binary vector
    int groupId         = -1;  // social label — can be shuffled
    int mobilityGroupId = -1;  // corridor/walk pattern — NEVER shuffled
    int corridorId      = -1;  // continuous corridor tag (0-6, same as groupId by default)
    double clusteringCoeff = 0.0;         // C_i
    double betweenness     = 0.0;         // Cb(i)
    double closeness       = 0.0;         // Cc(i)
    double socialCentrality = 0.0;        // CSC(i)

    // lightweight approximations
    int    degree          = 0;           // number of social edges
    double weightedDegree  = 0.0;         // sum of edge weights
};

// ============================================================
//  Social edge in the contact graph
// ============================================================
struct SocialEdge {
    double weight = 0.0;       // contact intensity
    double lastUpdate = 0.0;
};

// ============================================================
//  SocialGraphManager
//  Modes:
//    strict – full betweenness/closeness via BFS (O(N^2) per update)
//    light  – degree-based approximation (O(k) per node)
// ============================================================
class SocialGraphManager {
public:
    enum class Mode { STRICT, LIGHT };

    SocialGraphManager();

    void setMode(Mode m) { mode_ = m; }
    void setParams(double theta, double psi, double contactWindow,
                   double contactThreshDuration, int contactThreshFreq,
                   int profileLength, int profileSeed);

    // Called once per vehicle at initialization
    void registerNode(int nodeId);
    void setNodeGroup(int nodeId, int groupId);

    // SESSION: mobility group (corridor) — separate from social groupId
    void setMobilityGroup(int nodeId, int mobilityGroupId);
    int  getMobilityGroup(int nodeId) const;

    // Called when beacon heard (every 100ms) — updates contact tracking
    void recordContact(int myId, int otherId, double simTime);

    // Called when a neighbor disappears from beacon range
    void recordContactEnd(int myId, int otherId, double simTime);

    // Periodic global update of social graph (every deltaT_soc seconds)
    // This rebuilds edges & recomputes centrality metrics
    void periodicUpdate(double simTime);

    // --- Query interface ---

    // Get SM(i,j) ∈ [0,1]
    double getSocialMetric(int nodeI, int nodeJ) const;

    // Get homophily SHP(i,j) ∈ [0,1]
    double getHomophily(int nodeI, int nodeJ) const;

    // Get social centrality CSC(i)
    double getSocialCentrality(int nodeId) const;

    // Debug
    std::string dumpGraph() const;

    // Get all registered node IDs
    std::vector<int> getNodeIds() const;

private:
    Mode mode_ = Mode::LIGHT;

    // Parameters
    double theta_ = 0.9;
    double psi_   = 0.1;
    double contactWindow_ = 60.0;         // seconds
    double contactThreshDuration_ = 3.0;  // seconds for edge creation
    int    contactThreshFreq_ = 3;        // contacts for edge creation
    int    profileLength_ = 8;
    int    profileSeed_ = 42;

    // Node data
    std::map<int, NodeSocialData> nodeData_;

    // Contact records: contactRecords_[min(i,j)][max(i,j)]
    std::map<int, std::map<int, ContactRecord>> contactRecords_;

    // Social edges: edges_[i] = set of neighbors with edges
    std::map<int, std::map<int, SocialEdge>> edges_;

    // Helpers
    void generateProfile(int nodeId);
    void pruneOldContacts(double simTime);
    void rebuildEdges();
    void computeCentralityStrict();
    void computeCentralityLight();
    void computeClusteringCoeffs();

    double cosineSimBinary(const std::vector<uint8_t>& a,
                           const std::vector<uint8_t>& b) const;

    // BFS shortest paths for betweenness/closeness
    void bfsAllPairs(std::map<int, std::map<int, int>>& dist,
                     std::map<int, std::map<int, double>>& sigma,
                     std::map<int, std::map<int, double>>& delta) const;

    static int edgeKey(int a, int b) { return std::min(a, b); }
    static int edgeVal(int a, int b) { return std::max(a, b); }
};

} // namespace saf

#endif // SAF_SOCIAL_GRAPH_MANAGER_H
