#ifndef CMDP_APP_H
#define CMDP_APP_H
#define CMDP_NO_DN

#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <memory>

#include "veins/base/modules/BaseApplLayer.h"
#include "veins/modules/mobility/traci/TraCIMobility.h"
#include "veins/modules/mobility/traci/TraCICommandInterface.h"
// Veins 5.2: generated message headers use _m.h suffix
#include "veins/modules/messages/BaseFrame1609_4_m.h"
// SNIR extraction (SAF lesson: must include these)
#include "veins/base/phyLayer/PhyToMacControlInfo.h"
#include "veins/modules/phy/DeciderResult80211.h"
// Channel enum (cch, sch1, etc.)
#include "veins/modules/utility/Consts80211p.h"

#include "CMDPPacket_m.h"
#include "TrustManager.h"
#include "PIPMasker.h"
#include "CmdpRouter.h"
#include "DelayEstimator.h"
#include "NeighborTable.h"
#ifndef CMDP_NO_DN
#include "NeighborAdapter.h"
#include "veins_dn2/NeighborTable.h"
#include "dn/DNBackboneManager.h"
#endif // CMDP_NO_DN
#include "saf/SAFRelaySelector.h"
#include "saf/SocialGraphManager.h"
#include "saf/FuzzyEngine.h"

using namespace omnetpp;

class CMDPApp : public veins::BaseApplLayer {
public:
    virtual ~CMDPApp();
    void initialize(int stage) override;
    void finish() override;

protected:
    void handleSelfMsg(cMessage* msg) override;
    void handleLowerMsg(cMessage* msg) override;
    void handleLowerControl(cMessage* msg) override;

    // Message handlers (onBeacon takes snir parameter)
    void onBeacon(CMDPBeacon* beacon, double snir);
    void onDataPacket(CMDPDataPacket* pkt);
    void onRelayAck(CMDPRelayAck* ack);

    // Core functions
    void sendBeacon();
    void generateDataPacket();
    GraphSnapshot buildGraphSnapshot();
    GraphSnapshot buildFilteredGraphSnapshot();
    void assignSocialAttributes();
    bool shouldForwardPacket();
    int  selectGroupDestination();
    void onBeaconUpdateDNSAF(int senderId, double t);
    void dnPeriodicUpdate();
    void forwardDataPacket(CMDPDataPacket* pkt);
    void sendRelayAck(int forVehicle, bool accepted, double delay, double snir);

    // Utility
    int getMyVehicleId() const;
    veins::Coord getMyPosition() const;
    veins::Coord getMySpeed() const;  // FIXED: uses VehicleCommandInterface

    // Logging
    void logRouteResult(const RouteResult& result, double simTime);
    void logGNNTrainingRecord(int src, int dst, double dist, double snir,
                               double oracleDelay, double simTime);
    void queryGNNServer();

    // ---- SESSION CBR methods ----
    void onSessionInit();
    void onSessionPkt();
    void onSocialShuffle();
    int  selectSessionDst(int srcId, int mobGroup);
    void logSessionStats(int flowId);
    void recordSessionStats();

private:
    // Config
    bool enableCMDP_;
    bool cmdpUsePIP_;
    bool cmdpUseGNN_;
    std::string gnnServerHost_;
    int gnnServerPort_;
    int gnnSock_ = -1;
    std::string cmdpPolicyMode_;
    bool enableDN_;
    bool enableSAF_;

    // --- Social attributes (Ch6 Optics Valley) ---
    std::string socialGroup_ = "unknown";
    double contactFreq_      = 0.05;
    double socialHomophily_  = 0.5;
    double forwardTrust_     = 0.5;
    double selfishDropRate_  = 0.0;
    bool   ablationShuffleSocial_ = false;
    bool   ablationNoSelfish_     = false;
    double selfishDropProb_  = 0.4;
    bool   socialAttributesAssigned_ = false;
    double beaconInterval_;
    double neighborTimeout_;
    double dataGenInterval_;
    double commRange_;
    GraphSnapshot cachedGraph_;
    double lastGraphTime_ = -1.0;
    int totalVehicles_;
    int numGroups_;

    // Core components
    TrustManager trustMgr_;
    CmdpRouter router_;
    DelayEstimator delayEst_;
    NeighborTable neighborTable_;
    // DN subsystem
#ifndef CMDP_NO_DN
    veins_dn2::DNBackboneManager dnManager_;
    veins_dn2::NeighborTable dnNeighborTable_;
#endif
    cMessage* dnUpdateTimer_ = nullptr;
    // SAF subsystem
    saf::SAFRelaySelector safSelector_;
    saf::SocialGraphManager socialGraphMgr_;
    int safFwdCount_ = 0;
    int safCase1Count_ = 0;
    saf::FuzzyEngine fuzzyEngine_;

    // Mobility
    veins::TraCIMobility* mobility_;
    veins::TraCICommandInterface* traci_;

    // Timers
    cMessage* beaconTimer_ = nullptr;
    cMessage* dataGenTimer_ = nullptr;

    // State
    int myVehicleId_;
    int seqNum_;

    // Logging
    bool enableCSVLog_;
    std::ofstream csvLog_;
    std::ofstream deliveryLog_;
    std::ofstream gnnDataLog_;

    // Statistics counters (BUG-12 fix)
    int totalPktsSent_;
    int totalPktsRecv_;
    int totalPktsFailed_;
    int totalRouteQueries_;
    int totalFeasibleRoutes_;
    int totalConstraintViolations_;
    double totalHopCount_;
    int deliveredPkts_;
    double totalDelaySum_;

    // OMNeT++ signals
    simsignal_t sigEndToEndDelay_;
    simsignal_t sigComputeLatency_;
    simsignal_t sigTrustFeasible_;
    simsignal_t sigTrustViolation_;
    simsignal_t sigRouteHops_;
    simsignal_t sigPIPMaskedActions_;

    // ================================================================
    // SESSION CBR (SAF-native 验证)
    // ================================================================

    // 每个 flow 的静态描述（src/dst/tau，创建后不变）
    struct SessionSpec {
        int flowId    = -1;
        int srcNodeId = -1;
        int dstNodeId = -1;
        int tauTotal  = 30;
        int sent      = 0;
        simtime_t startTime = 0;
    };

    // 每个 flow 的运行时缓存（路径 + replan 统计）
    struct SessionState {
        int flowId    = -1;
        int replans   = 0;
        int cacheHits = 0;
        int sent      = 0;
        std::vector<int> cachedPath;
        simtime_t lastPlanTime = 0;
    };

    // ---- Session 参数 ----
    bool   dataMode_Session_    = false;  // dataMode == "SessionCBR"
    double sessionWarmup_       = 20.0;   // s
    double sessionPktInterval_  = 1.0;    // s
    int    sessionTau_          = 30;
    int    numSessionSources_   = 60;
    double maxSessionDist_min_ = 350.0;  // B1: corridor session min dist (m)
    double maxSessionDist_max_ = 900.0;  // B1: corridor session max dist (m)
    bool   enableSocialShuffle_ = false;
    double sessionCacheExpiry_  = 3.0;   // s，超时强制 replan

    // ---- Session 状态 ----
    std::map<int, SessionSpec>  activeSessions_;   // flowId → spec
    std::map<int, SessionState> sessionCacheMap_;  // flowId → cache
    int nextFlowId_ = 0;

    // ---- Session 统计 ----
    long   sessRouteChanges_ = 0;  // 触发 replan 次数
    long   sessCacheHits_    = 0;  // 命中缓存次数
    long   sessRepairCount_  = 0;  // SAF Case1 触发次数
    long   sessRepairSucc_   = 0;  // SAF Case1 成功次数
    double sessLinkLifeSum_  = 0;  // 选中链路寿命累计
    long   sessLinkLifeCnt_  = 0;
    long   sessSocialHit_    = 0;  // safScore>0.5 中继被选中次数
    long   sessSocialTotal_  = 0;  // 总中继选择次数
};

#endif
