#ifndef __CMDPPACKET_M_H
#define __CMDPPACKET_M_H

#include <omnetpp.h>
#include "veins/base/utils/Coord.h"
#include <vector>

// ============================================================
//  CMDPBeacon
// ============================================================
class CMDPBeacon : public omnetpp::cPacket {
  protected:
    int senderVehicleId_ = 0;
    int senderModuleIndex_ = 0;
    double senderPosX_ = 0;
    double senderPosY_ = 0;
    double senderSpeedX_ = 0;
    double senderSpeedY_ = 0;
    double senderTrust_ = 0;
    double senderTrustCost_ = 0;
    double timestamp_ = 0;
  public:
    CMDPBeacon(const char *name=nullptr, short kind=0) : cPacket(name, kind) {}
    CMDPBeacon(const CMDPBeacon& other) : cPacket(other) { operator=(other); }
    CMDPBeacon& operator=(const CMDPBeacon& other) {
        if (this==&other) return *this;
        cPacket::operator=(other);
        senderVehicleId_=other.senderVehicleId_; senderModuleIndex_=other.senderModuleIndex_;
        senderPosX_=other.senderPosX_; senderPosY_=other.senderPosY_;
        senderSpeedX_=other.senderSpeedX_; senderSpeedY_=other.senderSpeedY_;
        senderTrust_=other.senderTrust_; senderTrustCost_=other.senderTrustCost_;
        timestamp_=other.timestamp_;
        return *this;
    }
    virtual CMDPBeacon *dup() const override { return new CMDPBeacon(*this); }
    int getSenderVehicleId() const { return senderVehicleId_; }
    void setSenderVehicleId(int v) { senderVehicleId_ = v; }
    int getSenderModuleIndex() const { return senderModuleIndex_; }
    void setSenderModuleIndex(int v) { senderModuleIndex_ = v; }
    double getSenderPosX() const { return senderPosX_; }
    void setSenderPosX(double v) { senderPosX_ = v; }
    double getSenderPosY() const { return senderPosY_; }
    void setSenderPosY(double v) { senderPosY_ = v; }
    double getSenderSpeedX() const { return senderSpeedX_; }
    void setSenderSpeedX(double v) { senderSpeedX_ = v; }
    double getSenderSpeedY() const { return senderSpeedY_; }
    void setSenderSpeedY(double v) { senderSpeedY_ = v; }
    double getSenderTrust() const { return senderTrust_; }
    void setSenderTrust(double v) { senderTrust_ = v; }
    double getSenderTrustCost() const { return senderTrustCost_; }
    void setSenderTrustCost(double v) { senderTrustCost_ = v; }
    double getTimestamp() const { return timestamp_; }
    void setTimestamp(double v) { timestamp_ = v; }
};

Register_Class(CMDPBeacon)

// ============================================================
//  CMDPDataPacket
// ============================================================
class CMDPDataPacket : public omnetpp::cPacket {
  protected:
    int srcVehicleId_ = 0;
    int dstVehicleId_ = 0;
    int currentHop_ = 0;
    int totalHops_ = 0;
    std::vector<int> routePath_;
    double creationTime_ = 0;
    double totalDelay_ = 0;
    double totalTrustCost_ = 0;
    double trustBudget_ = 0;
    bool routingComplete_ = false;
    int seqNum_ = 0;
  public:
    CMDPDataPacket(const char *name=nullptr, short kind=0) : cPacket(name, kind) {}
    CMDPDataPacket(const CMDPDataPacket& other) : cPacket(other) { operator=(other); }
    CMDPDataPacket& operator=(const CMDPDataPacket& other) {
        if (this==&other) return *this;
        cPacket::operator=(other);
        srcVehicleId_=other.srcVehicleId_; dstVehicleId_=other.dstVehicleId_;
        currentHop_=other.currentHop_; totalHops_=other.totalHops_;
        routePath_=other.routePath_;
        creationTime_=other.creationTime_; totalDelay_=other.totalDelay_;
        totalTrustCost_=other.totalTrustCost_; trustBudget_=other.trustBudget_;
        routingComplete_=other.routingComplete_; seqNum_=other.seqNum_;
        return *this;
    }
    virtual CMDPDataPacket *dup() const override { return new CMDPDataPacket(*this); }
    int getSrcVehicleId() const { return srcVehicleId_; }
    void setSrcVehicleId(int v) { srcVehicleId_ = v; }
    int getDstVehicleId() const { return dstVehicleId_; }
    void setDstVehicleId(int v) { dstVehicleId_ = v; }
    int getCurrentHop() const { return currentHop_; }
    void setCurrentHop(int v) { currentHop_ = v; }
    int getTotalHops() const { return totalHops_; }
    void setTotalHops(int v) { totalHops_ = v; }
    unsigned int getRoutePathArraySize() const { return routePath_.size(); }
    int getRoutePath(unsigned int k) const { return routePath_.at(k); }
    void setRoutePathArraySize(unsigned int size) { routePath_.resize(size, 0); }
    void setRoutePath(unsigned int k, int v) { routePath_.at(k) = v; }
    void appendRoutePath(int v) { routePath_.push_back(v); }
    double getCreationTime() const { return creationTime_; }
    void setCreationTime(double v) { creationTime_ = v; }
    double getTotalDelay() const { return totalDelay_; }
    void setTotalDelay(double v) { totalDelay_ = v; }
    double getTotalTrustCost() const { return totalTrustCost_; }
    void setTotalTrustCost(double v) { totalTrustCost_ = v; }
    double getTrustBudget() const { return trustBudget_; }
    void setTrustBudget(double v) { trustBudget_ = v; }
    bool getRoutingComplete() const { return routingComplete_; }
    void setRoutingComplete(bool v) { routingComplete_ = v; }
    int getSeqNum() const { return seqNum_; }
    void setSeqNum(int v) { seqNum_ = v; }
};

Register_Class(CMDPDataPacket)

// ============================================================
//  CMDPRouteRequest
// ============================================================
class CMDPRouteRequest : public omnetpp::cPacket {
  protected:
    int srcVehicleId_ = 0;
    int dstVehicleId_ = 0;
    double trustBudget_ = 0;
    double requestTime_ = 0;
    int seqNum_ = 0;
  public:
    CMDPRouteRequest(const char *name=nullptr, short kind=0) : cPacket(name, kind) {}
    CMDPRouteRequest(const CMDPRouteRequest& other) : cPacket(other) { operator=(other); }
    CMDPRouteRequest& operator=(const CMDPRouteRequest& other) {
        if (this==&other) return *this;
        cPacket::operator=(other);
        srcVehicleId_=other.srcVehicleId_; dstVehicleId_=other.dstVehicleId_;
        trustBudget_=other.trustBudget_; requestTime_=other.requestTime_;
        seqNum_=other.seqNum_;
        return *this;
    }
    virtual CMDPRouteRequest *dup() const override { return new CMDPRouteRequest(*this); }
    int getSrcVehicleId() const { return srcVehicleId_; }
    void setSrcVehicleId(int v) { srcVehicleId_ = v; }
    int getDstVehicleId() const { return dstVehicleId_; }
    void setDstVehicleId(int v) { dstVehicleId_ = v; }
    double getTrustBudget() const { return trustBudget_; }
    void setTrustBudget(double v) { trustBudget_ = v; }
    double getRequestTime() const { return requestTime_; }
    void setRequestTime(double v) { requestTime_ = v; }
    int getSeqNum() const { return seqNum_; }
    void setSeqNum(int v) { seqNum_ = v; }
};

Register_Class(CMDPRouteRequest)

// ============================================================
//  CMDPRelayAck
// ============================================================
class CMDPRelayAck : public omnetpp::cPacket {
  protected:
    int relayVehicleId_ = 0;
    int forVehicleId_ = 0;
    bool accepted_ = false;
    double measuredDelay_ = 0;
    double measuredSnir_ = 0;
    int seqNum_ = 0;
  public:
    CMDPRelayAck(const char *name=nullptr, short kind=0) : cPacket(name, kind) {}
    CMDPRelayAck(const CMDPRelayAck& other) : cPacket(other) { operator=(other); }
    CMDPRelayAck& operator=(const CMDPRelayAck& other) {
        if (this==&other) return *this;
        cPacket::operator=(other);
        relayVehicleId_=other.relayVehicleId_; forVehicleId_=other.forVehicleId_;
        accepted_=other.accepted_; measuredDelay_=other.measuredDelay_;
        measuredSnir_=other.measuredSnir_; seqNum_=other.seqNum_;
        return *this;
    }
    virtual CMDPRelayAck *dup() const override { return new CMDPRelayAck(*this); }
    int getRelayVehicleId() const { return relayVehicleId_; }
    void setRelayVehicleId(int v) { relayVehicleId_ = v; }
    int getForVehicleId() const { return forVehicleId_; }
    void setForVehicleId(int v) { forVehicleId_ = v; }
    bool getAccepted() const { return accepted_; }
    void setAccepted(bool v) { accepted_ = v; }
    double getMeasuredDelay() const { return measuredDelay_; }
    void setMeasuredDelay(double v) { measuredDelay_ = v; }
    double getMeasuredSnir() const { return measuredSnir_; }
    void setMeasuredSnir(double v) { measuredSnir_ = v; }
    int getSeqNum() const { return seqNum_; }
    void setSeqNum(int v) { seqNum_ = v; }
};

Register_Class(CMDPRelayAck)

#endif // __CMDPPACKET_M_H
