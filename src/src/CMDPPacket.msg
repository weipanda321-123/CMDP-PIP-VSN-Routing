// CMDPPacket.msg - CMDP routing message definitions (Veins 5.2 compatible)

cplusplus {{
#include "veins/base/utils/Coord.h"
}}

class noncobject veins::Coord;

packet CMDPBeacon {
    int senderVehicleId;
    int senderModuleIndex;
    double senderPosX;
    double senderPosY;
    double senderSpeedX;
    double senderSpeedY;
    double senderTrust;
    double senderTrustCost;
    double timestamp;
}

packet CMDPDataPacket {
    int srcVehicleId;
    int dstVehicleId;
    int currentHop;
    int totalHops;
    int routePath[];
    double creationTime;
    double totalDelay;
    double totalTrustCost;
    double trustBudget;
    bool routingComplete;
    int seqNum;
}

packet CMDPRouteRequest {
    int srcVehicleId;
    int dstVehicleId;
    double trustBudget;
    double requestTime;
    int seqNum;
}

packet CMDPRelayAck {
    int relayVehicleId;
    int forVehicleId;
    bool accepted;
    double measuredDelay;
    double measuredSnir;
    int seqNum;
}
