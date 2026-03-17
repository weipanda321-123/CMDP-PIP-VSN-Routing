// CMDPApp.ned
// Uses "like IBaseApplLayer" NOT "extends BaseApplLayer" (SAF lesson)

simple CMDPApp like org.car2x.veins.base.modules.IBaseApplLayer
{
    parameters:
        @class(CMDPApp);
        int headerLength @unit(bit) = default(80bit);

        // ---- CMDP routing switches ----
        bool enableCMDP = default(true);
        bool cmdpUsePIP = default(true);
        bool cmdpUseGNN = default(false);
        bool useLearnedMask = default(false);
        bool strictSafeMode = default(false);   // E1: mask empty → abort (no repair)
        string cmdpPolicyMode = default("LagrangianGreedy");

        // ---- DN/SAF integration ----
        bool enableDN = default(false);
        bool enableSAF = default(false);
        double pipAlpha = default(1.0);  // PIP tightening factor (1.0=original, 2.0=strict)

        // --- Ch6 social ablation parameters ---
        bool ablationShuffleSocial = default(false);  // S1: shuffle CF/SHP/trust
        bool ablationNoSelfish     = default(false);  // S2: disable selfish behavior
        double selfishDropProb     = default(0.4);    // selfish node drop probability

        // ---- Trust model (Eq.11-14) ----
        double trustAlpha1 = default(0.5);
        double trustAlpha2 = default(0.5);
        double trustThresholdPth = default(0.1);
        double lagrangianLambda = default(1.0);
        double etaLambda = default(0.005);    // dual ascent step size

        // ---- Timing ----
        double beaconInterval @unit(s) = default(0.1s);   // SAF: 0.1s works best
        double neighborTimeout @unit(s) = default(5s);
        double dataGenInterval @unit(s) = default(2s);
        double warmupTime @unit(s) = default(10s);
        double commRange @unit(m) = default(300m);
        int maxHops = default(20);
        double maxDstDistance @unit(m) = default(9999m);

        // ---- Scenario ----
        int totalVehicles = default(50);
        int numGroups = default(3);

        // ---- Delay model (Eq.1-10, Table 1) ----
        double packetSizeBytes = default(512);
        double dataRateBps = default(6e6);
        double ackTimeS = default(0.3e-3);
        double procTimeS = default(0.1e-3);
        double pathLossExponent = default(2.7);
        double sinrThresholdDB = default(8);
        double txPowerW = default(0.1);        // 20 dBm = 0.1 W
        double noisePowerW = default(1e-10);
        double antennaGainLinear = default(1.0);

        // ---- GNN ----
        string gnnServerHost = default("127.0.0.1");
        int gnnServerPort = default(5555);

        // ---- Logging ----
        bool enableCSVLog = default(true);
        string csvLogPath = default("cmdp_log.csv");
        string gnnTrainingDataPath = default("gnn_training_data.csv");

        // ---- SESSION CBR (SAF-native 验证) ----
        string dataMode            = default("Periodic");  // "Periodic" | "SessionCBR"
        double sessionWarmup       = default(20.0);        // s，暖机后再发session
        double sessionPktInterval  = default(1.0);         // s，包间隔
        int    sessionTau          = default(30);          // 每个session连续发包数
        int    numSessionSources   = default(60);        double maxSessionDistMin = default(350.0);  // B1: corridor OD min dist (m)
        double maxSessionDistMax = default(900.0);  // B1: corridor OD max dist (m)
          // 参与session的源节点数
        bool   enableSocialShuffle = default(false);       // shuffle社交属性做消融

        // ---- OMNeT++ signal declarations ----
        @signal[cmdpEndToEndDelay](type=double);
        @signal[cmdpComputeLatency](type=double);
        @signal[cmdpTrustFeasible](type=double);
        @signal[cmdpTrustViolation](type=double);
        @signal[cmdpRouteHops](type=long);
        @signal[cmdpPIPMaskedActions](type=long);

        @statistic[cmdpEndToEndDelay](title="E2E Delay"; source=cmdpEndToEndDelay; record=vector,mean,max,min);
        @statistic[cmdpComputeLatency](title="Compute Latency ms"; source=cmdpComputeLatency; record=vector,mean,max);
        @statistic[cmdpTrustFeasible](title="Trust Feasible"; source=cmdpTrustFeasible; record=vector,mean);
        @statistic[cmdpTrustViolation](title="Trust Violation"; source=cmdpTrustViolation; record=vector,mean);
        @statistic[cmdpRouteHops](title="Route Hops"; source=cmdpRouteHops; record=vector,mean);
        @statistic[cmdpPIPMaskedActions](title="PIP Masked"; source=cmdpPIPMaskedActions; record=vector,mean);
        @display("i=block/app");

    gates:
        input lowerLayerIn;
        output lowerLayerOut;
        input lowerControlIn;
        output lowerControlOut;
}
