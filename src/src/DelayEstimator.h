#ifndef DELAY_ESTIMATOR_H
#define DELAY_ESTIMATOR_H

#include <map>
#include <deque>
#include <cmath>
#include <algorithm>
#include <sstream>

/**
 * DelayEstimator: V2V delay model (OJCOMS Eq.1-10)
 *
 * cij = Ttotal * (1 + Pretx/(1-Pretx))
 * Ttotal = Ttx + Tprop + Tack + 2*Tpro   (Eq.5)
 * Pretx  = P(SINR < gamma_th)             (Eq.6)
 */

struct LinkDelayRecord {
    double distance = 0;
    double snir = 0;
    double measuredDelay = 0;
    double timestamp = 0;
};

class DelayEstimator {
public:
    DelayEstimator()
        : packetSize_(512), dataRate_(6e6), ackTime_(0.3e-3),
          procTime_(0.1e-3), pathLossExp_(2.7), sinrThreshold_(8.0),
          txPower_(0.1), noisePower_(1e-10), antennaGain_(1.0),
          historyWindow_(20) {}

    void initialize(double packetSize, double dataRate, double ackTime,
                    double procTime, double pathLossExp, double sinrThresh,
                    double txPower, double noisePower, double antennaGain) {
        packetSize_ = packetSize;
        dataRate_ = dataRate;
        ackTime_ = ackTime;
        procTime_ = procTime;
        pathLossExp_ = pathLossExp;
        sinrThreshold_ = sinrThresh;
        txPower_ = txPower;
        noisePower_ = noisePower;
        antennaGain_ = antennaGain;
    }

    void recordOracleDelay(int srcId, int dstId, double delay,
                           double distance, double snir, double simTime) {
        uint64_t key = linkKey(srcId, dstId);
        LinkDelayRecord rec;
        rec.distance = distance;
        rec.snir = snir;
        rec.measuredDelay = delay;
        rec.timestamp = simTime;
        auto& hist = oracleHistory_[key];
        hist.push_back(rec);
        if ((int)hist.size() > historyWindow_) hist.pop_front();
    }

    double getOracleDelay(int srcId, int dstId) const {
        uint64_t key = linkKey(srcId, dstId);
        auto it = oracleHistory_.find(key);
        if (it != oracleHistory_.end() && !it->second.empty())
            return it->second.back().measuredDelay;
        return -1.0;
    }

    double getMeanOracleDelay(int srcId, int dstId) const {
        uint64_t key = linkKey(srcId, dstId);
        auto it = oracleHistory_.find(key);
        if (it != oracleHistory_.end() && !it->second.empty()) {
            double sum = 0;
            for (const auto& r : it->second) sum += r.measuredDelay;
            return sum / it->second.size();
        }
        return -1.0;
    }

    /** Predict delay (Eq.1-5) with proper SNIR handling */
    double predictDelay(double distance, double snirLinear) const {
        double txTime = (packetSize_ * 8.0) / dataRate_;                // Eq.3
        double propTime = distance / 3e8;                                // Eq.4
        double tTotal = txTime + propTime + ackTime_ + 2.0 * procTime_; // Eq.5

        // Retransmission probability (Eq.6,10)
        double sinrThreshLinear = std::pow(10.0, sinrThreshold_ / 10.0);
        double pRetx = 0.0;
        if (snirLinear > 1e-6) {
            // Rayleigh fading approximation: P(gamma < gamma_th) = 1 - exp(-gamma_th/gamma_avg)
            pRetx = 1.0 - std::exp(-sinrThreshLinear / snirLinear);
        } else {
            pRetx = 0.5; // Unknown SNIR fallback
        }
        pRetx = std::max(0.001, std::min(pRetx, 0.99));

        // Expected delay (Eq.1): cij = Ttotal * (1 + Pretx/(1-Pretx))
        return tTotal * (1.0 + pRetx / (1.0 - pRetx));
    }

    double predictDelayByDistance(double distance) const {
        double pathLoss = std::pow(std::max(distance, 1.0), pathLossExp_);
        double rxPower = txPower_ * antennaGain_ / pathLoss;
        double snirLinear = rxPower / noisePower_;
        return predictDelay(distance, snirLinear);
    }

    /** Best available delay estimate: oracle > historical > model */
    double getDelayEstimate(int srcId, int dstId, double distance,
                            double snirLinear, bool useOracle) const {
        if (useOracle) {
            double oracle = getOracleDelay(srcId, dstId);
            if (oracle > 0) return oracle;
        }
        double meanOracle = getMeanOracleDelay(srcId, dstId);
        if (meanOracle > 0) return meanOracle;
        if (snirLinear > 1e-6) return predictDelay(distance, snirLinear);
        return predictDelayByDistance(distance);
    }

    void setGNNPrediction(int srcId, int dstId, double delay, double variance) {
        uint64_t key = linkKey(srcId, dstId);
        gnnPredictions_[key] = {delay, variance};
    }

    double getGNNDelay(int srcId, int dstId) const {
        uint64_t key = linkKey(srcId, dstId);
        auto it = gnnPredictions_.find(key);
        if (it != gnnPredictions_.end()) return it->second.first;
        return -1.0;
    }

private:
    double packetSize_, dataRate_, ackTime_, procTime_;
    double pathLossExp_, sinrThreshold_;
    double txPower_, noisePower_, antennaGain_;
    int historyWindow_;
    std::map<uint64_t, std::deque<LinkDelayRecord>> oracleHistory_;
    std::map<uint64_t, std::pair<double, double>> gnnPredictions_;

    static uint64_t linkKey(int a, int b) {
        return ((uint64_t)a << 32) | (uint64_t)(unsigned int)b;
    }
};

#endif
