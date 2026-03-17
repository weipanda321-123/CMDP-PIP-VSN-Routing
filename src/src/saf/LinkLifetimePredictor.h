#ifndef SAF_LINK_LIFETIME_PREDICTOR_H
#define SAF_LINK_LIFETIME_PREDICTOR_H

#include "veins/base/utils/Coord.h"
#include <cmath>
#include "veins/base/utils/Coord.h"
using veins::Coord;
#include <algorithm>

namespace saf {

/**
 * Predicts link lifetime T_hat(v,u) based on current positions and velocities.
 *
 * Solves: find smallest t >= 0 such that |r0 + vr*t| = R
 * where r0 = pos_u - pos_v,  vr = speed_u - speed_v
 *
 * This is a quadratic in t:
 *   |vr|^2 * t^2 + 2*(r0 . vr)*t + (|r0|^2 - R^2) = 0
 */
class LinkLifetimePredictor {
public:
    /**
     * @param commRange Communication range R (meters)
     * @param tMax      Maximum predicted lifetime cap (seconds)
     */
    LinkLifetimePredictor(double commRange = 150.0, double tMax = 60.0)
        : R_(commRange), tMax_(tMax) {}

    /**
     * Predict link lifetime in seconds.
     * Returns 0 if currently out of range.
     * Returns tMax if vehicles are nearly stationary relative to each other.
     */
    double predict(const Coord& posV, const Coord& speedV,
                   const Coord& posU, const Coord& speedU) const
    {
        using veins::Coord;

        Coord r0(posU.x - posV.x, posU.y - posV.y, 0);
        Coord vr(speedU.x - speedV.x, speedU.y - speedV.y, 0);

        double dist2 = r0.x * r0.x + r0.y * r0.y;
        double R2 = R_ * R_;

        // Currently out of range?
        if (dist2 > R2) return 0.0;

        double vr2 = vr.x * vr.x + vr.y * vr.y;

        // Relative velocity ≈ 0 → vehicles co-moving
        if (vr2 < 1e-8) {
            return tMax_;  // effectively infinite within our horizon
        }

        double dot_r0_vr = r0.x * vr.x + r0.y * vr.y;

        // Quadratic: vr2*t^2 + 2*dot*t + (dist2 - R2) = 0
        double a = vr2;
        double b = 2.0 * dot_r0_vr;
        double c = dist2 - R2;

        double disc = b * b - 4.0 * a * c;

        if (disc < 0) {
            // No real solution — distance never reaches R
            // (vehicles are diverging but currently within range and
            //  their trajectory bends them back; in 2D linear model
            //  this shouldn't happen, but handle gracefully)
            return tMax_;
        }

        double sqrtDisc = std::sqrt(disc);
        double t1 = (-b + sqrtDisc) / (2.0 * a);
        double t2 = (-b - sqrtDisc) / (2.0 * a);

        // We want the smallest t > 0
        double tResult = tMax_;
        if (t1 > 1e-6 && t1 < tResult) tResult = t1;
        if (t2 > 1e-6 && t2 < tResult) tResult = t2;

        return std::min(tResult, tMax_);
    }

    void setCommRange(double r) { R_ = r; }
    void setTMax(double t) { tMax_ = t; }

private:
    double R_;
    double tMax_;
};

} // namespace saf

#endif // SAF_LINK_LIFETIME_PREDICTOR_H
