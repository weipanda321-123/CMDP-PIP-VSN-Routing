#ifndef SAF_FUZZY_ENGINE_H
#define SAF_FUZZY_ENGINE_H

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

namespace saf {

// ============================================================
//  Linguistic term indices
// ============================================================
enum class TLife { VeryShort=0, Short, Medium, Long, VeryLong, COUNT };
enum class SMLevel { VeryLow=0, Low, Medium, High, VeryHigh, COUNT };
enum class SOut { Weakest=0, Weaker, Weak, Normal, Strong, Strongest, COUNT };

// ============================================================
//  Trapezoidal membership function  (a <= b <= c <= d)
//  mu(x) = 0          if x < a or x > d
//         (x-a)/(b-a) if a <= x < b
//         1            if b <= x <= c
//         (d-x)/(d-c) if c < x <= d
// ============================================================
struct TrapMF {
    double a, b, c, d;
    double eval(double x) const {
        if (x <= a || x >= d) return 0.0;
        if (x >= b && x <= c) return 1.0;
        if (x < b) return (b == a) ? 1.0 : (x - a) / (b - a);
        /* x > c */ return (d == c) ? 1.0 : (d - x) / (d - c);
    }
};

// ============================================================
//  FuzzyEngine  — fully self-contained, no external libs
// ============================================================
class FuzzyEngine {
public:
    FuzzyEngine();

    // Set T-axis scale factor to adapt to different comm ranges / speeds
    // paperScale=45s (TNSE default), veinsScale depends on R/avgSpeed
    void setTScale(double maxExpectedLinkLife);
    
    // Main entry: returns crisp score in [0,1]
    double evaluate(double linkLifetime_s, double socialMetric) const;

    // For debug / unit-test
    std::string dumpRuleTable() const;

private:
    // --- Membership function banks ---
    // Link lifetime (seconds): 5 terms
    std::array<TrapMF, 5> mfT_;
    // Social metric [0,1]: 5 terms
    std::array<TrapMF, 5> mfSM_;
    // Output score [0,1]: 6 terms
    std::array<TrapMF, 6> mfS_;

    // --- Rule table  ruleTable_[tIdx][smIdx] = output term ---
    std::array<std::array<SOut, 5>, 5> ruleTable_;

    void initMembershipFunctions();
    void initRuleTable();

    // Mamdani inference helpers
    double fuzzifyAndInfer(double T, double SM) const;
};

} // namespace saf

#endif // SAF_FUZZY_ENGINE_H
