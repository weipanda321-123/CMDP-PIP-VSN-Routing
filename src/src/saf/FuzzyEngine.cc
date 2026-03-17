#include "FuzzyEngine.h"
#include <sstream>
#include <cassert>
#include <limits>

namespace saf {

// ================================================================
//  Constructor
// ================================================================
FuzzyEngine::FuzzyEngine() {
    initMembershipFunctions();
    initRuleTable();
}

// ================================================================
//  Membership functions — exact breakpoints from paper Tables III–V
// ================================================================
void FuzzyEngine::initMembershipFunctions() {
    // --- Link Lifetime T (seconds) — Table III ---
    //  VeryShort: trap(0, 0, 5, 10)    core=[0,5], ramp-down to 10
    //  Short:     trap(5, 10, 15, 20)
    //  Medium:    trap(15, 20, 30, 35)
    //  Long:      trap(30, 35, 40, 45)
    //  VeryLong:  trap(40, 45, INF, INF)
    constexpr double INF_T = 1e6;
    mfT_[0] = {0.0,  0.0,  5.0,  10.0};        // VeryShort
    mfT_[1] = {5.0,  10.0, 15.0, 20.0};         // Short
    mfT_[2] = {15.0, 20.0, 30.0, 35.0};         // Medium
    mfT_[3] = {30.0, 35.0, 40.0, 45.0};         // Long
    mfT_[4] = {40.0, 45.0, INF_T, INF_T};       // VeryLong

    // --- Social Metric SM [0,1] — Table IV ---
    //  VeryLow:  trap(0, 0, 0.1, 0.2)
    //  Low:      trap(0.1, 0.2, 0.3, 0.4)
    //  Medium:   trap(0.3, 0.4, 0.6, 0.7)
    //  High:     trap(0.6, 0.7, 0.8, 0.9)
    //  VeryHigh: trap(0.8, 0.9, 1.0, 1.0)
    mfSM_[0] = {0.0,  0.0,  0.1, 0.2};
    mfSM_[1] = {0.1,  0.2,  0.3, 0.4};
    mfSM_[2] = {0.3,  0.4,  0.6, 0.7};
    mfSM_[3] = {0.6,  0.7,  0.8, 0.9};
    mfSM_[4] = {0.8,  0.9,  1.0, 1.01};   // slightly > 1 to include 1.0

    // --- Output Score S [0,1] — Table V ---
    //  Weakest:   trap(0, 0, 0.1, 0.2)
    //  Weaker:    trap(0.1, 0.2, 0.3, 0.4)
    //  Weak:      trap(0.3, 0.4, 0.5, 0.6)
    //  Normal:    trap(0.5, 0.6, 0.7, 0.8)
    //  Strong:    trap(0.7, 0.8, 0.9, 1.0)
    //  Strongest: trap(0.9, 0.95, 1.0, 1.01)
    mfS_[0] = {0.0,  0.0,  0.1, 0.2};
    mfS_[1] = {0.1,  0.2,  0.3, 0.4};
    mfS_[2] = {0.3,  0.4,  0.5, 0.6};
    mfS_[3] = {0.5,  0.6,  0.7, 0.8};
    mfS_[4] = {0.7,  0.8,  0.9, 1.0};
    mfS_[5] = {0.9,  0.95, 1.0, 1.01};
}

// ================================================================
//  Rule Table — paper Table VI  (25 rules, R1–R25)
//  Rows: TLife  {VeryShort, Short, Medium, Long, VeryLong}
//  Cols: SM     {VeryLow, Low, Medium, High, VeryHigh}
// ================================================================
void FuzzyEngine::initRuleTable() {
    using S = SOut;
    // Row 0: T = VeryShort
    ruleTable_[0] = {S::Weakest, S::Weaker,  S::Weak,   S::Weak,    S::Normal};
    // Row 1: T = Short
    ruleTable_[1] = {S::Weakest, S::Weaker,  S::Weak,   S::Normal,  S::Normal};
    // Row 2: T = Medium
    ruleTable_[2] = {S::Weaker,  S::Weak,    S::Normal, S::Strong,  S::Strong};
    // Row 3: T = Long
    ruleTable_[3] = {S::Weaker,  S::Weak,    S::Normal, S::Strong,  S::Strongest};
    // Row 4: T = VeryLong
    ruleTable_[4] = {S::Weak,    S::Normal,  S::Strong, S::Strongest, S::Strongest};
}

// ================================================================
//  Mamdani inference with max-membership defuzzification
// ================================================================
double FuzzyEngine::fuzzifyAndInfer(double T, double SM) const {
    // Step 1: fuzzify inputs
    std::array<double, 5> muT, muSM;
    for (int i = 0; i < 5; ++i) {
        muT[i]  = mfT_[i].eval(T);
        muSM[i] = mfSM_[i].eval(SM);
    }

    // Step 2: evaluate all 25 rules; collect firing strengths per output term
    std::array<double, 6> outStrength{};  // one per SOut term
    for (int ti = 0; ti < 5; ++ti) {
        for (int si = 0; si < 5; ++si) {
            double alpha = std::min(muT[ti], muSM[si]);  // AND = min
            if (alpha <= 0.0) continue;
            int outIdx = static_cast<int>(ruleTable_[ti][si]);
            outStrength[outIdx] = std::max(outStrength[outIdx], alpha); // OR = max
        }
    }

    // Step 3: defuzzification — max criterion
    // Find the output term with the highest aggregated strength.
    // For that term, return the x where its membership function peaks
    // (center of the flat top = (b+c)/2).
    // If multiple terms tie, pick the one with the higher center (optimistic).
    double bestAlpha = -1.0;
    double bestCenter = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (outStrength[i] > bestAlpha ||
            (outStrength[i] == bestAlpha && (mfS_[i].b + mfS_[i].c) / 2.0 > bestCenter)) {
            bestAlpha  = outStrength[i];
            bestCenter = (mfS_[i].b + mfS_[i].c) / 2.0;
        }
    }

    // Scale the center by the firing strength for finer discrimination
    // (This is a refinement: within the winning term, the output is
    //  proportional to how strongly the rule fired.)
    // Pure max-criterion would just return bestCenter, but we add
    // alpha-weighted interpolation for better ranking resolution.
    if (bestAlpha <= 0.0) return 0.0;

    // Pure max-criterion output (paper spec):
    return bestCenter;
}

// ================================================================
//  Adaptive T-axis rescaling
//  Paper assumes max useful linkLife ~ 45s (R=500-1000m, v=10-15m/s)
//  For Veins R=300m, v=15m/s -> max ~ 20s, so ratio = 20/45
//  Rescale all T breakpoints proportionally to preserve fuzzy partition
// ================================================================
void FuzzyEngine::setTScale(double maxExpectedLinkLife) {
    // Paper's original breakpoints assume maxT ~ 45s
    constexpr double PAPER_MAX_T = 45.0;
    double ratio = maxExpectedLinkLife / PAPER_MAX_T;
    if (ratio < 0.1) ratio = 0.1;
    if (ratio > 3.0) ratio = 3.0;
    
    // Original paper breakpoints (Table III)
    // VeryShort: (0, 0, 5, 10)
    // Short:     (5, 10, 15, 20) 
    // Medium:    (15, 20, 30, 35)
    // Long:      (30, 35, 40, 45)
    // VeryLong:  (40, 45, INF, INF)
    constexpr double INF_T = 1e6;
    double bp[][4] = {
        {0,  0,  5,  10},
        {5,  10, 15, 20},
        {15, 20, 30, 35},
        {30, 35, 40, 45},
        {40, 45, INF_T, INF_T}
    };
    for (int i = 0; i < 5; ++i) {
        mfT_[i].a = bp[i][0] * ratio;
        mfT_[i].b = bp[i][1] * ratio;
        mfT_[i].c = (bp[i][2] < 1e5) ? bp[i][2] * ratio : INF_T;
        mfT_[i].d = (bp[i][3] < 1e5) ? bp[i][3] * ratio : INF_T;
    }
    fprintf(stderr, "[SAF] T-scale set: maxT=%.1f ratio=%.2f -> VShort<%.1f Short<%.1f Med<%.1f Long<%.1f VLong>%.1f\n",
            maxExpectedLinkLife, ratio,
            mfT_[0].d, mfT_[1].d, mfT_[2].d, mfT_[3].d, mfT_[4].a);
}

double FuzzyEngine::evaluate(double linkLifetime_s, double socialMetric) const {
    // Clamp inputs to valid ranges
    double T  = std::max(0.0, linkLifetime_s);
    double SM = std::max(0.0, std::min(1.0, socialMetric));
    return fuzzifyAndInfer(T, SM);
}

std::string FuzzyEngine::dumpRuleTable() const {
    static const char* tNames[] = {"VShort","Short","Medium","Long","VLong"};
    static const char* smNames[] = {"VLow","Low","Med","High","VHigh"};
    static const char* sNames[] = {"Weakest","Weaker","Weak","Normal","Strong","Strongest"};
    std::ostringstream os;
    os << "SAF Rule Table:\n";
    os << "T\\SM\t";
    for (int j = 0; j < 5; ++j) os << smNames[j] << "\t";
    os << "\n";
    for (int i = 0; i < 5; ++i) {
        os << tNames[i] << "\t";
        for (int j = 0; j < 5; ++j) {
            os << sNames[static_cast<int>(ruleTable_[i][j])] << "\t";
        }
        os << "\n";
    }
    return os.str();
}

} // namespace saf
