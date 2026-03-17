#pragma once
#include <cstddef>
#include <cstdint>
namespace omnetpp {
    class cRNG {
    public:
        virtual double doubleRand() = 0;
        virtual double doubleRandNormal(double mean, double stddev) = 0;
        virtual long intRand(long r) = 0;
    };
}
