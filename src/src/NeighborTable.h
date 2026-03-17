#ifndef NEIGHBOR_TABLE_H
#define NEIGHBOR_TABLE_H

#include <map>
#include <vector>
#include <algorithm>

struct NeighborEntry {
    int vehicleId = -1;
    double posX = 0, posY = 0;
    double speedX = 0, speedY = 0;
    double lastSeen = 0;
    double snir = 0;        // last measured SNIR (linear, NOT dB)
    double distance = 0;
    double trustScore = 0;  // cached trust score from beacon
    int moduleIndex = -1;
};

class NeighborTable {
public:
    void update(int vid, double x, double y, double sx, double sy,
                double snir, double dist, double trust, int modIdx, double simTime) {
        NeighborEntry& e = table_[vid];
        e.vehicleId = vid;
        e.posX = x; e.posY = y;
        e.speedX = sx; e.speedY = sy;
        e.snir = snir;
        e.distance = dist;
        e.trustScore = trust;
        e.moduleIndex = modIdx;
        e.lastSeen = simTime;
    }

    std::vector<NeighborEntry> getAllEntries() const {
        std::vector<NeighborEntry> result;
        for (const auto& kv : table_) {
            result.push_back(kv.second);
        }
        return result;
    }

    void purgeOld(double simTime, double timeout) {
        for (auto it = table_.begin(); it != table_.end(); ) {
            if (simTime - it->second.lastSeen > timeout)
                it = table_.erase(it);
            else ++it;
        }
    }

    bool has(int vid) const { return table_.find(vid) != table_.end(); }

    const NeighborEntry* get(int vid) const {
        auto it = table_.find(vid);
        return (it != table_.end()) ? &it->second : nullptr;
    }

    std::vector<int> getNeighborIds() const {
        std::vector<int> ids;
        for (const auto& kv : table_) ids.push_back(kv.first);
        return ids;
    }

    const std::map<int, NeighborEntry>& getAll() const { return table_; }
    int size() const { return (int)table_.size(); }
    void clear() { table_.clear(); }

private:
    std::map<int, NeighborEntry> table_;
};

#endif
