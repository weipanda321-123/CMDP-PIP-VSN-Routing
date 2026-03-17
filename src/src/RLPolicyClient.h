#ifndef RL_POLICY_CLIENT_H
#define RL_POLICY_CLIENT_H

#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>
#include <chrono>
#include <netinet/tcp.h>

class RLPolicyClient {
public:
    RLPolicyClient() : host_("127.0.0.1"), port_(5556), enabled_(false), sock_(-1) {}
    ~RLPolicyClient() { disconnect(); }

    void initialize(const std::string& host, int port) {
        host_ = host;
        port_ = port;
        enabled_ = true;
    }

    bool isEnabled() const { return enabled_; }

    int queryNextHop(
        int current, int dst, double remainingBudget, int step,
        const std::vector<int>& neighbors,
        const std::vector<double>& trustCosts,
        const std::vector<double>& delays,
        const std::vector<double>& hStar,
        double totalBudget,
        double& outConfidence)
    {
        if (!enabled_ || neighbors.empty()) return -1;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Build JSON request
        std::ostringstream oss;
        oss << "{\"current\":" << current
            << ",\"dst\":" << dst
            << ",\"remaining_budget\":" << remainingBudget
            << ",\"step\":" << step
            << ",\"budget\":" << totalBudget;

        oss << ",\"neighbors\":[";
        for (size_t i = 0; i < neighbors.size(); i++) {
            if (i > 0) oss << ",";
            oss << neighbors[i];
        }
        oss << "]";

        oss << ",\"trust_costs\":[";
        for (size_t i = 0; i < trustCosts.size(); i++) {
            if (i > 0) oss << ",";
            oss << trustCosts[i];
        }
        oss << "]";

        oss << ",\"delays\":[";
        for (size_t i = 0; i < delays.size(); i++) {
            if (i > 0) oss << ",";
            oss << delays[i];
        }
        oss << "]";

        oss << ",\"h_star\":[";
        for (size_t i = 0; i < hStar.size(); i++) {
            if (i > 0) oss << ",";
            oss << hStar[i];
        }
        oss << "]\n";  // NOTE: closing brace missing, add it
        // Fix: need closing brace
        std::string jsonReq = oss.str();
        // Replace last \n with }\n
        if (!jsonReq.empty() && jsonReq.back() == '\n') {
            jsonReq.pop_back();
            jsonReq += "}\n";
        }

        // Ensure persistent connection
        if (!ensureConnected()) return -1;

        // Send
        int sent = ::send(sock_, jsonReq.c_str(), jsonReq.size(), MSG_NOSIGNAL);
        if (sent < 0) {
            disconnect();
            if (!ensureConnected()) return -1;
            sent = ::send(sock_, jsonReq.c_str(), jsonReq.size(), MSG_NOSIGNAL);
            if (sent < 0) return -1;
        }

        // Receive newline-delimited response
        std::string resp = recvLine();
        if (resp.empty()) {
            disconnect();
            return -1;
        }

        int nextHop = parseJsonInt(resp, "next_hop");
        outConfidence = parseJsonDouble(resp, "confidence");

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        queryCount_++;
        totalLatencyMs_ += ms;

        return nextHop;
    }

    double getAvgLatencyMs() const {
        return queryCount_ > 0 ? totalLatencyMs_ / queryCount_ : 0;
    }
    int getQueryCount() const { return queryCount_; }

private:
    std::string host_;
    int port_;
    bool enabled_;
    int sock_;
    int queryCount_ = 0;
    double totalLatencyMs_ = 0;
    std::string recvBuf_;

    bool ensureConnected() {
        if (sock_ >= 0) return true;
        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0) return false;

        int flag = 1;
        setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        struct timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(sock_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port_);
        inet_pton(AF_INET, host_.c_str(), &addr.sin_addr);

        if (connect(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ::close(sock_);
            sock_ = -1;
            return false;
        }
        recvBuf_.clear();
        return true;
    }

    void disconnect() {
        if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
        recvBuf_.clear();
    }

    std::string recvLine() {
        // Check buffer for existing newline
        size_t nl = recvBuf_.find('\n');
        if (nl != std::string::npos) {
            std::string line = recvBuf_.substr(0, nl);
            recvBuf_.erase(0, nl + 1);
            return line;
        }
        // Read until newline
        char buf[8192];
        for (int attempts = 0; attempts < 20; attempts++) {
            int n = recv(sock_, buf, sizeof(buf) - 1, 0);
            if (n <= 0) return "";
            recvBuf_.append(buf, n);
            nl = recvBuf_.find('\n');
            if (nl != std::string::npos) {
                std::string line = recvBuf_.substr(0, nl);
                recvBuf_.erase(0, nl + 1);
                return line;
            }
        }
        return "";
    }

    int parseJsonInt(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return -1;
        pos += search.size();
        while (pos < json.size() && json[pos] == ' ') pos++;
        int val = 0; bool neg = false;
        if (json[pos] == '-') { neg = true; pos++; }
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            val = val * 10 + (json[pos] - '0'); pos++;
        }
        return neg ? -val : val;
    }

    double parseJsonDouble(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.size();
        while (pos < json.size() && json[pos] == ' ') pos++;
        std::string num;
        while (pos < json.size() && (json[pos] == '.' || json[pos] == '-' ||
               json[pos] == 'e' || json[pos] == 'E' || json[pos] == '+' ||
               (json[pos] >= '0' && json[pos] <= '9'))) {
            num += json[pos]; pos++;
        }
        try { return std::stod(num); } catch (...) { return 0; }
    }
};

#endif
