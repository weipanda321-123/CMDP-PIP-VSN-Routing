#!/usr/bin/env python3
"""
PPO-Lagrangian Training on Optics Valley Real Snapshots
========================================================
Loads 4 time-slice graph snapshots (t=15,25,35,45s) from SUMO,
trains PPO policy that generalizes across different traffic densities.

Usage:
  cd ~/veins_workspace/cmdp_pkg_v2/simulations/manhattan
  python3 ../../scripts/ppo_train_optics_valley.py \
    --snapshots optics_valley_snapshots.json \
    --gnn-model gnn_delay_model.pt \
    --output ppo_policy_optics_valley.pt \
    --epochs 100 --episodes 64 --max-hops 10
"""
import os, sys, math, json, heapq, argparse, time
import numpy as np
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import GNN model for delay prediction
try:
    from gnn_train import EdgeFeatureNet
    HAS_GNN = True
except ImportError:
    HAS_GNN = False
    print("[WARN] gnn_train not found, using distance-based delay model")


# ============================================================
#  Snapshot-based Routing Environment
# ============================================================
class SnapshotRoutingEnv:
    """
    Routing environment built from a SUMO snapshot.
    Supports multiple snapshots for curriculum/rotation training.
    """
    def __init__(self, snapshot, budget, max_hops=10, comm_range=300.0):
        self.budget = budget
        self.max_hops = max_hops
        self.comm_range = comm_range
        
        # Parse snapshot
        vehicles = snapshot["vehicles"]
        edges = snapshot["edges"]
        
        # Build node ID mapping (original SUMO IDs -> contiguous 0..N-1)
        self.orig_ids = sorted([int(k) for k in vehicles.keys()])
        self.id_map = {orig: idx for idx, orig in enumerate(self.orig_ids)}
        self.rev_map = {idx: orig for orig, idx in self.id_map.items()}
        self.num_nodes = len(self.orig_ids)
        
        # Node attributes
        self.positions = {}
        self.trust_costs = {}
        self.social_attrs = {}
        for orig_id_str, vinfo in vehicles.items():
            orig_id = int(orig_id_str)
            idx = self.id_map[orig_id]
            self.positions[idx] = np.array(vinfo["pos"][:2])
            # Trust cost r_j = -log(trust), clipped
            trust_val = vinfo.get("trust", 0.5)
            self.trust_costs[idx] = max(0.01, -math.log(max(trust_val, 0.01)))
            self.social_attrs[idx] = {
                "cf": vinfo.get("cf", 0.5),
                "shp": vinfo.get("shp", 0.3),
                "trust": trust_val,
                "selfish": vinfo.get("selfish", False),
                "vtype": vinfo.get("vtype", "unknown"),
            }
        
        # Adjacency list (mapped IDs)
        self.adj_list = defaultdict(list)
        self.edge_distances = {}
        for e in edges:
            s_orig, d_orig = e["src"], e["dst"]
            if s_orig not in self.id_map or d_orig not in self.id_map:
                continue
            s, d = self.id_map[s_orig], self.id_map[d_orig]
            dist = e["distance"]
            self.adj_list[s].append(d)
            self.adj_list[d].append(s)
            self.edge_distances[(s, d)] = dist
            self.edge_distances[(d, s)] = dist
        
        # Precompute distance-based delay function
        # Using simplified physical model: c_ij ≈ dist/c + retx overhead
        self.delay_cache = {}
        
        # State for current episode
        self.current = 0
        self.dst = 0
        self.remaining_budget = budget
        self.path = []
        self.step_count = 0
        self.total_delay = 0.0
        self.total_trust = 0.0
        self.h_star = {}
        
        # Find largest connected component for sampling
        self.largest_cc = self._find_largest_cc()
        print(f"  Snapshot: {self.num_nodes} nodes, "
              f"{sum(len(v) for v in self.adj_list.values())//2} edges, "
              f"avg_degree={sum(len(v) for v in self.adj_list.values())/max(self.num_nodes,1):.1f}, "
              f"largest_cc={len(self.largest_cc)} ({len(self.largest_cc)*100//max(self.num_nodes,1)}%)")
    
    def _find_largest_cc(self):
        from collections import deque
        visited = set()
        components = []
        for start in range(self.num_nodes):
            if start in visited: continue
            comp = set()
            q = deque([start])
            while q:
                u = q.popleft()
                if u in visited: continue
                visited.add(u)
                comp.add(u)
                for v in self.adj_list.get(u, []):
                    if v not in visited:
                        q.append(v)
            components.append(comp)
        return list(max(components, key=len))
    
    def delay_fn(self, src, dst):
        """Estimate link delay based on distance"""
        key = (src, dst)
        if key in self.delay_cache:
            return self.delay_cache[key]
        
        dist = self.edge_distances.get(key, 0)
        if dist == 0:
            # Compute from positions
            if src in self.positions and dst in self.positions:
                dist = np.linalg.norm(self.positions[src] - self.positions[dst])
            else:
                dist = 150.0  # default
        
        # Simple delay model: base + distance-dependent + random jitter
        # T_tx(1500B, 6Mbps) = 2ms, T_prop = dist/3e8 ≈ 0, 
        # retx probability increases with distance
        base_delay = 0.002  # 2ms base
        p_retx = min(0.9, (dist / self.comm_range) ** 2 * 0.5)
        expected_delay = base_delay * (1 + p_retx / max(1 - p_retx, 0.01))
        
        self.delay_cache[key] = expected_delay
        return expected_delay
    
    def reset(self, src=None, dst=None):
        """Reset episode. Sample src/dst from largest connected component."""
        if src is None:
            src = self.largest_cc[np.random.randint(0, len(self.largest_cc))]
        if dst is None:
            dst = self.largest_cc[np.random.randint(0, len(self.largest_cc))]
            while dst == src:
                dst = self.largest_cc[np.random.randint(0, len(self.largest_cc))]
        
        self.src, self.dst, self.current = src, dst, src
        self.remaining_budget = self.budget
        self.path = [src]
        self.total_delay = 0.0
        self.total_trust = 0.0
        self.step_count = 0
        self.h_star = self._dijkstra_trust_to_go(dst)
        return self._get_state()
    
    def step(self, action):
        r_j = self.trust_costs.get(action, 0.5)
        c_ij = self.delay_fn(self.current, action)
        self.remaining_budget -= r_j
        self.total_delay += c_ij
        self.total_trust += r_j
        self.current = action
        self.path.append(action)
        self.step_count += 1
        done = (action == self.dst or 
                self.step_count >= self.max_hops or 
                self.remaining_budget < -0.5)
        return self._get_state(), c_ij, r_j, done
    
    def get_feasible_actions(self):
        """PIP feasibility mask"""
        neighbors = self.adj_list.get(self.current, [])
        # Filter visited + trust budget
        feasible = [j for j in neighbors 
                    if j not in self.path
                    and self.trust_costs.get(j, 0.5) + self.h_star.get(j, 999) 
                        <= self.remaining_budget + 1e-9]
        if not feasible and neighbors:
            cands = [n for n in neighbors if n not in self.path] or neighbors
            feasible = [min(cands, key=lambda j: self.trust_costs.get(j, 0.5) + self.h_star.get(j, 0))]
        return feasible
    
    def _get_state(self):
        return dict(
            current=self.current, dst=self.dst,
            remaining_budget=self.remaining_budget,
            step=self.step_count,
            neighbors=self.adj_list.get(self.current, []),
            # Extra features for better spatial awareness
            current_pos=self.positions.get(self.current, np.zeros(2)),
            dst_pos=self.positions.get(self.dst, np.zeros(2)),
        )
    
    def _dijkstra_trust_to_go(self, dst):
        INF = float("inf")
        dist = defaultdict(lambda: INF)
        dist[dst] = 0.0
        pq = [(0.0, dst)]
        visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            # Reverse edges: who can reach u?
            for v in self.adj_list.get(u, []):
                nd = d + self.trust_costs.get(u, 0.5)
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dict(dist)


# ============================================================
#  Dynamic Snapshot Environment (link evolution between steps)
# ============================================================
class DynamicSnapshotEnv(SnapshotRoutingEnv):
    """Add per-step link dynamics calibrated from Veins data."""
    
    def __init__(self, snapshot, budget, max_hops=10, comm_range=300.0):
        super().__init__(snapshot, budget, max_hops, comm_range)
        self.p_break = 0.0354 + 0.000340 * self.num_nodes
        self.p_create = self.p_break * 0.3
        self.rng = np.random.default_rng()
        self.initial_adj = self._copy_adj(self.adj_list)
    
    def reset(self, src=None, dst=None):
        self.adj_list = self._copy_adj(self.initial_adj)
        return super().reset(src, dst)
    
    def step(self, action):
        state, c_ij, r_j, done = super().step(action)
        if not done:
            self._evolve_links()
        return state, c_ij, r_j, done
    
    def _evolve_links(self):
        """Markov chain link evolution"""
        to_remove = []
        for u in list(self.adj_list.keys()):
            for v in self.adj_list[u]:
                if self.rng.random() < self.p_break:
                    to_remove.append((u, v))
        for u, v in to_remove:
            if v in self.adj_list[u]:
                self.adj_list[u].remove(v)
        
        # Create new links
        nodes = list(self.positions.keys())
        n_try = int(self.num_nodes * self.p_create * 2)
        for _ in range(n_try):
            u = int(self.rng.choice(nodes))
            v = int(self.rng.choice(nodes))
            if u != v and v not in self.adj_list.get(u, []):
                if u in self.positions and v in self.positions:
                    d = np.linalg.norm(self.positions[u] - self.positions[v])
                    if d <= self.comm_range:
                        self.adj_list[u].append(v)
                        self.adj_list[v].append(u)
                        self.edge_distances[(u, v)] = d
                        self.edge_distances[(v, u)] = d
    
    def _copy_adj(self, adj):
        return {k: list(v) for k, v in adj.items()}


# ============================================================
#  Enhanced TransformerActor with spatial features
# ============================================================
class TransformerActor(nn.Module):
    """
    Same architecture as original, but state_dim=16 to include
    spatial features (distance to dst, relative position).
    """
    def __init__(self, state_dim=16, hidden_dim=128, n_heads=4, n_layers=2, max_neighbors=40):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.state_enc = nn.Linear(state_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads,
                                                dim_feedforward=hidden_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.neighbor_enc = nn.Linear(state_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, state_feat, neighbor_feats, mask=None):
        s = self.state_enc(state_feat).unsqueeze(1)  # (B,1,H)
        n = self.neighbor_enc(neighbor_feats)          # (B,K,H)
        seq = torch.cat([s, n], dim=1)                 # (B,1+K,H)
        if mask is not None:
            full_mask = torch.cat([torch.zeros(mask.size(0), 1, device=mask.device, dtype=torch.bool), mask], dim=1)
            out = self.transformer(seq, src_key_padding_mask=full_mask)
        else:
            out = self.transformer(seq)
        s_out = out[:, 0, :]        # state token
        n_out = out[:, 1:, :]       # neighbor tokens
        s_exp = s_out.unsqueeze(1).expand_as(n_out)
        logits = self.action_head(torch.cat([s_exp, n_out], dim=-1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        return logits


class DualCritic(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.vc = nn.Linear(hidden_dim, 1)
        self.vd = nn.Linear(hidden_dim, 1)
    
    def forward(self, s):
        h = self.shared(s)
        return self.vc(h).squeeze(-1), self.vd(h).squeeze(-1)


# ============================================================
#  PPO Trainer with multi-snapshot rotation
# ============================================================
class PPOTrainer:
    def __init__(self, envs, gnn_model, gnn_norm, args):
        """
        envs: list of SnapshotRoutingEnv (one per time slice)
        """
        self.envs = envs
        self.current_env_idx = 0
        self.env = envs[0]  # current active env
        self.gnn = gnn_model
        self.gnn_norm = gnn_norm
        self.args = args
        
        max_N = max(e.num_nodes for e in envs)
        self.max_N = max_N
        self.max_neighbors = args.max_neighbors
        
        self.actor = TransformerActor(state_dim=16, hidden_dim=128, 
                                       n_heads=4, n_layers=2,
                                       max_neighbors=self.max_neighbors)
        self.critic = DualCritic(state_dim=16, hidden_dim=128)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.lam = 1.0  # Lagrangian multiplier
        self.eta = args.eta_lambda
    
    def rotate_env(self):
        """Switch to next snapshot environment"""
        self.current_env_idx = (self.current_env_idx + 1) % len(self.envs)
        self.env = self.envs[self.current_env_idx]
    
    def state_feat(self, s):
        """16-dim state feature with spatial awareness"""
        cur_pos = s.get("current_pos", np.zeros(2))
        dst_pos = s.get("dst_pos", np.zeros(2))
        
        # Euclidean distance to destination (normalized by comm_range)
        dist_to_dst = np.linalg.norm(cur_pos - dst_pos) / self.env.comm_range
        # Direction to destination (angle)
        diff = dst_pos - cur_pos
        angle_to_dst = math.atan2(diff[1], diff[0]) / math.pi  # normalized to [-1, 1]
        
        return np.array([
            s["current"] / max(self.max_N, 1),
            s["dst"] / max(self.max_N, 1),
            s["remaining_budget"] / max(self.env.budget, 1e-6),
            s["step"] / self.env.max_hops,
            len(s["neighbors"]) / self.max_neighbors,
            0.5,  # avg_trust placeholder
            1.0 if s["current"] == s["dst"] else 0.0,
            self.lam / 10.0,
            # ---- NEW: spatial features ----
            min(dist_to_dst / 10.0, 1.0),    # distance to dst (normalized)
            angle_to_dst,                       # direction to dst
            cur_pos[0] / 5000.0 if len(cur_pos) > 0 else 0,  # normalized x
            cur_pos[1] / 5000.0 if len(cur_pos) > 1 else 0,  # normalized y
            dst_pos[0] / 5000.0 if len(dst_pos) > 0 else 0,
            dst_pos[1] / 5000.0 if len(dst_pos) > 1 else 0,
            s["step"] / max(dist_to_dst + 1, 1),  # progress ratio
            self.env.num_nodes / 1000.0,           # density indicator
        ], dtype=np.float32)
    
    def neighbor_feat(self, s, feasible):
        """16-dim neighbor features with spatial info"""
        f = np.zeros((self.max_neighbors, 16), dtype=np.float32)
        cur_pos = s.get("current_pos", np.zeros(2))
        dst_pos = s.get("dst_pos", np.zeros(2))
        
        for i, n in enumerate(feasible[:self.max_neighbors]):
            r = self.env.trust_costs.get(n, 0.5)
            h = self.env.h_star.get(n, 0)
            d = self.env.delay_fn(s["current"], n)
            n_pos = self.env.positions.get(n, np.zeros(2))
            
            # Distance from neighbor to destination
            n_to_dst = np.linalg.norm(n_pos - dst_pos) / self.env.comm_range
            # Distance from current to neighbor
            cur_to_n = np.linalg.norm(n_pos - cur_pos) / self.env.comm_range
            # Progress: how much closer does this neighbor get us to dst?
            cur_to_dst = np.linalg.norm(cur_pos - dst_pos)
            progress = (cur_to_dst - np.linalg.norm(n_pos - dst_pos)) / max(cur_to_dst, 1)
            
            f[i] = [
                n / max(self.max_N, 1),
                r,
                h / max(self.env.budget, 1e-6),
                d * 100,
                1.0 if n == s["dst"] else 0.0,
                1.0,  # is_feasible
                s["remaining_budget"] / max(self.env.budget, 1e-6),
                s["step"] / self.env.max_hops,
                # ---- NEW: spatial features ----
                min(n_to_dst / 10.0, 1.0),    # neighbor's distance to dst
                min(cur_to_n, 1.0),            # distance to this neighbor
                progress,                       # forward progress
                n_pos[0] / 5000.0 if len(n_pos) > 0 else 0,
                n_pos[1] / 5000.0 if len(n_pos) > 1 else 0,
                self.env.social_attrs.get(n, {}).get("cf", 0.5),
                self.env.social_attrs.get(n, {}).get("shp", 0.3),
                1.0 if self.env.social_attrs.get(n, {}).get("selfish", False) else 0.0,
            ]
        return f
    
    def collect(self, n_ep=64):
        """Collect episodes with snapshot rotation"""
        data = []
        for ep in range(n_ep):
            # Rotate environment every few episodes
            if ep % max(1, n_ep // len(self.envs)) == 0:
                self.rotate_env()
            
            s = self.env.reset()
            ep_data = []
            while True:
                feas = self.env.get_feasible_actions()
                if not feas:
                    break
                sf = torch.tensor(self.state_feat(s), dtype=torch.float32).unsqueeze(0)
                nf = torch.tensor(self.neighbor_feat(s, feas), dtype=torch.float32).unsqueeze(0)
                mask = torch.ones(1, self.max_neighbors, dtype=torch.bool)
                mask[0, :len(feas)] = False
                
                with torch.no_grad():
                    logits = self.actor(sf, nf, mask)
                    probs = F.softmax(logits, dim=-1)
                    dist_obj = Categorical(probs)
                    a_idx = dist_obj.sample()
                    logp = dist_obj.log_prob(a_idx)
                    vc, vd = self.critic(sf)
                
                action = feas[min(a_idx.item(), len(feas) - 1)]
                s2, c_ij, r_j, done = self.env.step(action)
                ep_data.append((sf.squeeze(0), nf.squeeze(0), mask.squeeze(0),
                                a_idx.item(), logp.item(), c_ij, r_j,
                                vc.item(), vd.item(), done))
                s = s2
                if done:
                    break
            
            if ep_data:
                data.append(ep_data)
        return data
    
    def gae(self, rews, vals, dones, gamma=1.0, lam=0.95):
        """Generalized Advantage Estimation"""
        T = len(rews)
        advs = np.zeros(T)
        last = 0
        for t in reversed(range(T)):
            nv = 0 if dones[t] else (vals[t + 1] if t + 1 < T else 0)
            delta = rews[t] + gamma * nv - vals[t]
            advs[t] = last = delta + gamma * lam * (0 if dones[t] else last)
        returns = advs + np.array(vals[:T])
        return advs, returns
    
    def update(self, rollouts):
        """PPO-Lagrangian update"""
        all_sf, all_nf, all_mask = [], [], []
        all_aidx, all_logp_old = [], []
        all_adv_c, all_ret_c, all_adv_d, all_ret_d = [], [], [], []
        
        total_delay = 0
        total_violation = 0
        total_episodes = len(rollouts)
        total_hops = 0
        n_reached = 0
        
        for ep_data in rollouts:
            T = len(ep_data)
            c_rews = [-(x[5]) for x in ep_data]  # negative delay as reward
            d_rews = [x[6] for x in ep_data]       # trust cost
            vc_vals = [x[7] for x in ep_data]
            vd_vals = [x[8] for x in ep_data]
            dones = [x[9] for x in ep_data]
            
            adv_c, ret_c = self.gae(c_rews, vc_vals, dones)
            adv_d, ret_d = self.gae(d_rews, vd_vals, dones, gamma=1.0, lam=0.95)
            
            for t in range(T):
                all_sf.append(ep_data[t][0])
                all_nf.append(ep_data[t][1])
                all_mask.append(ep_data[t][2])
                all_aidx.append(ep_data[t][3])
                all_logp_old.append(ep_data[t][4])
                all_adv_c.append(adv_c[t])
                all_ret_c.append(ret_c[t])
                all_adv_d.append(adv_d[t])
                all_ret_d.append(ret_d[t])
            
            total_delay += sum(x[5] for x in ep_data)
            total_violation += max(0, sum(x[6] for x in ep_data) - self.env.budget)
            total_hops += T
            if ep_data[-1][9] and self.env.current == self.env.dst:
                n_reached += 1
        
        if not all_sf:
            return 0, 0, 0, 0
        
        sf_t = torch.stack(all_sf)
        nf_t = torch.stack(all_nf)
        mask_t = torch.stack(all_mask)
        aidx_t = torch.tensor(all_aidx, dtype=torch.long)
        logp_old_t = torch.tensor(all_logp_old, dtype=torch.float32)
        
        adv_c_t = torch.tensor(all_adv_c, dtype=torch.float32)
        adv_d_t = torch.tensor(all_adv_d, dtype=torch.float32)
        ret_c_t = torch.tensor(all_ret_c, dtype=torch.float32)
        ret_d_t = torch.tensor(all_ret_d, dtype=torch.float32)
        
        # Combined advantage: A_c - lambda * A_d
        combined_adv = adv_c_t - self.lam * adv_d_t
        combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)
        
        # PPO update (K epochs)
        clip_eps = 0.2
        for _ in range(4):
            logits = self.actor(sf_t, nf_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            dist_obj = Categorical(probs)
            logp_new = dist_obj.log_prob(aidx_t)
            entropy = dist_obj.entropy().mean()
            
            ratio = (logp_new - logp_old_t).exp()
            surr1 = ratio * combined_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * combined_adv
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.opt_a.step()
            
            vc, vd = self.critic(sf_t)
            critic_loss = F.mse_loss(vc, ret_c_t) + F.mse_loss(vd, ret_d_t)
            
            self.opt_c.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.opt_c.step()
        
        # Dual ascent on lambda
        avg_violation = total_violation / max(total_episodes, 1)
        self.lam = max(0, self.lam + self.eta * avg_violation)
        
        avg_delay = total_delay / max(total_episodes, 1)
        avg_hops = total_hops / max(total_episodes, 1)
        reach_rate = n_reached / max(total_episodes, 1)
        
        return avg_delay, avg_violation, avg_hops, reach_rate
    
    def schedule(self, ep, total):
        """Cosine entropy schedule"""
        progress = ep / max(total, 1)
        ent = 0.02 * (1 + math.cos(math.pi * progress)) / 2 + 0.005
        return ent


# ============================================================
#  GNN Delay Integration
# ============================================================
def load_gnn_delay(model_path, env):
    """Load GNN model and use it to update env delay function"""
    if not os.path.exists(model_path) or not HAS_GNN:
        print(f"  [GNN] Not found or not available: {model_path}")
        return None, None
    
    ckpt = torch.load(model_path, map_location="cpu")
    norm_stats = ckpt.get("norm_stats", {})
    
    # Try to load MLP (EdgeFeatureNet)
    if "mlp_state_dict" in ckpt:
        mlp = EdgeFeatureNet(input_dim=6, hidden_dim=128, num_layers=3)
        mlp.load_state_dict(ckpt["mlp_state_dict"])
        mlp.eval()
        print(f"  [GNN] Loaded MLP from {model_path}")
        return mlp, norm_stats
    
    print(f"  [GNN] No MLP found in {model_path}, using distance-based delays")
    return None, None


def update_env_delays_with_gnn(env, gnn_model, norm_stats):
    """Replace env delay function with GNN predictions"""
    if gnn_model is None:
        return
    
    dist_mean = norm_stats.get("dist_mean", 470.0)
    dist_std = norm_stats.get("dist_std", 213.0)
    snir_mean = norm_stats.get("snir_mean", 760.0)
    snir_std = norm_stats.get("snir_std", 9755.0)
    dist_lo = norm_stats.get("dist_lo", 8.4)
    dist_hi = norm_stats.get("dist_hi", 816.0)
    snir_lo = norm_stats.get("snir_lo", 3.1)
    snir_hi = norm_stats.get("snir_hi", 9182.0)
    
    print(f"  [GNN] Precomputing delays for {len(env.edge_distances)} edges...")
    count = 0
    with torch.no_grad():
        for (s, d), distance in env.edge_distances.items():
            # Estimate SNIR from distance (simplified)
            snir = max(3.0, 1000.0 * (1 - distance / env.comm_range))
            
            feat = torch.tensor([[
                (distance - dist_mean) / dist_std,
                (snir - snir_mean) / snir_std,
                math.log1p(distance) / 7.0,
                math.log1p(snir) / 10.0,
                distance / 1000.0,
                snir / (snir_hi + 1),
            ]], dtype=torch.float32)
            
            pred = gnn_model(feat)
            delay = pred[0, 0].item()
            env.delay_cache[(s, d)] = max(0.0005, delay)  # min 0.5ms
            count += 1
    
    print(f"  [GNN] Updated {count} edge delays")


# ============================================================
#  Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="PPO training on Optics Valley snapshots")
    p.add_argument("--snapshots", default="optics_valley_snapshots.json")
    p.add_argument("--gnn-model", default="gnn_delay_model.pt")
    p.add_argument("--output", default="ppo_policy_optics_valley.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--max-hops", type=int, default=10)
    p.add_argument("--max-neighbors", type=int, default=40)
    p.add_argument("--trust-budget", type=float, default=4.6052)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--eta-lambda", type=float, default=5e-3)
    p.add_argument("--dynamic", action="store_true", help="Use dynamic link evolution")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("PPO Training on Optics Valley Snapshots")
    print("=" * 60)
    
    # Load snapshots
    print(f"\nLoading snapshots from {args.snapshots}...")
    with open(args.snapshots) as f:
        all_snapshots = json.load(f)
    
    # Build environments
    envs = []
    snapshot_names = sorted(all_snapshots.keys())
    print(f"Found {len(snapshot_names)} snapshots: {snapshot_names}")
    
    EnvClass = DynamicSnapshotEnv if args.dynamic else SnapshotRoutingEnv
    
    for name in snapshot_names:
        snap = all_snapshots[name]
        print(f"\n  [{name}] t={snap['time']}s:")
        env = EnvClass(snap, args.trust_budget, args.max_hops, comm_range=300.0)
        envs.append(env)
    
    # Load GNN model for delay estimation
    print(f"\nLoading GNN model...")
    gnn_model, gnn_norm = load_gnn_delay(args.gnn_model, envs[0])
    
    # Update delays in all envs with GNN predictions
    if gnn_model is not None:
        for i, env in enumerate(envs):
            print(f"  Updating env {i} delays...")
            update_env_delays_with_gnn(env, gnn_model, gnn_norm)
    
    # Initialize trainer
    print(f"\nInitializing PPO trainer...")
    print(f"  max_hops={args.max_hops}, max_neighbors={args.max_neighbors}")
    print(f"  trust_budget={args.trust_budget}, lr={args.lr_actor}")
    
    trainer = PPOTrainer(envs, gnn_model, gnn_norm, args)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs × {args.episodes} episodes")
    print(f"{'='*60}\n")
    
    best_delay = float("inf")
    t_start = time.time()
    
    for epoch in range(args.epochs):
        rollouts = trainer.collect(args.episodes)
        avg_delay, avg_viol, avg_hops, reach_rate = trainer.update(rollouts)
        
        # Log
        env_name = snapshot_names[trainer.current_env_idx]
        elapsed = time.time() - t_start
        print(f"Ep {epoch+1:3d}/{args.epochs} [{env_name}] "
              f"delay={avg_delay:.6f} viol={avg_viol:.3f} "
              f"hops={avg_hops:.1f} reach={reach_rate:.1%} "
              f"λ={trainer.lam:.4f} [{elapsed:.0f}s]")
        
        # Save best
        if avg_delay < best_delay and avg_delay > 0 and reach_rate > 0.1:
            best_delay = avg_delay
            save_dict = {
                "actor_state": trainer.actor.state_dict(),
                "critic_state": trainer.critic.state_dict(),
                "lambda": trainer.lam,
                "best_delay": best_delay,
                "epoch": epoch,
                "state_dim": 16,
                "max_neighbors": args.max_neighbors,
                "max_hops": args.max_hops,
                "snapshots": snapshot_names,
                "training_scene": "optics_valley",
            }
            torch.save(save_dict, args.output)
            print(f"  ★ Saved best model (delay={best_delay:.6f})")
    
    # Final save
    final_path = args.output.replace(".pt", "_final.pt")
    torch.save({
        "actor_state": trainer.actor.state_dict(),
        "critic_state": trainer.critic.state_dict(),
        "lambda": trainer.lam,
        "final_delay": avg_delay,
        "epochs": args.epochs,
        "state_dim": 16,
        "max_neighbors": args.max_neighbors,
        "max_hops": args.max_hops,
    }, final_path)
    
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best delay: {best_delay:.6f}")
    print(f"Final λ: {trainer.lam:.4f}")
    print(f"Saved: {args.output} (best), {final_path} (final)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
