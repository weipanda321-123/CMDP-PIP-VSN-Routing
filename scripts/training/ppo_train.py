#!/usr/bin/env python3
"""
PPO-Lagrangian Policy Training for VSN Routing CMDP
(OJCOMS Section III-B, Fig.2, Table 2)
"""
import os, sys, math, json, heapq, argparse, numpy as np
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn_train import EdgeFeatureNet

class RoutingEnv:
    def __init__(self, num_nodes, adj_list, trust_costs, delay_fn, budget, max_hops=20):
        self.num_nodes = num_nodes
        self.adj_list = adj_list
        self.trust_costs = trust_costs
        self.delay_fn = delay_fn
        self.budget = budget
        self.max_hops = max_hops
        self.h_star = {}

    def reset(self, src, dst):
        self.src, self.dst, self.current = src, dst, src
        self.remaining_budget = self.budget
        self.path = [src]
        self.total_delay = self.total_trust = 0.0
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
        done = (action == self.dst or self.step_count >= self.max_hops or self.remaining_budget < 0)
        return self._get_state(), c_ij, r_j, done

    def get_feasible_actions(self):
        neighbors = self.adj_list.get(self.current, [])
        feasible = [j for j in neighbors if j not in self.path
                     and self.trust_costs.get(j,0.5) + self.h_star.get(j,999) <= self.remaining_budget + 1e-9]
        if not feasible and neighbors:
            cands = [n for n in neighbors if n not in self.path] or neighbors
            feasible = [min(cands, key=lambda j: self.trust_costs.get(j,0.5)+self.h_star.get(j,0))]
        return feasible

    def _get_state(self):
        return dict(current=self.current, dst=self.dst, remaining_budget=self.remaining_budget,
                    step=self.step_count, neighbors=self.adj_list.get(self.current, []))

    def _dijkstra_trust_to_go(self, dst):
        INF = float("inf")
        dist = defaultdict(lambda: INF); dist[dst] = 0.0
        pq = [(0.0, dst)]; visited = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            for v in range(self.num_nodes):
                if u in self.adj_list.get(v, []):
                    nd = d + self.trust_costs.get(u, 0.5)
                    if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))
        return dict(dist)



# ---- Dynamic Routing Environment with link evolution ----
class DynamicRoutingEnv(RoutingEnv):
    """
    Extends RoutingEnv with per-step link dynamics.
    At each routing step, adjacency evolves via Markov chain
    calibrated from Veins data.
    
    p_break(N) = 0.0354 + 0.000340 * N  (per 0.1s step)
    """
    def __init__(self, num_nodes, adj_list, trust_costs, delay_fn, budget, 
                 max_hops=20, positions=None, comm_range=300.0, p_break=None):
        super().__init__(num_nodes, adj_list, trust_costs, delay_fn, budget, max_hops)
        self.positions = positions
        self.comm_range = comm_range
        if p_break is None:
            p_break = 0.0354 + 0.000340 * num_nodes
        self.p_break = p_break
        self.p_create = p_break * 0.3
        self.rng = np.random.default_rng()
        self.initial_adj = dict(adj_list)
    
    def reset(self, src, dst):
        # Reset adjacency to initial state (with small perturbation)
        self.adj_list = self._copy_adj(self.initial_adj)
        return super().reset(src, dst)
    
    def step(self, action):
        # First take the action on current topology
        state, cost, trust, done = super().step(action)
        
        # Then evolve topology for next step (simulates 0.1s passing)
        if not done:
            self._evolve_links()
            # Update h_star with new topology
            self.h_star = self._dijkstra_trust_to_go(self.dst)
        
        return state, cost, trust, done
    
    def _evolve_links(self):
        if self.positions is None:
            return  # fallback: no evolution without positions
        
        N = self.num_nodes
        # Current edge set
        edge_set = set()
        for i, nbs in self.adj_list.items():
            for j in nbs:
                edge_set.add((min(i,j), max(i,j)))
        
        new_edges = set()
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                if d > self.comm_range:
                    continue
                e = (i, j)
                if e in edge_set:
                    if self.rng.random() > self.p_break:
                        new_edges.add(e)
                else:
                    if self.rng.random() < self.p_create:
                        new_edges.add(e)
        
        # Rebuild adj
        new_adj = defaultdict(list)
        for i, j in new_edges:
            new_adj[i].append(j)
            new_adj[j].append(i)
        
        # Keep current node reachable
        if len(new_adj[self.current]) == 0 and self.positions is not None:
            dists = [(np.linalg.norm(self.positions[self.current] - self.positions[j]), j)
                     for j in range(N) if j != self.current]
            dists.sort()
            j = dists[0][1]
            new_adj[self.current].append(j)
            new_adj[j].append(self.current)
        
        self.adj_list = dict(new_adj)
    
    def _copy_adj(self, adj):
        return {k: list(v) for k, v in adj.items()}


class TransformerActor(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, n_heads=4, n_layers=2, max_neighbors=20):
        super().__init__()
        self.max_neighbors = max_neighbors
        self.state_embed = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
                                                dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.action_head = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.neighbor_embed = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU())

    def forward(self, state_feat, neighbor_feats, mask=None):
        s = self.state_embed(state_feat).unsqueeze(1)
        n = self.neighbor_embed(neighbor_feats)
        seq = torch.cat([s, n], dim=1)
        out = self.transformer(seq)
        s_out = out[:,0:1,:].expand(-1, self.max_neighbors, -1)
        n_out = out[:,1:,:]
        logits = self.action_head(torch.cat([s_out, n_out], dim=-1)).squeeze(-1)
        if mask is not None: logits = logits.masked_fill(~mask, -1e9)
        return logits

class DualCritic(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.vc_head = nn.Linear(hidden_dim, 1)
        self.vd_head = nn.Linear(hidden_dim, 1)
    def forward(self, s):
        h = self.shared(s)
        return self.vc_head(h).squeeze(-1), self.vd_head(h).squeeze(-1)

class PPOTrainer:
    def __init__(self, env, gnn_model, gnn_norm, args):
        self.env, self.args = env, args
        sd = 8
        self.actor = TransformerActor(state_dim=sd, hidden_dim=args.hidden_dim, n_heads=args.n_heads,
                                       n_layers=args.n_transformer_layers, max_neighbors=args.max_neighbors)
        self.critic = DualCritic(state_dim=sd, hidden_dim=args.hidden_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.lam = args.lambda_init
        self.eta_lam = args.eta_lambda
        self.alpha_ent = args.alpha_ent
        self.alpha_ent_init = args.alpha_ent
        self.budget_tight = env.budget
        self.budget_loose = env.budget * 1.5

    def state_feat(self, s):
        nb = s["neighbors"]; nn_ = len(nb) if nb else 0
        at = np.mean([self.env.trust_costs.get(n,0.5) for n in nb]) if nb else 0.5
        return np.array([s["current"]/max(self.env.num_nodes,1), s["dst"]/max(self.env.num_nodes,1),
                         s["remaining_budget"]/max(self.env.budget,1e-6), s["step"]/self.env.max_hops,
                         nn_/self.args.max_neighbors, at, 1.0 if s["current"]==s["dst"] else 0.0,
                         self.lam/10.0], dtype=np.float32)

    def neighbor_feat(self, s, feas):
        mx = self.args.max_neighbors; f = np.zeros((mx,8),dtype=np.float32); m = np.zeros(mx,dtype=bool)
        for i, n in enumerate(s["neighbors"][:mx]):
            r = self.env.trust_costs.get(n,0.5); h = self.env.h_star.get(n,1.0)
            d = self.env.delay_fn(s["current"], n)
            f[i] = [n/max(self.env.num_nodes,1), r, h/max(self.env.budget,1e-6), d*100,
                     1.0 if n==s["dst"] else 0.0, 1.0 if n in feas else 0.0,
                     s["remaining_budget"]/max(self.env.budget,1e-6), s["step"]/self.env.max_hops]
            m[i] = True
        return f, m

    def collect(self, n_ep=64):
        R = {k:[] for k in ["sf","nf","mk","ai","lp","dc","tc","dn","vc","vd"]}
        for _ in range(n_ep):
            src = np.random.randint(0, self.env.num_nodes)
            dst = np.random.randint(0, self.env.num_nodes)
            while dst == src: dst = np.random.randint(0, self.env.num_nodes)
            s = self.env.reset(src, dst); done = False
            while not done:
                feas = self.env.get_feasible_actions()
                if not feas: break
                sf = self.state_feat(s); nf, mk = self.neighbor_feat(s, feas)
                st = torch.FloatTensor(sf).unsqueeze(0)
                nt = torch.FloatTensor(nf).unsqueeze(0)
                mt = torch.BoolTensor(mk).unsqueeze(0)
                with torch.no_grad():
                    logits = self.actor(st, nt, mt); vc, vd = self.critic(st)
                probs = F.softmax(logits, dim=-1); cat = Categorical(probs)
                nb = s["neighbors"][:self.args.max_neighbors]
                if np.random.rand() < self.args.epsilon:
                    fi = [i for i,n in enumerate(nb) if n in feas]
                    ai = np.random.choice(fi) if fi else cat.sample().item()
                else: ai = cat.sample().item()
                lp = cat.log_prob(torch.tensor(ai))
                if ai >= len(nb): break
                ns, cij, rj, done = self.env.step(nb[ai])
                for k,v in zip(R.keys(),[sf,nf,mk,ai,lp.item(),cij,rj,done,vc.item(),vd.item()]):
                    R[k].append(v)
                s = ns
        return R

    def gae(self, rews, vals, dones, gamma=1.0, lam=0.95):
        n = len(rews); adv = np.zeros(n); lg = 0
        for t in reversed(range(n)):
            nv = 0 if (t==n-1 or dones[t]) else vals[t+1]
            d = rews[t] + gamma*nv - vals[t]
            adv[t] = lg = d + gamma*lam*(1-dones[t])*lg
        return adv, adv + np.array(vals)

    def update(self, R):
        if len(R["sf"]) < 4: return {}
        dc = np.array(R["dc"]); tc = np.array(R["tc"])
        ac, rc = self.gae(dc, R["vc"], R["dn"], lam=self.args.gae_lambda)
        ad, rd = self.gae(tc, R["vd"], R["dn"], lam=self.args.gae_lambda)
        A = ac + self.lam * ad; A = (A - A.mean())/(A.std()+1e-8)
        ST = torch.FloatTensor(np.array(R["sf"])); NT = torch.FloatTensor(np.array(R["nf"]))
        MT = torch.BoolTensor(np.array(R["mk"])); AI = torch.LongTensor(R["ai"])
        OLP = torch.FloatTensor(R["lp"]); AT = torch.FloatTensor(A)
        RC = torch.FloatTensor(rc); RD = torch.FloatTensor(rd)
        n = len(ST); met = dict(actor_loss=0, critic_loss=0, entropy=0)
        for _ in range(self.args.ppo_epochs):
            pm = torch.randperm(n)
            for i in range(0, n, self.args.minibatch_size):
                ix = pm[i:i+self.args.minibatch_size]
                lo = self.actor(ST[ix], NT[ix], MT[ix])
                pr = F.softmax(lo, dim=-1); ca = Categorical(pr)
                nlp = ca.log_prob(AI[ix]); ent = ca.entropy().mean()
                rat = torch.exp(nlp - OLP[ix])
                s1 = rat * AT[ix]; s2 = torch.clamp(rat, 1-self.args.ppo_clip, 1+self.args.ppo_clip)*AT[ix]
                al = torch.max(s1, s2).mean()
                vc, vd = self.critic(ST[ix])
                cl = F.mse_loss(vc, RC[ix]) + F.mse_loss(vd, RD[ix])
                loss = al + 0.5*cl - self.alpha_ent*ent
                self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor_opt.step(); self.critic_opt.step()
                met["actor_loss"] += al.item(); met["critic_loss"] += cl.item(); met["entropy"] += ent.item()
        nu = max(1, self.args.ppo_epochs * (n//self.args.minibatch_size+1))
        for k in met: met[k] /= nu
        tv = np.sum(tc)/max(sum(R["dn"]),1) - self.env.budget
        self.lam = max(0.0, self.lam + self.eta_lam * tv)
        met.update(dict(lam=self.lam, avg_delay=np.mean(dc), avg_trust=np.mean(tc), trust_viol=max(0,tv)))
        return met

    def schedule(self, ep, total):
        p = ep/total; self.alpha_ent = self.alpha_ent_init*(0.5*(1+math.cos(math.pi*p)))
        if self.args.budget_curriculum:
            r = min(1.0, ep/(total*0.5))
            self.env.budget = self.budget_loose*(1-r) + self.budget_tight*r

def build_graph(data_path, N=50):
    adj = defaultdict(list); dm = {}; np.random.seed(42)
    try:
        with open(data_path) as f:
            f.readline()
            for l in f:
                p = l.strip().split(",")
                if len(p)>=6:
                    s,d,dl = int(p[1]),int(p[2]),float(p[5])
                    if d not in adj[s]: adj[s].append(d)
                    if s not in adj[d]: adj[d].append(s)
                    dm[(s,d)] = dm[(d,s)] = dl
    except FileNotFoundError: pass
    for i in range(N):
        if len(adj[i])<2:
            for _ in range(3):
                j = np.random.randint(0,N)
                if j!=i and j not in adj[i]: adj[i].append(j); adj[j].append(i)
    tc = {i: -np.log(max(np.random.uniform(0.3,0.95),0.01)) for i in range(N)}
    def dfn(s,d): return dm.get((s,d), np.random.exponential(0.005))
    return dict(adj), tc, dfn

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="gnn_training_data.csv")
    p.add_argument("--gnn-model", default="gnn_delay_model.pt")
    p.add_argument("--output", default="ppo_policy.pt")
    p.add_argument("--num-nodes", type=int, default=50)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--episodes-per-epoch", type=int, default=64)
    p.add_argument("--ppo-clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--alpha-ent", type=float, default=0.02)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-transformer-layers", type=int, default=2)
    p.add_argument("--max-neighbors", type=int, default=20)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--lambda-init", type=float, default=1.0)
    p.add_argument("--eta-lambda", type=float, default=5e-3)
    p.add_argument("--trust-budget", type=float, default=2.3026)
    p.add_argument("--max-hops", type=int, default=20)
    p.add_argument("--budget-curriculum", action="store_true")
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("="*60); print("  PPO-Lagrangian Training (OJCOMS Sec.III-B, Table 2)"); print("="*60)
    gnn = None
    if os.path.exists(args.gnn_model):
        ck = torch.load(args.gnn_model, map_location="cpu"); cfg = ck["config"]
        gnn = EdgeFeatureNet(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"])
        gnn.load_state_dict(ck["model_state"]); gnn.eval()
        print(f"  GNN loaded from {args.gnn_model}")
    adj, tc, dfn = build_graph(args.data, args.num_nodes)
    print(f"  Graph: {args.num_nodes} nodes, {sum(len(v) for v in adj.values())} edges")
    env = RoutingEnv(args.num_nodes, adj, tc, dfn, args.trust_budget, args.max_hops)
    tr = PPOTrainer(env, gnn, None, args)
    print(f"  clip={args.ppo_clip} GAE_lam={args.gae_lambda} ent={args.alpha_ent} lr={args.lr_actor}")
    print(f"  budget B={args.trust_budget} lambda_init={args.lambda_init} eta={args.eta_lambda}")
    print()
    best = float("inf"); hist = []
    for ep in range(args.epochs):
        tr.schedule(ep, args.epochs); args.epsilon = max(0.01, args.epsilon*args.epsilon_decay)
        R = tr.collect(args.episodes_per_epoch)
        if len(R["sf"])<4: continue
        m = tr.update(R); hist.append(m)
        if m.get("avg_delay",999) < best and ep > 10:
            best = m["avg_delay"]
            torch.save(dict(actor_state=tr.actor.state_dict(), critic_state=tr.critic.state_dict(),
                            lam=tr.lam, config=dict(state_dim=8, hidden_dim=args.hidden_dim,
                            n_heads=args.n_heads, n_transformer_layers=args.n_transformer_layers,
                            max_neighbors=args.max_neighbors), epoch=ep, history=hist), args.output)
        if ep % 10 == 0:
            print(f"  Ep {ep:4d}: delay={m.get('avg_delay',0):.6f} trust={m.get('avg_trust',0):.4f} "
                  f"viol={m.get('trust_viol',0):.4f} lam={m.get('lam',0):.4f} "
                  f"ent={m.get('entropy',0):.3f} aL={m.get('actor_loss',0):.4f} cL={m.get('critic_loss',0):.4f}")
    print(f"\nDone! Best delay={best:.6f} Final lam={tr.lam:.4f} Saved: {args.output}")

if __name__ == "__main__": main()
