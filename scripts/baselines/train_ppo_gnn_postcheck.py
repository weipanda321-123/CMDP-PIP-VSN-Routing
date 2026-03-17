#!/usr/bin/env python3
"""
PPO-GNN with Post-Decision Constraint Validation
Adapted from: Frontiers PPO-GNN (2025) constraint validation mechanism
  - Same Transformer+GNN architecture as ours
  - NO PIP mask during routing
  - After path completion, check trust constraint; reject if violated
  - Training uses Lagrangian penalty (same as PPO-Lag)
  - Inference adds post-decision feasibility check
This represents the state-of-the-art "learn then validate" paradigm.
"""
import sys, os, time, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_comparison import build_dynamic_graph
from ppo_train import RoutingEnv, TransformerActor

CSV = '../simulations/manhattan/gnn_training_data_750cars_1500s.csv'
# Use PPO-Lag trained policy (already trained without PIP)
POLICY_LAG = '../simulations/manhattan/ppo_lag_baseline.pt'
# Also use our PIP-trained policy for PPO-NoPIP+PostCheck variant
POLICY_PIP = '../simulations/manhattan/ppo_policy_active.pt'

N_LIST = [50, 75, 100, 150, 200, 300, 400]
SEEDS = [42, 123, 456, 789, 1024]
budget = 2.3
max_hops = 20
max_n = 20
n_queries = 100

def load_actor(policy_path):
    ck = torch.load(policy_path, map_location='cpu', weights_only=False)
    cfg = ck['config']
    actor = TransformerActor(state_dim=cfg['state_dim'], hidden_dim=cfg['hidden_dim'],
                              n_heads=cfg['n_heads'], n_layers=cfg['n_transformer_layers'],
                              max_neighbors=cfg['max_neighbors'])
    actor.load_state_dict(ck['actor_state']); actor.eval()
    return actor, cfg['max_neighbors']

def eval_with_postcheck(actor, adj, dfn, tc, queries, N, use_pip=False):
    """Route without PIP, then post-validate trust constraint."""
    results = []
    for src, dst in queries:
        if src >= N or dst >= N:
            results.append(dict(delay=float('inf'), feasible=False, hops=0, latency_ms=0))
            continue
        env = RoutingEnv(N, adj, tc, dfn, budget, max_hops)
        state = env.reset(src, dst)
        t0 = time.perf_counter()
        for _ in range(max_hops):
            if env.current == dst: break
            if use_pip:
                feas = env.get_feasible_actions()
            else:
                feas = [n for n in adj.get(env.current, []) if n not in env.path]
            if not feas: break
            nb = state['neighbors'][:max_n]
            if not nb: break
            nn_ = len(nb)
            at = np.mean([tc.get(n,0.5) for n in nb])
            sf = torch.FloatTensor([[env.current/max(N,1), dst/max(N,1),
                  state['remaining_budget']/max(budget,1e-6), state['step']/max_hops,
                  nn_/max_n, at, 1.0 if env.current==dst else 0.0, 0.0]])
            nf = torch.zeros(1, max_n, 8); mk = torch.zeros(1, max_n, dtype=torch.bool)
            for i, n in enumerate(nb):
                nf[0,i] = torch.tensor([n/max(N,1), tc.get(n,0.5),
                           env.h_star.get(n,1.0)/max(budget,1e-6), dfn(env.current,n)*100,
                           1.0 if n==dst else 0.0, 1.0 if n in feas else 0.0,
                           state['remaining_budget']/max(budget,1e-6), state['step']/max_hops])
                mk[0,i] = True
            with torch.no_grad():
                logits = actor(sf, nf, mk)
            ml = logits[0,:nn_].clone()
            if use_pip:
                fs = set(feas)
                for i, n in enumerate(nb):
                    if n not in fs: ml[i] = -1e9
            ai = ml.argmax().item()
            if ai >= len(nb): break
            state, _, _, done = env.step(nb[ai])
            if done: break
        elapsed = (time.perf_counter() - t0) * 1000
        reached = env.current == dst
        # POST-DECISION CONSTRAINT VALIDATION (key difference from PIP)
        trust_ok = env.total_trust <= budget + 1e-9
        feasible = reached and trust_ok
        results.append(dict(delay=env.total_delay if reached else float('inf'),
                           feasible=feasible, hops=len(env.path)-1, latency_ms=elapsed,
                           reached=reached, trust_ok=trust_ok))
    return results

# Load both policies
print("Loading policies...")
actor_lag, _ = load_actor(POLICY_LAG)
actor_pip, _ = load_actor(POLICY_PIP)

print(f"\n{'N':>4s}  {'Method':>30s}  {'Feas%':>8s}  {'Reach%':>8s}  {'TrustOK%':>10s}")
print("-" * 70)

for Nv in N_LIST:
    for method_name, actor, use_pip in [
        ('PPO-GNN+PostCheck (Frontiers25)', actor_lag, False),
        ('PPO+PIP (Ours)', actor_pip, True),
        ('PPO-NoPIP', actor_pip, False),
    ]:
        feas_all = []; reach_all = []; tok_all = []
        for seed in SEEDS:
            adj, tc2, dfn, _, _, _ = build_dynamic_graph(CSV, Nv, seed)
            nodes = list(adj.keys())
            np.random.seed(seed)
            queries = [(np.random.choice(nodes), np.random.choice(nodes)) for _ in range(n_queries*2)]
            queries = [(s,d) for s,d in queries if s!=d][:n_queries]
            rs = eval_with_postcheck(actor, adj, dfn, tc2, queries, max(nodes)+1, use_pip=use_pip)
            feas_all.append(sum(1 for r in rs if r['feasible']))
            reach_all.append(sum(1 for r in rs if r.get('reached',False)))
            tok_all.append(sum(1 for r in rs if r.get('reached',False) and r.get('trust_ok',False)))
        mf = np.mean(feas_all); sf = np.std(feas_all)
        mr = np.mean(reach_all); mt = np.mean(tok_all)
        print(f"{Nv:4d}  {method_name:>30s}  {mf:5.1f}±{sf:4.1f}  {mr:5.1f}     {mt:5.1f}")
    print()
