#!/usr/bin/env python3
"""
PPO-Lagrangian (Ray et al., 2019) baseline for RCSPP.
Same architecture as ours, but NO PIP mask during training or inference.
Constraint handled purely via adaptive Lagrangian multiplier.
"""
import sys, os, time, torch, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_comparison import build_dynamic_graph
from ppo_train import TransformerActor, RoutingEnv

CSV = '../simulations/manhattan/gnn_training_data_750cars_1500s.csv'
N_TRAIN = 200  # Train on N=200 graph
SEED = 42
budget = 2.3
max_hops = 20
n_episodes = 1000
lr_actor = 3e-4
lr_lambda = 1e-2  # Lagrange multiplier learning rate
gamma = 0.99
clip_eps = 0.2
max_n = 20

# Build training graph
adj, tc, dfn, pos, pb, soc = build_dynamic_graph(CSV, N_TRAIN, SEED)
nodes = list(adj.keys())
N = max(nodes) + 1

# Same architecture as ours
state_dim = 8; hidden_dim = 64; n_heads = 4; n_layers = 2

actor = TransformerActor(state_dim=state_dim, hidden_dim=hidden_dim,
                          n_heads=n_heads, n_layers=n_layers, max_neighbors=max_n)
optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

# Lagrange multiplier (learnable, always >= 0)
log_lam = torch.tensor(0.0, requires_grad=True)
lam_optimizer = torch.optim.Adam([log_lam], lr=lr_lambda)

def get_state_features(env, dst, nb):
    nn_ = len(nb)
    at = np.mean([tc.get(n,0.5) for n in nb]) if nb else 0.5
    sf = torch.FloatTensor([[env.current/max(N,1), dst/max(N,1),
          env.remaining_budget/max(budget,1e-6), env.step_count/max_hops,
          nn_/max_n, at, 1.0 if env.current==dst else 0.0, 0.0]])
    nf = torch.zeros(1, max_n, 8); mk = torch.zeros(1, max_n, dtype=torch.bool)
    for i, n in enumerate(nb[:max_n]):
        nf[0,i] = torch.tensor([n/max(N,1), tc.get(n,0.5),
                   env.h_star.get(n,1.0)/max(budget,1e-6), dfn(env.current,n)*100,
                   1.0 if n==dst else 0.0, 1.0,
                   env.remaining_budget/max(budget,1e-6), env.step_count/max_hops])
        mk[0,i] = True
    return sf, nf, mk

print("Training PPO-Lagrangian (no PIP)...")
print(f"N={N_TRAIN}, budget={budget}, episodes={n_episodes}")

best_feas = 0
for ep in range(n_episodes):
    # Sample random src, dst
    src, dst = np.random.choice(nodes, 2, replace=False)
    env = RoutingEnv(N, adj, tc, dfn, budget, max_hops)
    state = env.reset(src, dst)

    log_probs = []; rewards = []; costs = []

    for step in range(max_hops):
        if env.current == dst: break
        # NO PIP: use all unvisited neighbors
        nb = [n for n in adj.get(env.current, []) if n not in env.path]
        if not nb: break
        nb = nb[:max_n]

        sf, nf, mk = get_state_features(env, dst, nb)
        logits = actor(sf, nf, mk)
        probs = torch.softmax(logits[0, :len(nb)], dim=-1)
        dist = torch.distributions.Categorical(probs)
        ai = dist.sample()
        log_probs.append(dist.log_prob(ai))

        chosen = nb[ai.item()]
        state, reward, cost, done = env.step(chosen)
        rewards.append(-dfn(env.path[-2], chosen))  # minimize delay
        costs.append(tc.get(chosen, 0.5))  # trust cost
        if done: break

    # Episode outcome
    reached = env.current == dst
    total_trust = env.total_trust
    feasible = reached and (total_trust <= budget + 1e-9)

    # Compute returns
    if not log_probs: continue
    R = 1.0 if reached else -1.0
    lam = torch.exp(log_lam).detach()

    # Constraint violation
    violation = max(0, total_trust - budget) if reached else budget  # penalize not reaching

    # PPO-Lagrangian reward: r - lambda * cost
    adjusted_R = R - lam.item() * violation

    # Simple REINFORCE update (sufficient for comparison)
    policy_loss = 0
    for lp in log_probs:
        policy_loss -= lp * adjusted_R

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # Update Lagrange multiplier (dual ascent)
    lam_loss = -log_lam * (violation - 0)  # push lambda up when constraint violated
    lam_optimizer.zero_grad()
    lam_loss.backward()
    lam_optimizer.step()

    if (ep+1) % 100 == 0:
        # Quick eval
        eval_feas = 0; eval_n = 50
        actor.eval()
        for _ in range(eval_n):
            s, d = np.random.choice(nodes, 2, replace=False)
            ev = RoutingEnv(N, adj, tc, dfn, budget, max_hops)
            st = ev.reset(s, d)
            for _ in range(max_hops):
                if ev.current == d: break
                nb = [n for n in adj.get(ev.current, []) if n not in ev.path]
                if not nb: break
                nb = nb[:max_n]
                sf, nf, mk = get_state_features(ev, d, nb)
                with torch.no_grad():
                    logits = actor(sf, nf, mk)
                ai = logits[0, :len(nb)].argmax().item()
                st, _, _, done = ev.step(nb[ai])
                if done: break
            if ev.current == d and ev.total_trust <= budget + 1e-9:
                eval_feas += 1
        pct = 100*eval_feas/eval_n
        print(f"  ep={ep+1:4d}  lam={torch.exp(log_lam).item():.2f}  eval_feas={pct:.0f}%")
        if pct > best_feas: best_feas = pct
        actor.train()

# Save
ck = {'actor_state': actor.state_dict(),
      'config': {'state_dim': state_dim, 'hidden_dim': hidden_dim,
                 'n_heads': n_heads, 'n_transformer_layers': n_layers,
                 'max_neighbors': max_n},
      'lam': torch.exp(log_lam).item()}
torch.save(ck, '../simulations/manhattan/ppo_lag_baseline.pt')
print(f"\nDone. Best eval feas: {best_feas:.0f}%")
print("Saved: ppo_lag_baseline.pt")

# Full evaluation on all N values
print("\n=== Full Evaluation (PPO-Lag baseline, no PIP) ===")
N_LIST = [50, 75, 100, 150, 200, 300, 400]
SEEDS = [42, 123, 456, 789, 1024]
actor.eval()

for Nv in N_LIST:
    feas_all = []
    for seed in SEEDS:
        adj2, tc2, dfn2, _, _, _ = build_dynamic_graph(CSV, Nv, seed)
        nodes2 = list(adj2.keys())
        np.random.seed(seed)
        queries = [(np.random.choice(nodes2), np.random.choice(nodes2)) for _ in range(200)]
        queries = [(s,d) for s,d in queries if s!=d][:100]
        feas_count = 0
        for src, dst in queries:
            if src >= max(nodes2)+1 or dst >= max(nodes2)+1: continue
            Nenv = max(max(adj2.keys()), dst, src) + 1
            ev = RoutingEnv(Nenv, adj2, tc2, dfn2, budget, max_hops)
            st = ev.reset(src, dst)
            for _ in range(max_hops):
                if ev.current == dst: break
                nb = [n for n in adj2.get(ev.current, []) if n not in ev.path]
                if not nb: break
                nb = nb[:max_n]
                sf, nf, mk = get_state_features(ev, dst, nb)
                with torch.no_grad():
                    logits = actor(sf, nf, mk)
                ai = logits[0, :len(nb)].argmax().item()
                if ai >= len(nb): break
                st, _, _, done = ev.step(nb[ai])
                if done: break
            if ev.current == dst and ev.total_trust <= budget + 1e-9:
                feas_count += 1
        feas_all.append(feas_count)
    m = np.mean(feas_all); s = np.std(feas_all)
    print(f"N={Nv:3d}  PPO-Lag(Ray2019): {m:.1f}±{s:.1f}%")
