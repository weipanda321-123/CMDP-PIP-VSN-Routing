#!/usr/bin/env python3
"""
PPO-Penalty Baseline: Standard PPO with reward shaping, NO PIP mask.
This isolates the contribution of PIP by showing what happens without it.

Key differences from proposed method:
  - No PIP mask: all neighbors are valid actions (no feasibility filtering)
  - Reward: r = -delay - beta * max(0, cum_trust - B)
  - No Lagrangian dual update (fixed penalty coefficient beta)
"""
import os, sys, time, copy, heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_train import TransformerActor, DualCritic, parse_args


class NoPIPRoutingEnv:
    """RoutingEnv variant that does NOT apply PIP feasibility masking.
    All neighbors are always available as actions.
    Trust violation is penalized in reward instead."""

    def __init__(self, N, adj, tc, dfn, budget, max_hops, penalty_beta=10.0):
        self.N = N
        self.adj = adj
        self.tc = tc
        self.dfn = dfn
        self.budget = budget
        self.max_hops = max_hops
        self.beta = penalty_beta
        self.reset(0, 1)

    def reset(self, src, dst):
        self.src = src
        self.dst = dst
        self.current = src
        self.total_delay = 0.0
        self.total_trust = 0.0
        self.remaining_budget = self.budget
        self.path = [src]
        self.step_count = 0
        self.visited = {src}
        return self._get_state()

    def _get_state(self):
        nb = [j for j in self.adj.get(self.current, []) if j not in self.visited]
        return {
            'current': self.current,
            'dst': self.dst,
            'remaining_budget': self.remaining_budget,
            'step': self.step_count,
            'neighbors': nb,
        }

    def get_feasible_actions(self):
        """No PIP: all unvisited neighbors are 'feasible'"""
        return [j for j in self.adj.get(self.current, []) if j not in self.visited]

    def step(self, action):
        j = action
        delay = self.dfn(self.current, j)
        trust_cost = self.tc.get(j, 0.5)

        self.total_delay += delay
        self.total_trust += trust_cost
        self.remaining_budget -= trust_cost
        self.current = j
        self.path.append(j)
        self.visited.add(j)
        self.step_count += 1

        violation = max(0.0, self.total_trust - self.budget)
        reward = -delay - self.beta * violation

        done = (
            (j == self.dst)
            or (self.step_count >= self.max_hops)
            or (len(self.get_feasible_actions()) == 0)
        )

        if done and j != self.dst:
            reward -= 100.0

        return self._get_state(), reward, done, {
            'reached': j == self.dst,
            'feasible': self.total_trust <= self.budget + 1e-9,
            'violation': violation,
        }


class PPOPenaltyTrainer:
    """PPO trainer without Lagrangian dual — uses fixed penalty instead."""

    def __init__(self, env, args):
        self.env = env
        self.actor = TransformerActor(
            state_dim=8, hidden_dim=args.hidden_dim,
            n_heads=args.n_heads, n_layers=args.n_transformer_layers,
            max_neighbors=args.max_neighbors,
        )
        self.critic = DualCritic(state_dim=16)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.clip_eps = args.ppo_clip
        self.gae_lam = args.gae_lambda
        self.alpha_ent = args.alpha_ent
        self.max_neighbors = args.max_neighbors
        self.N = env.N
        self.budget = env.budget

    def _encode(self, env, nb):
        """Encode state and neighbor features into tensors."""
        sf = torch.FloatTensor([[
            env.current / max(self.N, 1),
            env.dst / max(self.N, 1),
            env.remaining_budget / max(self.budget, 1e-6),
            env.step_count / 20.0,
            len(nb) / self.max_neighbors,
            np.mean([env.tc.get(n, 0.5) for n in nb]) if nb else 0.5,
            1.0 if env.current == env.dst else 0.0,
            0.0,
        ]])
        nf = torch.zeros(1, self.max_neighbors, 8)
        mk = torch.zeros(1, self.max_neighbors, dtype=torch.bool)
        for i, n in enumerate(nb[:self.max_neighbors]):
            nf[0, i] = torch.tensor([
                n / max(self.N, 1),
                env.tc.get(n, 0.5),
                0.5,                               # no h_star
                env.dfn(env.current, n) * 100.0,
                1.0 if n == env.dst else 0.0,
                1.0,                               # all "feasible" (no PIP)
                env.remaining_budget / max(self.budget, 1e-6),
                env.step_count / 20.0,
            ])
            mk[0, i] = True
        return sf, nf, mk

    def collect(self, n_episodes=64):
        data = {
            'sf': [], 'nf': [], 'mk': [], 'act': [],
            'rew': [], 'done': [], 'logp': [], 'val_c': [], 'val_d': [],
        }
        stats = {
            'delays': [], 'trusts': [], 'reached': [], 'feasible': [], 'violations': [],
        }

        nodes_with_adj = [i for i in range(self.N) if len(self.env.adj.get(i, [])) > 0]
        if len(nodes_with_adj) < 2:
            return data

        for _ in range(n_episodes):
            src, dst = np.random.choice(nodes_with_adj, 2, replace=False)
            state = self.env.reset(src, dst)

            for _ in range(20):
                nb = state['neighbors'][:self.max_neighbors]
                if not nb:
                    break

                sf, nf, mk = self._encode(self.env, nb)

                with torch.no_grad():
                    logits = self.actor(sf, nf, mk)
                    vc, vd = self.critic(torch.cat([sf, sf], dim=-1))

                n_valid = len(nb)
                probs = F.softmax(logits[0, :n_valid], dim=-1)
                dist = torch.distributions.Categorical(probs)
                ai = dist.sample().item()
                logp = dist.log_prob(torch.tensor(ai))

                data['sf'].append(sf)
                data['nf'].append(nf)
                data['mk'].append(mk)
                data['act'].append(ai)
                data['logp'].append(logp.item())
                data['val_c'].append(vc.item())
                data['val_d'].append(vd.item())

                next_state, reward, done, info = self.env.step(nb[ai])
                data['rew'].append(reward)
                data['done'].append(done)

                state = next_state
                if done:
                    break

            stats['delays'].append(self.env.total_delay)
            stats['trusts'].append(self.env.total_trust)
            stats['reached'].append(self.env.current == dst)
            stats['feasible'].append(
                self.env.total_trust <= self.budget + 1e-9
                and self.env.current == dst
            )
            stats['violations'].append(max(0.0, self.env.total_trust - self.budget))

        if not data['sf']:
            return data

        data['sf'] = torch.cat(data['sf'], dim=0)
        data['nf'] = torch.cat(data['nf'], dim=0)
        data['mk'] = torch.cat(data['mk'], dim=0)
        for k in ['act', 'rew', 'done', 'logp', 'val_c', 'val_d']:
            data[k] = torch.tensor(data[k], dtype=torch.float32)
        data['stats'] = stats
        return data

    def update(self, data):
        if not isinstance(data['sf'], torch.Tensor) or data['sf'].shape[0] < 4:
            return {}

        T = len(data['rew'])
        adv = torch.zeros(T)
        ret = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = data['val_c'][t + 1].item() if t + 1 < T and not data['done'][t] else 0.0
            delta = data['rew'][t] + 1.0 * next_val - data['val_c'][t]
            gae = delta + 1.0 * self.gae_lam * (1.0 - data['done'][t].item()) * gae
            adv[t] = gae
            ret[t] = adv[t] + data['val_c'][t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(3):
            logits = self.actor(data['sf'], data['nf'], data['mk'])

            log_probs = []
            entropies = []
            for b in range(logits.shape[0]):
                n_valid = max(int(data['mk'][b].sum().item()), 1)
                p = F.softmax(logits[b, :n_valid], dim=-1)
                d = torch.distributions.Categorical(p)
                ai = min(int(data['act'][b].item()), n_valid - 1)
                log_probs.append(d.log_prob(torch.tensor(ai)))
                entropies.append(d.entropy())

            new_logp = torch.stack(log_probs)
            entropy = torch.stack(entropies).mean()

            ratio = torch.exp(new_logp - data['logp'])
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.alpha_ent * entropy

            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()

            vc, _ = self.critic(torch.cat([data['sf'], data['sf']], dim=-1))
            critic_loss = F.mse_loss(vc.squeeze(), ret)
            self.opt_c.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.opt_c.step()

        s = data['stats']
        return {
            'avg_delay': np.mean(s['delays']) if s['delays'] else 999,
            'avg_trust': np.mean(s['trusts']) if s['trusts'] else 0,
            'reach_rate': np.mean(s['reached']) if s['reached'] else 0,
            'feasibility': np.mean(s['feasible']) if s['feasible'] else 0,
            'avg_violation': np.mean(s['violations']) if s['violations'] else 0,
            'entropy': entropy.item(),
            'actor_loss': actor_loss.item(),
        }


def main():
    args = parse_args()
    args.epochs = 200
    args.episodes_per_epoch = 64
    args.output = '../simulations/manhattan/ppo_penalty_baseline.pt'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  PPO-Penalty Baseline (NO PIP mask)")
    print("  Purpose: Isolate PIP contribution")
    print("=" * 60)

    from baseline_comparison import build_dynamic_graph
    adj, tc, dfn, positions, p_break, _extra = build_dynamic_graph(
        args.data, args.num_nodes, args.seed
    )
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  Graph: N={args.num_nodes}, edges={n_edges}, p_break={p_break:.4f}")
    print(f"  Budget B={args.trust_budget}")
    print(f"  Penalty beta=10.0")
    print()

    env = NoPIPRoutingEnv(
        args.num_nodes, adj, tc, dfn, args.trust_budget, 20, penalty_beta=10.0
    )
    trainer = PPOPenaltyTrainer(env, args)

    best_feas = 0.0
    for ep in range(args.epochs):
        R = trainer.collect(args.episodes_per_epoch)
        if not isinstance(R.get('sf'), torch.Tensor) or R['sf'].shape[0] < 4:
            continue
        m = trainer.update(R)

        feas = m.get('feasibility', 0)
        if feas > best_feas and ep > 20:
            best_feas = feas
            torch.save({
                'actor_state': trainer.actor.state_dict(),
                'critic_state': trainer.critic.state_dict(),
                'config': {
                    'state_dim': 8,
                    'hidden_dim': args.hidden_dim,
                    'n_heads': args.n_heads,
                    'n_transformer_layers': args.n_transformer_layers,
                    'max_neighbors': args.max_neighbors,
                },
                'type': 'ppo_penalty_no_pip',
                'penalty_beta': 10.0,
                'epoch': ep,
                'feasibility': feas,
            }, args.output)

        if ep % 10 == 0:
            print(
                f"  Ep {ep:4d}: delay={m.get('avg_delay',0):.4f} "
                f"feas={m.get('feasibility',0):.3f} "
                f"reach={m.get('reach_rate',0):.3f} "
                f"trust={m.get('avg_trust',0):.4f} "
                f"viol={m.get('avg_violation',0):.4f} "
                f"ent={m.get('entropy',0):.3f} "
                f"{'*BEST' if feas >= best_feas and ep > 20 else ''}"
            )

    print(f"\nDone! Best feasibility={best_feas:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
