#!/usr/bin/env python3
"""
PPO Curriculum Training — 用 PPOTrainer 原生接口
"""
import os, sys, time, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_comparison import build_dynamic_graph
from ppo_train import DynamicRoutingEnv, PPOTrainer, TransformerActor, DualCritic, parse_args

def main():
    args = parse_args()
    args.epochs = 220
    args.episodes_per_epoch = 128
    args.lr_actor = 3e-4
    args.lr_critic = 5e-4
    output_path = '../simulations/manhattan/ppo_policy_comm150.pt'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    curriculum = [
        (10, 40), (20, 40), (30, 40), (50, 40), (75, 30), (100, 30)
    ]
    total_epochs = sum(e for _, e in curriculum)

    print("=" * 60)
    print("  PPO Curriculum Training")
    print(f"  Schedule: {curriculum}")
    print(f"  Total epochs: {total_epochs}")
    print("=" * 60)

    # 初始化: 用第一个N建环境和trainer
    N0, _ = curriculum[0]
    adj, tc, dfn, pos, pb, social = build_dynamic_graph(args.data, N0, args.seed)
    env = DynamicRoutingEnv(N0, adj, tc, dfn, args.trust_budget,
                            args.max_hops, pos, 300.0, pb)
    tr = PPOTrainer(env, None, None, args)

    # 加载旧权重
    old_path = '../simulations/manhattan/ppo_policy_dynamic.pt'
    if os.path.exists(old_path):
        try:
            ck = torch.load(old_path, map_location='cpu', weights_only=False)
            tr.actor.load_state_dict(ck['actor_state'])
            tr.critic.load_state_dict(ck['critic_state'])
            print(f"  Loaded pretrained weights from {old_path}")
        except Exception as e:
            print(f"  Could not load: {e}")

    global_epoch = 0
    best_score = float('inf')
    seeds_pool = [42, 123, 456, 789, 1024]

    for N, n_epochs in curriculum:
        print(f"\n{'='*50}")
        print(f"  Phase: N={N}, {n_epochs} epochs")
        print(f"{'='*50}")

        for ep in range(n_epochs):
            seed = seeds_pool[ep % len(seeds_pool)]
            adj, tc, dfn, pos, pb, social = build_dynamic_graph(args.data, N, seed)
            env = DynamicRoutingEnv(N, adj, tc, dfn, args.trust_budget,
                                    args.max_hops, pos, 300.0, pb)
            tr.env = env
            tr.schedule(global_epoch, total_epochs)

            # 用原生 collect + update
            R = tr.collect(n_ep=args.episodes_per_epoch)
            info = tr.update(R)
            global_epoch += 1

            # 统计
            delays = R.get('dc', [])
            trusts = R.get('tc', [])
            avg_d = np.mean(delays) if delays else 0
            avg_t = np.mean(trusts) if trusts else 0

            if global_epoch % 10 == 0 or ep == n_epochs - 1:
                print(f"  [{global_epoch:>3d}/{total_epochs}] N={N} "
                      f"delay={avg_d:.4f} trust={avg_t:.3f} "
                      f"lam={float(tr.lam):.3f} "
                      f"info={info}")

            # 保存最优
            score = avg_d + 10 * max(0, avg_t - args.trust_budget)
            if score < best_score:
                best_score = score
                torch.save({
                    'actor_state': tr.actor.state_dict(),
                    'critic_state': tr.critic.state_dict(),
                    'lam': tr.lam,
                    'config': {
                        'state_dim': 8, 'hidden_dim': args.hidden_dim,
                        'n_heads': args.n_heads,
                        'n_transformer_layers': args.n_transformer_layers,
                        'max_neighbors': args.max_neighbors
                    },
                    'epoch': global_epoch,
                    'curriculum': curriculum
                }, output_path)

    # 最终保存
    torch.save({
        'actor_state': tr.actor.state_dict(),
        'critic_state': tr.critic.state_dict(),
        'lam': tr.lam,
        'config': {
            'state_dim': 8, 'hidden_dim': args.hidden_dim,
            'n_heads': args.n_heads,
            'n_transformer_layers': args.n_transformer_layers,
            'max_neighbors': args.max_neighbors
        },
        'epoch': global_epoch,
        'curriculum': curriculum
    }, output_path)

    print(f"\n{'='*60}")
    print(f"  Done: {global_epoch} epochs, saved to {output_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
