#!/usr/bin/env python3
"""PPO Curriculum Training for area=4km², comm=300m"""
import os, sys, time, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_comparison import build_dynamic_graph
from ppo_train import DynamicRoutingEnv, PPOTrainer, parse_args

def main():
    args = parse_args()
    args.episodes_per_epoch = 128
    args.lr_actor = 3e-4
    args.lr_critic = 5e-4
    output_path = '../simulations/manhattan/ppo_policy_area4.pt'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    curriculum = [
        (30, 25),   # 稀疏热身
        (50, 25),
        (75, 25),
        (100, 30),  # 核心范围
        (150, 30),
        (200, 30),  # 目标规模
        (300, 25),  # 扩展
    ]
    total_epochs = sum(e for _, e in curriculum)

    print("=" * 60)
    print("  PPO Curriculum Training (area=4km², comm=300m)")
    print(f"  Schedule: {curriculum}")
    print(f"  Total epochs: {total_epochs}")
    print("=" * 60, flush=True)

    N0, _ = curriculum[0]
    adj, tc, dfn, pos, pb, social = build_dynamic_graph(args.data, N0, args.seed)
    env = DynamicRoutingEnv(N0, adj, tc, dfn, args.trust_budget,
                            args.max_hops, pos, 300.0, pb)
    tr = PPOTrainer(env, None, None, args)

    # 可选: 加载旧权重热启动
    old_path = '../simulations/manhattan/ppo_policy_curriculum.pt'
    if os.path.exists(old_path):
        try:
            ck = torch.load(old_path, map_location='cpu', weights_only=False)
            tr.actor.load_state_dict(ck['actor_state'])
            tr.critic.load_state_dict(ck['critic_state'])
            print(f"  Loaded pretrained weights from {old_path}", flush=True)
        except Exception as e:
            print(f"  Could not load pretrained: {e}", flush=True)

    global_epoch = 0
    best_score = float('inf')
    seeds_pool = [42, 123, 456, 789, 1024]

    for N, n_epochs in curriculum:
        print(f"\n  Phase: N={N}, {n_epochs} epochs", flush=True)
        for ep in range(n_epochs):
            seed = seeds_pool[ep % len(seeds_pool)]
            adj, tc, dfn, pos, pb, social = build_dynamic_graph(args.data, N, seed)
            env = DynamicRoutingEnv(N, adj, tc, dfn, args.trust_budget,
                                    args.max_hops, pos, 300.0, pb)
            tr.env = env
            tr.schedule(global_epoch, total_epochs)
            R = tr.collect(n_ep=args.episodes_per_epoch)
            info = tr.update(R)
            global_epoch += 1

            if global_epoch % 10 == 0 or ep == n_epochs - 1:
                print(f"  [{global_epoch:>3d}/{total_epochs}] N={N} "
                      f"delay={info.get('avg_delay',0):.4f} "
                      f"trust={info.get('avg_trust',0):.3f} "
                      f"viol={info.get('trust_viol',0)} "
                      f"lam={info.get('lam',0):.3f}", flush=True)

            score = info.get('avg_delay', 1) + 10 * max(0, info.get('avg_trust', 3) - args.trust_budget)
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
                    'curriculum': curriculum,
                    'area_km2': 4.0,
                    'comm_range': 300.0
                }, output_path)

    # 最终保存
    final_path = output_path.replace('.pt', '_final.pt')
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
        'curriculum': curriculum,
        'area_km2': 4.0,
        'comm_range': 300.0
    }, final_path)

    print(f"\n{'='*60}")
    print(f"  训练完成: {global_epoch} epochs")
    print(f"  Best model: {output_path}")
    print(f"  Final model: {final_path}")
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
