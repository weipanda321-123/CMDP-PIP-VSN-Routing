#!/usr/bin/env python3
"""
PPO Fine-tuning on Dynamic Environment
- Load pretrained ppo_policy.pt (static graph)
- Train 50 epochs on DynamicRoutingEnv
- Save to ppo_policy_dynamic.pt
"""
import os, sys, time, numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_comparison import build_dynamic_graph
from ppo_train import DynamicRoutingEnv, PPOTrainer, TransformerActor, DualCritic, parse_args

def main():
    args = parse_args()
    # Override key params for fine-tuning
    args.epochs = 50
    args.episodes_per_epoch = 64
    args.lr_actor = 5e-5       # lower LR for fine-tuning
    args.lr_critic = 1e-4
    args.num_nodes = 30        # mid-range density
    args.output = './ppo_policy_dynamic.pt'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("  PPO Fine-tuning on Dynamic Environment")
    print("=" * 60)
    
    # Build dynamic graph
    adj, tc, dfn, positions, p_break, social = build_dynamic_graph(
        args.data, args.num_nodes, args.seed)
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  Graph: N={args.num_nodes}, edges={n_edges}, p_break={p_break:.4f}")
    print(f"  Link lifetime ~ {0.1/p_break:.2f}s")
    
    # Create dynamic env
    env = DynamicRoutingEnv(args.num_nodes, adj, tc, dfn, args.trust_budget,
                            args.max_hops, positions, 300.0, p_break)
    
    # Create trainer with env
    tr = PPOTrainer(env, None, None, args)
    
    # Load pretrained weights
    old_path = './ppo_policy.pt'
    if os.path.exists(old_path):
        ck = torch.load(old_path, map_location='cpu', weights_only=False)
        tr.actor.load_state_dict(ck['actor_state'])
        tr.critic.load_state_dict(ck['critic_state'])
        tr.lam = ck.get('lam', 1.0)
        print(f"  Loaded pretrained weights from {old_path}")
        print(f"  Initial lambda={tr.lam:.4f}")
    else:
        print(f"  WARNING: {old_path} not found, training from scratch")
    
    # Evaluate before fine-tuning
    print(f"\n  --- Before fine-tuning (static weights on dynamic env) ---")
    R0 = tr.collect(32)
    if len(R0['sf']) > 0:
        print(f"  Pre-finetune: {len(R0['sf'])} transitions collected")
    
    # Fine-tune
    print(f"\n  --- Fine-tuning: {args.epochs} epochs, {args.episodes_per_epoch} episodes/epoch ---")
    print(f"  LR: actor={args.lr_actor} critic={args.lr_critic}")
    print()
    
    best = float('inf')
    hist = []
    
    for ep in range(args.epochs):
        tr.schedule(ep, args.epochs)
        args.epsilon = max(0.01, getattr(args, 'epsilon', 0.1) * getattr(args, 'epsilon_decay', 0.995))
        
        R = tr.collect(args.episodes_per_epoch)
        if len(R['sf']) < 4:
            continue
        
        m = tr.update(R)
        hist.append(m)
        
        avg_delay = m.get('avg_delay', 999)
        if avg_delay < best and ep > 5:
            best = avg_delay
            torch.save({
                'actor_state': tr.actor.state_dict(),
                'critic_state': tr.critic.state_dict(),
                'lam': tr.lam,
                'config': {
                    'state_dim': 8,
                    'hidden_dim': args.hidden_dim,
                    'n_heads': args.n_heads,
                    'n_transformer_layers': args.n_transformer_layers,
                    'max_neighbors': args.max_neighbors,
                },
                'epoch': ep,
                'history': hist,
                'dynamic_params': {
                    'p_break': p_break,
                    'p_create': p_break * 0.3,
                    'comm_range': 300.0,
                    'area_km2': 4.0,
                    'calibration': 'Veins 750-car 1500s',
                },
            }, args.output)
        
        if ep % 5 == 0:
            print(f"  Ep {ep:3d}: delay={avg_delay:.6f} trust={m.get('avg_trust',0):.4f} "
                  f"viol={m.get('trust_viol',0):.4f} lam={m.get('lam',0):.4f} "
                  f"ent={m.get('entropy',0):.3f} {'*' if avg_delay <= best else ''}")
    
    print(f"\n  Done! Best delay={best:.6f}")
    print(f"  Saved: {args.output}")
    
    # Also backup original
    if os.path.exists(old_path) and not os.path.exists(old_path.replace('.pt', '_static_backup.pt')):
        import shutil
        shutil.copy2(old_path, old_path.replace('.pt', '_static_backup.pt'))
        print(f"  Original backed up to ppo_policy_static_backup.pt")

if __name__ == '__main__':
    main()
