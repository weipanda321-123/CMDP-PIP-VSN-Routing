# CMDP-PIP: Constrained Routing in V2V Social Vehicle Networks

Code and data for the paper:
**"Learn to Solve Constrained Routing Problems in V2V Social Vehicle Networks"**
(IEEE Open Journal of the Communications Society)

## Repository Structure
```
├── src/                    # Veins/OMNeT++ source code
│   ├── CmdpRouter.h/cc    # PPO+PIP routing agent
│   ├── PIPMasker.h/cc     # Proactive Infeasibility Pruning
│   ├── GnnProxy.h/cc      # GNN delay surrogate client
│   └── ...
├── models/                 # Trained checkpoints
│   ├── ppo_policy_*.pt    # PPO policy network
│   └── gnn_delay_model_*.pt  # MPNN delay surrogate
├── logs/                   # Raw experiment CSV logs
│   ├── cmdp_log_PPO-PIP_s0/1/2.csv   # 3-seed PPO+PIP
│   ├── cmdp_log_LS_s0/1/2.csv        # 3-seed Label-Setting
│   ├── cmdp_log_Strict_s0/1/2.csv    # Strict-PIP mode
│   ├── cmdp_log_PostCheck_s0/1/2.csv  # PostCheck baseline
│   ├── cmdp_log_PPOLag_s0/1/2.csv    # PPO-Lag baseline
│   ├── cmdp_log_LearnedMask_s0/1/2.csv # LearnedMask baseline
│   ├── cmdp_log_200s_PPO-PIP_s0.csv  # 200s density experiment
│   └── E3_results/                     # Budget sensitivity
├── scripts/                # Figure generation scripts
│   ├── gen_veins_figs.py
│   ├── gen_density_fig.py
│   └── gen_budget_fig.py
└── simulations/            # OMNeT++ configuration
    └── manhattan/omnetpp.ini
```

## Requirements

- Veins 5.2 (OMNeT++ 5.6.2 + SUMO 1.8.0)
- Python 3.8+ with PyTorch 1.12+
- Ubuntu 20.04/24.04

## Quick Start
```bash
# 1. Start SUMO launcher
python3 veins_launchd -vv -c sumo &

# 2. Run PPO+PIP (default 50s)
opp_run -u Cmdenv -c Ch6-OV-G4-Lag-Full -r 0 \
  -n "../../src:." -l ../../libcmdp_release.so \
  --sim-time-limit=50s --seed-set=0 omnetpp.ini

# 3. Analyze results
python3 scripts/gen_veins_figs.py
```

## Key Results (3-seed mean±std)

| Method | Feas. | Reach | Compute |
|--------|-------|-------|---------|
| PPO+PIP (Ours) | 30.8±0.2% | 87.4±0.5% | 4.33±0.11ms |
| PPO-Lag | 2.3±0.5% | 5.7±0.7% | 0.49±0.01ms |
| PostCheck | 2.3±0.5% | 5.7±0.6% | 0.25±0.00ms |
| PPO+LM | 0.9±0.3% | 2.7±0.4% | 0.56ms |
| LS (oracle) | 30.3±0.8% | 30.3±0.8% | 3.45±0.12ms |

## Citation
```bibtex
@article{He2026OJCOMS_CMDP_PIP,
  author  = {He, Wei and Han, Tao and Ye, Junliang and Ge, Xiaohu},
  title   = {Learn to Solve Constrained Routing Problems in {V2V} Social Vehicle Networks},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2026}
}
```

## License

MIT License
