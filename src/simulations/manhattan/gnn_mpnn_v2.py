#!/usr/bin/env python3
"""
MPNN v2: Fix mode collapse by injecting edge features into readout.
Key change: readout = MLP(h_src || h_dst || raw_edge_features)
"""
import os, sys, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_hidden, dropout=0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2 + edge_hidden, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, edge_index, e):
        src, dst = edge_index
        msg = self.msg_mlp(torch.cat([h[src], h[dst], e], -1))
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        h_new = self.drop(self.upd_mlp(torch.cat([h, agg], -1)))
        return self.norm(h + h_new)

class MPNNv2(nn.Module):
    """MPNN with edge-feature-aware readout to prevent mode collapse."""
    def __init__(self, node_dim=4, edge_dim=3, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU())
        self.edge_enc = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList([MPNNLayer(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)])
        # Readout: h_src || h_dst || raw_edge_feat → delay
        readout_in = hidden_dim * 2 + edge_dim  # KEY FIX: include raw edge features
        self.mean_head = nn.Sequential(
            nn.Linear(readout_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        self.var_head = nn.Sequential(
            nn.Linear(readout_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, node_feat, edge_index, edge_feat, batch_idx=None):
        h = self.node_enc(node_feat)
        e = self.edge_enc(edge_feat)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        src, dst = edge_index
        # KEY: concat node embeddings WITH raw edge features
        readout_input = torch.cat([h[src], h[dst], edge_feat], dim=-1)
        mean = self.mean_head(readout_input).squeeze(-1)
        log_var = self.var_head(readout_input).squeeze(-1)
        return mean, log_var

# Keep backward-compatible wrapper
class EdgeFeatureNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.backbone(x)
        return self.mean_head(h).squeeze(-1), self.var_head(h).squeeze(-1)

def nll_loss(mean, log_var, target):
    precision = torch.exp(-log_var)
    return 0.5 * (precision * (target - mean)**2 + log_var).mean()

def load_and_build(csv_path):
    records = []
    with open(csv_path) as f:
        f.readline()
        for l in f:
            p = l.strip().split(',')
            if len(p) >= 6:
                try:
                    r = dict(simTime=float(p[0]), src=int(p[1]), dst=int(p[2]),
                             distance=float(p[3]), snir=float(p[4]), oracleDelay=float(p[5]))
                    if r['snir'] > 0.01 and 0 < r['oracleDelay'] < 0.1:
                        records.append(r)
                except: pass
    print(f"Loaded {len(records)} records")
    nodes = sorted(set(r['src'] for r in records) | set(r['dst'] for r in records))
    n2i = {n:i for i,n in enumerate(nodes)}; N = len(nodes)
    np.random.seed(42)
    nf = np.zeros((N,4), dtype=np.float32)
    deg = {}
    for r in records: deg[r['src']] = deg.get(r['src'],0)+1
    for i,n in enumerate(nodes):
        nf[i] = [np.random.uniform(0,1), np.random.uniform(0,1), deg.get(n,0)/max(len(records),1), i/N]
    dists = np.array([r['distance'] for r in records])
    snirs = np.array([r['snir'] for r in records])
    dm, ds = dists.mean(), max(dists.std(), 1e-6)
    sm, ss = snirs.mean(), max(snirs.std(), 1e-6)
    dl, dh = np.percentile(dists, [0.5,99.5])
    sl, sh = np.percentile(snirs, [0.5,99.5])
    ei = [[], []]; ef = []; tgt = []
    for r in records:
        ei[0].append(n2i[r['src']]); ei[1].append(n2i[r['dst']])
        dc = np.clip(r['distance'],dl,dh); sc = np.clip(r['snir'],sl,sh)
        ef.append([(dc-dm)/ds, (sc-sm)/ss, np.log1p(dc)/10])
        tgt.append(r['oracleDelay'])
    ns = dict(dist_mean=float(dm),dist_std=float(ds),snir_mean=float(sm),snir_std=float(ss),
              dist_lo=float(dl),dist_hi=float(dh),snir_lo=float(sl),snir_hi=float(sh))
    return nf, np.array(ei,dtype=np.int64), np.array(ef,dtype=np.float32), np.array(tgt,dtype=np.float32), ns, n2i

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='../simulations/manhattan/gnn_training_data.csv')
    p.add_argument('--model-out', default='../simulations/manhattan/gnn_delay_model.pt')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    nf, ei, ef, tgt, ns, nm = load_and_build(args.data)
    N_n, N_e = nf.shape[0], ei.shape[1]
    print(f"Graph: {N_n} nodes, {N_e} edges")

    n_val = max(1, int(N_e*0.1)); idx = np.random.permutation(N_e)
    vi, ti = idx[:n_val], idx[n_val:]

    nf_t = torch.FloatTensor(nf); ei_t = torch.LongTensor(ei)
    ef_t = torch.FloatTensor(ef); tgt_t = torch.FloatTensor(tgt)

    model = MPNNv2(node_dim=4, edge_dim=3, hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers, dropout=args.dropout)
    print(f"MPNNv2 params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)

    best_vl = float('inf'); best_st = None
    for ep in range(args.epochs):
        model.train()
        pm = np.random.permutation(len(ti)); el = 0; nb = 0
        for i in range(0, len(ti), args.bs):
            bix = ti[pm[i:i+args.bs]]
            # Full graph forward
            mean, lv = model(nf_t, ei_t, ef_t)
            loss = nll_loss(mean[bix], lv[bix], tgt_t[bix])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); el += loss.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vm, vlv = model(nf_t, ei_t, ef_t)
            vl = nll_loss(vm[vi], vlv[vi], tgt_t[vi]).item()
            vmse = ((vm[vi]-tgt_t[vi])**2).mean().item()
        if vl < best_vl: best_vl = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}
        if ep % 10 == 0:
            print(f"  Ep {ep:3d}: tNLL={el/max(nb,1):.4f} vNLL={vl:.4f} vMSE={vmse:.8f}")

    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pm, plv = model(nf_t, ei_t, ef_t)
        pn = pm.numpy(); y = tgt
    mse=np.mean((pn-y)**2); mae=np.mean(np.abs(pn-y))
    r2 = 1-np.sum((y-pn)**2)/np.sum((y-y.mean())**2)
    corr = np.corrcoef(y, pn)[0,1]
    print(f"\n{'='*50}")
    print(f"  MPNNv2 Results:")
    print(f"    MSE={mse:.8f}  MAE={mae:.6f}  R²={r2:.4f}")
    print(f"    Pearson r={corr:.4f}")
    print(f"    pred std={pn.std():.6f}  target std={y.std():.6f}")
    print(f"{'='*50}")

    # Save (compatible with inference server)
    mlp = EdgeFeatureNet(6, args.hidden_dim, args.num_layers)
    X_m = np.column_stack([ef[:,0],ef[:,1],ef[:,2],ef[:,2],ei[0]/max(N_n,1),ei[1]/max(N_n,1)])
    X_t = torch.FloatTensor(X_m)
    mo = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    mlp.train()
    for _ in range(50):
        pm2 = np.random.permutation(N_e)
        for j in range(0,N_e,256):
            m2,lv2=mlp(X_t[pm2[j:j+256]]); l2=nll_loss(m2,lv2,tgt_t[pm2[j:j+256]])
            mo.zero_grad(); l2.backward(); mo.step()

    torch.save({
        'mpnn_state': model.state_dict(),
        'mpnn_config': dict(node_dim=4,edge_dim=3,hidden_dim=args.hidden_dim,
                             num_layers=args.num_layers,dropout=args.dropout),
        'model_state': mlp.state_dict(),
        'config': dict(input_dim=6,hidden_dim=args.hidden_dim,num_layers=args.num_layers),
        'norm_stats': ns, 'node_map': nm,
        'metrics': dict(mse=float(mse),mae=float(mae),r2=float(r2),corr=float(corr)),
    }, args.model_out)
    print(f"  Saved to {args.model_out}")

if __name__=='__main__': main()
