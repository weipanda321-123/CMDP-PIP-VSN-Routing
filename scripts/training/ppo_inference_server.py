#!/usr/bin/env python3
"""
PPO-Lagrangian Policy Inference Server (persistent TCP connection)
Protocol: newline-delimited JSON over persistent TCP
"""

import json
import socket
import sys
import os
import argparse
import numpy as np
import threading

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppo_train import TransformerActor


class PPOInferenceServer:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']
        self.lam = checkpoint.get('lam', 0.0)

        self.actor = TransformerActor(
            state_dim=self.config['state_dim'],
            hidden_dim=self.config['hidden_dim'],
            n_heads=self.config['n_heads'],
            n_layers=self.config['n_transformer_layers'],
            max_neighbors=self.config['max_neighbors'],
        )
        self.actor.load_state_dict(checkpoint['actor_state'])
        self.actor.eval()
        self.max_n = self.config['max_neighbors']

        print(f"  PPO Actor loaded: hidden={self.config['hidden_dim']}, "
              f"heads={self.config['n_heads']}, layers={self.config['n_transformer_layers']}")
        print(f"  Lambda={self.lam:.4f}, max_neighbors={self.max_n}")

    def predict_next_hop(self, query):
        current = query['current']
        dst = query['dst']
        budget = query.get('budget', 2.3026)
        remaining = query.get('remaining_budget', budget)
        step = query.get('step', 0)
        neighbors = query.get('neighbors', [])
        trust_costs = query.get('trust_costs', [])
        delays = query.get('delays', [])
        h_star_vals = query.get('h_star', [])

        if not neighbors:
            return {'next_hop': -1, 'action_idx': -1, 'confidence': 0.0,
                    'feasible_count': 0, 'error': 'no neighbors'}

        feasible = []
        for i, n in enumerate(neighbors):
            tc = trust_costs[i] if i < len(trust_costs) else 0.5
            hs = h_star_vals[i] if i < len(h_star_vals) else 0.0
            if tc + hs <= remaining + 1e-9:
                feasible.append(i)

        if not feasible:
            best_i = 0
            best_cost = float('inf')
            for i, n in enumerate(neighbors):
                tc = trust_costs[i] if i < len(trust_costs) else 0.5
                hs = h_star_vals[i] if i < len(h_star_vals) else 0.0
                if tc + hs < best_cost:
                    best_cost = tc + hs
                    best_i = i
            feasible = [best_i]

        N_approx = max(max(neighbors) + 1, dst + 1, current + 1, 50)
        nb = neighbors[:self.max_n]
        nn_ = len(nb)
        avg_trust = np.mean(trust_costs[:nn_]) if trust_costs else 0.5

        state_feat = torch.FloatTensor([[
            current / N_approx, dst / N_approx,
            remaining / max(budget, 1e-6), step / 20.0,
            nn_ / self.max_n, avg_trust,
            1.0 if current == dst else 0.0, self.lam / 10.0,
        ]])

        neigh_feat = torch.zeros(1, self.max_n, 8)
        mask = torch.zeros(1, self.max_n, dtype=torch.bool)
        for i, n in enumerate(nb):
            tc = trust_costs[i] if i < len(trust_costs) else 0.5
            hs = h_star_vals[i] if i < len(h_star_vals) else 0.0
            dl = delays[i] if i < len(delays) else 0.003
            neigh_feat[0, i] = torch.tensor([
                n / N_approx, tc, hs / max(budget, 1e-6), dl * 100,
                1.0 if n == dst else 0.0, 1.0 if i in feasible else 0.0,
                remaining / max(budget, 1e-6), step / 20.0,
            ])
            mask[0, i] = True

        with torch.no_grad():
            logits = self.actor(state_feat, neigh_feat, mask)

        masked_logits = logits[0].clone()
        for i in range(min(nn_, self.max_n)):
            if i not in feasible:
                masked_logits[i] = -1e9
        for i in range(nn_, self.max_n):
            masked_logits[i] = -1e9

        probs = F.softmax(masked_logits[:nn_], dim=-1)
        action_idx = int(probs.argmax().item())
        confidence = float(probs[action_idx].item())
        next_hop = nb[action_idx]

        return {
            'next_hop': next_hop, 'action_idx': action_idx,
            'confidence': confidence, 'feasible_count': len(feasible),
        }


def handle_client(conn, addr, server, query_counter):
    """Handle persistent connection from one C++ client."""
    print(f"  [PPO] Persistent connection from {addr}")
    buf = ""
    try:
        while True:
            data = conn.recv(8192)
            if not data:
                break
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    result = server.predict_next_hop(req)
                    resp = json.dumps(result) + "\n"
                    conn.sendall(resp.encode())
                    query_counter[0] += 1
                    if query_counter[0] % 100 == 0:
                        print(f"  [PPO] Queries served: {query_counter[0]}")
                except json.JSONDecodeError as e:
                    err = json.dumps({'error': f'JSON parse: {e}', 'next_hop': -1}) + "\n"
                    conn.sendall(err.encode())
                except Exception as e:
                    err = json.dumps({'error': str(e), 'next_hop': -1}) + "\n"
                    conn.sendall(err.encode())
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        conn.close()
        print(f"  [PPO] Connection from {addr} closed (total: {query_counter[0]})")


def run_server(args):
    server = PPOInferenceServer(args.model)
    print(f"\nPPO Inference Server listening on {args.host}:{args.port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(5)

    query_counter = [0]
    try:
        while True:
            conn, addr = sock.accept()
            t = threading.Thread(target=handle_client,
                                 args=(conn, addr, server, query_counter),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print(f"\nShutting down. Total queries: {query_counter[0]}")
    finally:
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO-Lagrangian inference server')
    parser.add_argument('--model', default='../simulations/manhattan/ppo_policy.pt')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5556)
    args = parser.parse_args()
    run_server(args)
