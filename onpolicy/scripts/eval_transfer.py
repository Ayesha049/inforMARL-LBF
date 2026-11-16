#!/usr/bin/env python
"""Evaluate an LBF-trained InforMARL actor at other team sizes (train N, test M), zero-shot.

Rebuilds the policy from the run's config.json, loads actor.pt, and runs ``--eval_episodes``
episodes per test size in parallel (one env per episode, each env's first episode is used),
with deterministic actions unless ``--stochastic``. The world scales with M under the same
constant-density rule as training.

usage: python onpolicy/scripts/eval_transfer.py --model_dir <run_dir>/models \
           --test_num_agents 2 4 6 8 12 16 --eval_episodes 100 --out_json out.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.scripts.train_lbf import make_env, parse  # noqa: E402


def _t2n(x):
    return x.detach().cpu().numpy()


@torch.no_grad()
def evaluate(args, actor_state, m, episodes, seed, deterministic=True):
    from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy

    env = make_env(args, n_agents=m, num_envs=episodes, seed=seed)
    policy = GR_MAPPOPolicy(args, env.observation_space[0], env.share_observation_space[0],
                            env.node_observation_space[0], env.edge_observation_space[0],
                            env.action_space[0], device=torch.device("cpu"))
    policy.actor.load_state_dict(actor_state)
    policy.actor.eval()

    obs, agent_id, node_obs, adj = env.reset()
    rnn = np.zeros((episodes, m, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((episodes, m, 1), dtype=np.float32)
    first = [None] * episodes
    for _ in range(args.max_steps):
        act, rnn_out = policy.act(obs.reshape(-1, obs.shape[-1]), node_obs.reshape(-1, *node_obs.shape[2:]),
                                  adj.reshape(-1, *adj.shape[2:]), agent_id.reshape(-1, 1),
                                  rnn.reshape(-1, *rnn.shape[2:]), masks.reshape(-1, 1),
                                  deterministic=deterministic)
        actions = _t2n(act).reshape(episodes, m)
        rnn = _t2n(rnn_out).reshape(rnn.shape)
        obs, agent_id, node_obs, adj, _, dones, infos = env.step(actions)
        rnn[dones] = 0.0
        masks = np.where(dones[..., None], 0.0, 1.0).astype(np.float32)
        for e, info in enumerate(infos):
            if info and first[e] is None:
                first[e] = info
        if all(f is not None for f in first):
            break
    assert all(f is not None for f in first), "every env ends within max_steps"
    keys = ("team_return", "food_frac", "success", "length")
    per_ep = {k: [float(f[k]) for f in first] for k in keys}
    summ = {"m": m, "grid": env.G, "n_food": env.F, "episodes": episodes}
    for k in keys:
        v = np.array(per_ep[k])
        summ[k] = float(v.mean())
        summ[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    summ["length_norm"] = summ["length"] / args.max_steps
    summ["food_per_agent"] = summ["food_frac"] * env.F / m
    return summ, per_ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="<run_dir>/models")
    p.add_argument("--test_num_agents", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--actor", default="actor.pt",
                   help="actor file relative to --model_dir, e.g. snapshots/actor_5001600.pt")
    p.add_argument("--out_json", required=True)
    cli = p.parse_args()

    run_dir = Path(cli.model_dir).resolve().parent
    cfg = json.load(open(run_dir / "config.json"))
    args, _ = parse([])
    for k, v in cfg.items():  # the exact training args
        setattr(args, k, v)
    args.use_render = True
    actor = torch.load(Path(cli.model_dir) / cli.actor, map_location="cpu")

    out = {"model_dir": str(cli.model_dir), "actor": cli.actor, "train_num_agents": cfg["num_agents"], "kappa": cfg["kappa"],
           "seed": cfg["seed"], "deterministic": not cli.stochastic, "summary": [], "episodes": {}}
    for m in cli.test_num_agents:
        if m < args.kappa:
            continue
        s, per_ep = evaluate(args, actor, m, cli.eval_episodes, cli.seed + m, deterministic=not cli.stochastic)
        out["summary"].append(s)
        out["episodes"][str(m)] = per_ep
        print(f"m={m:2d} grid={s['grid']:2d} food={s['n_food']:2d}  return {s['team_return']:.3f}  "
              f"food {s['food_frac']:.3f}±{s['food_frac_se']:.3f}  success {s['success']:.2f}  "
              f"length {s['length_norm']:.2f}", flush=True)
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cli.out_json, "w"), indent=1)
    print("saved", cli.out_json)


if __name__ == "__main__":
    main()
