#!/usr/bin/env python
"""Behaviour diagnostics for an LBF-trained InforMARL actor tested at other team sizes.

For each test size M, runs ``--eval_episodes`` episodes (one per env, deterministic actions) for
``--horizon`` steps (default 50). The policy sees the remaining-time feature of the training horizon
(``max_steps`` from config.json, 25), clipped at 0, so the first 25 steps are identical to
eval_transfer.py and the extra steps only ask "would more time help?".

Per (M) it reports, over the first ``max_steps`` steps unless stated:
* food_curve[t]        mean fraction of food loaded after t steps, t = 1 .. horizon
* success_25 / _50     all food loaded within 25 / horizon steps
* act_*                fraction of agent-steps: LOAD next to food, LOAD with no adjacent food, move, NONE
* solo_load            LOAD steps next to food that did not load it (fewer than kappa loaders)
* excess_loaders       mean (loaders - kappa) per loaded food
* left_*               state of food still present at step 25: adjacent to >= 1 agent,
                       within sight of >= 1 agent, seen by nobody
* left_n               mean number of food items left at step 25
* t_first_load, t_last_load   mean step of the first / last load (episodes with >= 1 / all food loaded)
* no_mate_in_sight     agent-steps with no teammate within sight
* no_entity_in_sight   agent-steps with neither teammate nor alive food within sight
* partners_per_agent   distinct loading partners per agent that loaded at least once (M >= 3)
* participation        fraction of agents that took part in >= 1 load

usage: python onpolicy/scripts/diagnose_lbf.py --model_dir <run_dir>/models --test_num_agents 2 4 8 16 \
           --eval_episodes 200 --out_json out.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.getcwd()))

from lbf.vec_lbf import DELTA, EAST, LOAD, NONE, NORTH, SOUTH, WEST, GraphLBFVecEnv  # noqa: E402
from onpolicy.scripts.train_lbf import parse  # noqa: E402


class DiagEnv(GraphLBFVecEnv):
    """Longer episodes, but the remaining-time feature follows the training horizon (clipped at 0)."""

    def __init__(self, *a, obs_horizon, **kw):
        self.obs_horizon = obs_horizon
        super().__init__(*a, **kw)

    def _observe(self):
        obs, agent_id, node_obs, adj = super()._observe()
        remain = np.clip(1.0 - self.t / self.obs_horizon, 0.0, None)
        obs[..., 4] = remain[:, None]
        return obs, agent_id, node_obs, adj


def adjacent_choice(env, loader):
    """Food each LOAD agent targets (first adjacent alive food in N, S, W, E order), else -1."""
    E, N, G = env.E, env.N, env.G
    ar = np.arange(E)[:, None]
    choice = -np.ones((E, N), dtype=np.int64)
    for d in (NORTH, SOUTH, WEST, EAST):
        nb = env.agent_pos + DELTA[d]
        ok = ((nb >= 0) & (nb < G)).all(-1)
        fid = env.food_id[ar, np.clip(nb[..., 0], 0, G - 1), np.clip(nb[..., 1], 0, G - 1)]
        take = loader & ok & (fid >= 0) & (choice < 0)
        choice = np.where(take, fid, choice)
    return choice


@torch.no_grad()
def diagnose(args, actor_state, m, episodes, seed, horizon):
    from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy

    T0 = args.max_steps
    env = DiagEnv(episodes, m, args.kappa, sight=args.sight, max_steps=horizon,
                  cells_per_agent=args.cells_per_agent, food_per_agent=args.food_per_agent,
                  seed=seed, obs_horizon=T0)
    policy = GR_MAPPOPolicy(args, env.observation_space[0], env.share_observation_space[0],
                            env.node_observation_space[0], env.edge_observation_space[0],
                            env.action_space[0], device=torch.device("cpu"))
    policy.actor.load_state_dict(actor_state)
    policy.actor.eval()
    E, F, K, s = episodes, env.F, args.kappa, args.sight

    obs, agent_id, node_obs, adj = env.reset()
    rnn = np.zeros((E, m, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((E, m, 1), dtype=np.float32)

    active = np.ones(E, bool)                    # still in the first episode
    loaded_cnt = np.zeros(E, np.int64)
    food_curve = np.zeros((E, horizon))
    t_first = np.full(E, np.nan)
    t_last = np.full(E, np.nan)
    acc = dict(load_adj=0, load_blind=0, move=0, none=0, solo=0, steps=0, no_mate=0, no_ent=0)
    excess, n_load_events = 0, 0
    partners = [[set() for _ in range(m)] for _ in range(E)]
    took_part = np.zeros((E, m), bool)
    left = dict(adj=0, seen=0, unseen=0, n=0)

    for t in range(horizon):
        if t == T0:  # snapshot of the food left at the training horizon
            for e in np.flatnonzero(active):
                fa = np.flatnonzero(env.food_alive[e])
                if fa.size == 0:
                    continue
                fp = env.food_pos[e, fa]
                d = np.abs(fp[:, None, :] - env.agent_pos[e][None, :, :])
                cheb, manh = d.max(-1), d.sum(-1)
                left["adj"] += int((manh == 1).any(1).sum())
                left["seen"] += int((cheb <= s).any(1).sum())
                left["unseen"] += int((~(cheb <= s).any(1)).sum())
                left["n"] += fa.size

        act, rnn_out = policy.act(obs.reshape(-1, obs.shape[-1]), node_obs.reshape(-1, *node_obs.shape[2:]),
                                  adj.reshape(-1, *adj.shape[2:]), agent_id.reshape(-1, 1),
                                  rnn.reshape(-1, *rnn.shape[2:]), masks.reshape(-1, 1), deterministic=True)
        a = _t2n(act).reshape(E, m).astype(np.int64)
        rnn = _t2n(rnn_out).reshape(rnn.shape)

        # --- pre-step analysis on the current state (loaders do not move, so this is exact)
        loader = a == LOAD
        choice = adjacent_choice(env, loader)
        n_loaders = np.zeros((E, F), np.int64)
        ei, ii = np.nonzero(choice >= 0)
        np.add.at(n_loaders, (ei, choice[ei, ii]), 1)
        will_load = env.food_alive & (n_loaders >= K)

        if t < T0:
            act_mask = active[:, None] & np.ones((E, m), bool)
            acc["steps"] += int(act_mask.sum())
            acc["load_adj"] += int((loader & (choice >= 0) & act_mask).sum())
            acc["load_blind"] += int((loader & (choice < 0) & act_mask).sum())
            acc["none"] += int(((a == NONE) & act_mask).sum())
            acc["move"] += int(((a >= NORTH) & (a <= EAST) & act_mask).sum())
            solo = loader & (choice >= 0)
            solo &= ~np.take_along_axis(will_load, np.clip(choice, 0, None), 1)
            acc["solo"] += int((solo & act_mask).sum())
            # visibility of teammates / alive food
            ap = env.agent_pos
            dmate = np.abs(ap[:, :, None, :] - ap[:, None, :, :]).max(-1)
            dmate[:, np.arange(m), np.arange(m)] = 10 ** 6
            mate = (dmate <= s).any(-1)
            dfood = np.abs(ap[:, :, None, :] - env.food_pos[:, None, :, :]).max(-1)
            food_vis = ((dfood <= s) & env.food_alive[:, None, :]).any(-1)
            acc["no_mate"] += int((~mate & act_mask).sum())
            acc["no_ent"] += int((~mate & ~food_vis & act_mask).sum())

        for e, f in zip(*np.nonzero(will_load & active[:, None])):
            who = np.flatnonzero(choice[e] == f)
            if t < T0:
                excess += len(who) - K
                n_load_events += 1
                took_part[e, who] = True
                for i in who:
                    partners[e][i].update(int(j) for j in who if j != i)
            loaded_cnt[e] += 1
            if np.isnan(t_first[e]):
                t_first[e] = t + 1
            if loaded_cnt[e] == F:
                t_last[e] = t + 1

        obs, agent_id, node_obs, adj, _, dones, infos = env.step(a)
        rnn[dones] = 0.0
        masks = np.where(dones[..., None], 0.0, 1.0).astype(np.float32)
        food_curve[:, t] = loaded_cnt / F
        active &= ~dones[:, 0]
        if not active.any():
            food_curve[:, t + 1:] = food_curve[:, t:t + 1]
            break

    st = acc["steps"]
    out = {
        "m": m, "grid": env.G, "n_food": F, "episodes": E, "horizon": horizon, "train_horizon": T0,
        "food_curve": food_curve.mean(0).tolist(),
        "food_25": float(food_curve[:, T0 - 1].mean()), "food_50": float(food_curve[:, -1].mean()),
        "success_25": float((food_curve[:, T0 - 1] == 1).mean()),
        "success_50": float((food_curve[:, -1] == 1).mean()),
        "act_load_adj": acc["load_adj"] / st, "act_load_blind": acc["load_blind"] / st,
        "act_move": acc["move"] / st, "act_none": acc["none"] / st,
        "solo_load": acc["solo"] / st,
        "solo_load_share_of_adj_loads": acc["solo"] / max(acc["load_adj"], 1),
        "excess_loaders": excess / max(n_load_events, 1),
        "left_n": left["n"] / E,
        "left_adj_frac": left["adj"] / max(left["n"], 1),
        "left_seen_frac": left["seen"] / max(left["n"], 1),
        "left_unseen_frac": left["unseen"] / max(left["n"], 1),
        "t_first_load": float(np.nanmean(t_first)) if (~np.isnan(t_first)).any() else None,
        "t_last_load": float(np.nanmean(t_last)) if (~np.isnan(t_last)).any() else None,
        "no_mate_in_sight": acc["no_mate"] / st, "no_entity_in_sight": acc["no_ent"] / st,
        "participation": float(took_part.mean()),
    }
    if m >= 3:
        pp = [len(partners[e][i]) for e in range(E) for i in range(m) if took_part[e, i]]
        out["partners_per_agent"] = float(np.mean(pp)) if pp else None
    return out


def _t2n(x):
    return x.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--actor", default="actor.pt")
    p.add_argument("--test_num_agents", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--eval_episodes", type=int, default=200)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out_json", required=True)
    cli = p.parse_args()

    run_dir = Path(cli.model_dir).resolve().parent
    cfg = json.load(open(run_dir / "config.json"))
    args, _ = parse([])
    for k, v in cfg.items():
        setattr(args, k, v)
    args.use_render = True
    actor = torch.load(Path(cli.model_dir) / cli.actor, map_location="cpu")

    out = {"model_dir": str(cli.model_dir), "actor": cli.actor, "train_num_agents": cfg["num_agents"],
           "kappa": cfg["kappa"], "seed": cfg["seed"], "summary": []}
    for m in cli.test_num_agents:
        if m < args.kappa:
            continue
        r = diagnose(args, actor, m, cli.eval_episodes, cli.seed + m, cli.horizon)
        out["summary"].append(r)
        print(f"m={m:2d} food25 {r['food_25']:.2f} food50 {r['food_50']:.2f} succ25 {r['success_25']:.2f} "
              f"succ50 {r['success_50']:.2f} solo {r['solo_load']:.3f} left {r['left_n']:.2f} "
              f"(adj {r['left_adj_frac']:.2f} seen {r['left_seen_frac']:.2f}) "
              f"no_mate {r['no_mate_in_sight']:.2f} partners {r.get('partners_per_agent')}", flush=True)
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cli.out_json, "w"), indent=1)
    print("saved", cli.out_json)


if __name__ == "__main__":
    main()
