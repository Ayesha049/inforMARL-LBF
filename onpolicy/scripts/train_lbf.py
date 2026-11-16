#!/usr/bin/env python
"""Train InforMARL (RMAPPO + graph network) on vectorized Level-Based Foraging.

Same structure as inforMARL-rep's onpolicy/scripts/train_mpe.py, with the MPE env replaced by
lbf.vec_lbf.GraphLBFVecEnv (one process, all rollout threads batched in NumPy).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

sys.path.append(os.path.abspath(os.getcwd()))

from onpolicy.config import get_config, graph_config  # noqa: E402
from utils.utils import print_args, print_box  # noqa: E402


def lbf_args(parser):
    parser.add_argument("--scenario_name", type=str, default="lbf")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--kappa", type=int, default=1, help="food level = agents needed to load")
    parser.add_argument("--sight", type=int, default=3, help="Chebyshev sight / edge radius (cells)")
    parser.add_argument("--max_steps", type=int, default=25, help="LBF episode length")
    parser.add_argument("--cells_per_agent", type=float, default=16.0)
    parser.add_argument("--food_per_agent", type=float, default=1.0)
    parser.add_argument("--num_obstacles", type=int, default=0)  # read by base_runner only
    parser.add_argument("--snapshot_every", type=int, default=1_000_000,
                        help="save an actor snapshot every this many env steps (0 = off)")
    return parser


def make_env(args, n_agents=None, num_envs=None, seed=None):
    from lbf.vec_lbf import GraphLBFVecEnv

    return GraphLBFVecEnv(
        num_envs or args.n_rollout_threads, n_agents or args.num_agents, args.kappa,
        sight=args.sight, max_steps=args.max_steps, cells_per_agent=args.cells_per_agent,
        food_per_agent=args.food_per_agent, seed=args.seed if seed is None else seed,
    )


def parse(argv):
    parser = lbf_args(get_config())
    all_args, parser = graph_config(argv, parser)
    all_args.env_name = "GraphLBF"
    all_args.max_edge_dist = all_args.sight + 0.5  # connect entities within sight (Chebyshev)
    all_args.num_embeddings = 3  # agent, food, loaded food
    return all_args, parser


def main(argv):
    all_args, _ = parse(argv)
    assert all_args.algorithm_name == "rmappo" and all_args.share_policy
    assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy

    if all_args.cuda and torch.cuda.is_available():
        print_box("Choose to use gpu...")
        device = torch.device("cuda:0")
    else:
        print_box("Choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.verbose:
        print_args(all_args)

    run_dir = (Path(__file__).resolve().parents[1] / "results" / all_args.env_name /
               f"k{all_args.kappa}" / all_args.experiment_name)
    runs = [int(p.name[3:]) for p in run_dir.glob("run*") if p.name[3:].isdigit()]
    run_dir = run_dir / f"run{max(runs, default=0) + 1}"
    os.makedirs(run_dir)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(all_args), f, indent=1, default=str)
    setproctitle.setproctitle(f"informarl-lbf-{all_args.experiment_name}@{all_args.user_name}")

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_env(all_args)
    from onpolicy.runner.shared.graph_lbf_runner import GLBFRunner

    all_args.use_wandb = False
    runner = GLBFRunner(dict(all_args=all_args, envs=envs, eval_envs=None, num_agents=all_args.num_agents,
                             device=device, run_dir=run_dir))
    runner.run()
    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
