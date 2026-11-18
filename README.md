# InforMARL on Level-Based Foraging (Phase 0b)

Team-size transfer of InforMARL (Nayak et al., ICML 2023) on Level-Based Foraging (LBF), an environment the
original paper does not evaluate. Train with N agents, test zero-shot with M agents, at two coupling levels.

The InforMARL code (graph actor-critic, GNN, RMAPPO trainer, buffers, runner) is imported unchanged from
[Ayesha049/inforMARL-rep](https://github.com/Ayesha049/inforMARL-rep) at `b2eae80`, the version whose
Table 2 / Figure 3 reproduction is reported in the thesis repo (`docs/06_phase0.md`). See the first commit of this
repo for the byte-identical import.

## Design

| Item | Choice |
|---|---|
| Coupling κ | All agents level 1, every food item level κ, so exactly κ adjacent agents must LOAD together. κ ∈ {1, 2} |
| Team sizes | Train N ∈ {2, 4, 8}; test M ∈ {2, 4, 6, 8, 12, 16} |
| Density | Constant: grid side `ceil(sqrt(16 · N))` (6, 8, 10, 12, 14, 16 for N = 2 … 16), food count = N |
| Episode | 25 steps (InforMARL's Target length); ends early when all food is loaded |
| Reward | Shared: each agent gets (food loaded this step) / (food at reset); team return ∈ [0, 1] = fraction of food loaded |
| Observation | Local, sight radius 3 (Chebyshev). Ego vector: wall distances clipped to sight, remaining episode fraction. Graph: nodes = all agents and food, features relative to the ego agent `[d_row/3, d_col/3, level, type]`, edges only between entities within sight of each other |
| Algorithm | RMAPPO + InforMARL GNN, hyper-parameters identical to the Target reproduction (`scripts/train_one.sh`) |
| Seeds | 3 per (κ, N) |
| Metrics | Team return, food fraction, success (all food loaded), normalised episode length, food per agent, retention ρ(N→M) = P(N, M) / P(M, M) for M ∈ {2, 4, 8} |

## LBF rules

Follow [semitable/lb-foraging](https://github.com/semitable/lb-foraging): moves into walls or food are no-ops;
agents that target the same cell all stay; swaps are allowed; food is placed off the border and never in another
food's 8-neighbourhood; a loading agent loads the first adjacent food in N, S, W, E order. Two deliberate
differences: collision resolution is iterated until no two agents share a cell (lb-foraging resolves once), and the
reward is shared rather than split between loaders. The env is batched in NumPy (`lbf/vec_lbf.py`, about 2·10⁵
env-steps/s at N = 4), so training time is the PPO update, as on Target.

Rule tests: `python tests/test_vec_lbf.py`.

## Changes to the imported InforMARL code

| File | Change |
|---|---|
| `onpolicy/runner/shared/base_runner.py` | Graph branch selected by `env_name.startswith("Graph")` instead of `== "GraphMPE"` |
| `onpolicy/runner/shared/graph_lbf_runner.py` (new) | Subclass of `GMPERunner`; only the logging of the training loop differs (LBF episode stats) |
| `onpolicy/scripts/train_lbf.py`, `onpolicy/scripts/eval_transfer.py` (new) | Train and train-N / test-M evaluation for LBF |

## Run

```bash
bash scripts/train_one.sh <kappa> <N> <seed> <env_steps> [gpu]          # logs/train_lbf_k{K}_n{N}_s{S}.log
python onpolicy/scripts/eval_transfer.py --model_dir <run_dir>/models \
    --test_num_agents 2 4 6 8 12 16 --eval_episodes 200 --out_json results/eval/<name>.json
```

Uses the `informarl` conda env of inforMARL-rep (`scripts/repro/setup_env.sh` there).

## Status

- 2026-11-16: pilot (N = 4, seed 1, 10M steps) fixed the budget: κ = 1 3M, κ = 2 10M env steps; actor snapshots every 1M.
- 2026-11-18: main grid done. 18 runs (κ ∈ {1, 2} × N ∈ {2, 4, 8} × 3 seeds), each evaluated at M ∈ {2, 4, 6, 8, 12, 16}
  with 200 episodes (`results/eval/main/`, on the server). Results and analysis: thesis repo `docs/06_phase0.md`, Sections 5–7.
