"""Vectorized Level-Based Foraging (LBF) with a coupling knob and InforMARL graph observations.

Rules follow semitable/lb-foraging (Papoudakis et al., 2021, "Benchmarking Multi-Agent Deep RL
Algorithms in Cooperative Tasks"), specialised to:

* every agent has level 1 and every food item has level ``kappa``, so a food item is loaded
  only when at least ``kappa`` adjacent agents choose LOAD in the same step: coupling is exact;
* constant density: the grid side and the food count scale with the number of agents
  (``grid = ceil(sqrt(cells_per_agent * N))``, ``food = round(food_per_agent * N)``).

Actions: 0 NONE, 1 NORTH (row - 1), 2 SOUTH (row + 1), 3 WEST (col - 1), 4 EAST (col + 1), 5 LOAD.

Kept from lb-foraging:
* a move is invalid (agent stays) if it leaves the grid or enters a food cell;
* agents that target the same cell all stay; swaps are allowed;
* food is placed away from the border and never in the 8-neighbourhood of other food;
  agents are placed on any free cell;
* a loading agent loads the first adjacent food in the order N, S, W, E;
* the episode ends when all food is loaded or after ``max_steps``.

Deliberate differences (documented in README.md):
* collision resolution is iterated to a fixed point, so two agents never share a cell
  (lb-foraging resolves once, which can leave two agents on one cell in chained moves);
* the reward is shared: every agent receives the team reward
  ``(sum of levels of food loaded this step) / (sum of all food levels at reset)``,
  so the team return is in [0, 1] and equals the fraction of food loaded
  (lb-foraging gives each loader its share; InforMARL's Target task is also shared-reward).

Observations, per agent i (InforMARL interface, see onpolicy/runner/shared/graph_mpe_runner.py):
* ``obs``      (5,)      distances to the N, S, W, E walls clipped to ``sight`` and divided by
                         it, and the remaining fraction of the episode;
* ``node_obs`` (K, 4)    one row per entity (N agents, then F food):
                         [d_row / sight, d_col / sight, level, entity_type], relative to agent i;
                         entity_type 0 = agent, 1 = food, 2 = food already loaded;
* ``adj``      (K, K)    Chebyshev distance between entities; pairs farther apart than
                         ``sight``, and loaded food, get distance FAR, i.e. no edge
                         (the GNN connects 0 < d < max_edge_dist, use max_edge_dist = sight + 0.5);
* ``agent_id`` (1,)      the node index of agent i (= i).
Only entities within ``sight`` of each other are connected, as in InforMARL's local graph.
"""
from __future__ import annotations

import math

import numpy as np

NONE, NORTH, SOUTH, WEST, EAST, LOAD = range(6)
N_ACTIONS = 6
DELTA = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int64)
ENT_AGENT, ENT_FOOD, ENT_LOADED = 0, 1, 2
NUM_ENTITY_TYPES = 3
FAR = 1e3  # distance used for "no edge" (loaded food)


def default_grid(n_agents: int, cells_per_agent: float) -> int:
    return int(math.ceil(math.sqrt(cells_per_agent * n_agents)))


class VecLBF:
    def __init__(
        self,
        num_envs: int,
        n_agents: int,
        kappa: int,
        sight: int = 3,
        max_steps: int = 50,
        cells_per_agent: float = 16.0,
        food_per_agent: float = 1.0,
        grid: int | None = None,
        n_food: int | None = None,
        seed: int = 0,
    ):
        assert n_agents >= kappa, "need at least kappa agents to load anything"
        self.E, self.N, self.kappa = num_envs, n_agents, kappa
        self.sight, self.max_steps = sight, max_steps
        self.G = grid if grid is not None else default_grid(n_agents, cells_per_agent)
        self.F = n_food if n_food is not None else max(1, int(round(food_per_agent * n_agents)))
        self.K = self.N + self.F
        interior = (self.G - 2) ** 2
        # food needs a 3x3 exclusion zone; leave plenty of room
        assert self.F * 4 <= interior, f"grid {self.G} too small for {self.F} food"
        assert self.N + self.F <= self.G * self.G
        self.rng = np.random.default_rng(seed)

        self.agent_pos = np.zeros((self.E, self.N, 2), dtype=np.int64)
        self.food_pos = np.zeros((self.E, self.F, 2), dtype=np.int64)
        self.food_alive = np.zeros((self.E, self.F), dtype=bool)
        self.food_id = -np.ones((self.E, self.G, self.G), dtype=np.int64)  # food index per cell
        self.t = np.zeros(self.E, dtype=np.int64)
        self.ep_return = np.zeros(self.E, dtype=np.float64)
        self.completed: list[dict] = []  # finished-episode stats, drained by the runner

    # ------------------------------------------------------------------ reset
    def _reset_env(self, e: int) -> None:
        G, rng = self.G, self.rng
        blocked = np.zeros((G, G), dtype=bool)  # cells where new food may not go
        self.food_id[e] = -1
        for f in range(self.F):
            while True:
                r, c = rng.integers(1, G - 1, size=2)
                if not blocked[r, c]:
                    break
            self.food_pos[e, f] = (r, c)
            self.food_id[e, r, c] = f
            blocked[max(r - 1, 0): r + 2, max(c - 1, 0): c + 2] = True
        self.food_alive[e] = True
        free = np.flatnonzero(self.food_id[e].ravel() < 0)
        cells = rng.choice(free, size=self.N, replace=False)
        self.agent_pos[e, :, 0], self.agent_pos[e, :, 1] = np.divmod(cells, G)
        self.t[e] = 0
        self.ep_return[e] = 0.0

    def reset(self):
        for e in range(self.E):
            self._reset_env(e)
        return self._observe()

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray):
        """actions: (E, N) ints, or (E, N, 6) one-hot as produced by the InforMARL runner."""
        a = np.asarray(actions)
        if a.ndim == 3:
            a = a.argmax(-1)
        a = a.astype(np.int64)
        E, N, G = self.E, self.N, self.G
        ar = np.arange(E)[:, None]

        # --- movement
        target = self.agent_pos + DELTA[a]
        inside = ((target >= 0) & (target < G)).all(-1)
        tr, tc = np.clip(target[..., 0], 0, G - 1), np.clip(target[..., 1], 0, G - 1)
        on_food = self.food_id[ar, tr, tc] >= 0
        stay = ~inside | on_food | (a == NONE) | (a == LOAD)
        while True:
            dest = np.where(stay[..., None], self.agent_pos, target)
            cell = dest[..., 0] * G + dest[..., 1]
            counts = np.zeros((E, G * G), dtype=np.int64)
            np.add.at(counts, (np.repeat(np.arange(E), N), cell.ravel()), 1)
            conflict = (counts[ar, cell] > 1) & ~stay
            if not conflict.any():
                break
            stay |= conflict
        self.agent_pos = dest

        # --- loading: each loader picks the first adjacent alive food in N, S, W, E order
        loader = a == LOAD
        choice = -np.ones((E, N), dtype=np.int64)
        for d in (NORTH, SOUTH, WEST, EAST):
            nb = self.agent_pos + DELTA[d]
            ok = ((nb >= 0) & (nb < G)).all(-1)
            fid = self.food_id[ar, np.clip(nb[..., 0], 0, G - 1), np.clip(nb[..., 1], 0, G - 1)]
            take = loader & ok & (fid >= 0) & (choice < 0)
            choice = np.where(take, fid, choice)
        n_loaders = np.zeros((E, self.F), dtype=np.int64)
        e_idx, i_idx = np.nonzero(choice >= 0)
        np.add.at(n_loaders, (e_idx, choice[e_idx, i_idx]), 1)  # agent level is 1
        loaded = self.food_alive & (n_loaders >= self.kappa)
        if loaded.any():
            le, lf = np.nonzero(loaded)
            self.food_id[le, self.food_pos[le, lf, 0], self.food_pos[le, lf, 1]] = -1
            self.food_alive &= ~loaded
        team_r = loaded.sum(1) / self.F  # food levels are all kappa, so the ratio is a count ratio

        self.t += 1
        self.ep_return += team_r
        all_loaded = ~self.food_alive.any(1)
        done_env = all_loaded | (self.t >= self.max_steps)

        infos = [{} for _ in range(E)]
        for e in np.flatnonzero(done_env):
            stats = {
                "food_frac": 1.0 - self.food_alive[e].mean(),
                "success": float(all_loaded[e]),
                "length": float(self.t[e]),
                "team_return": float(self.ep_return[e]),
            }
            self.completed.append(stats)
            infos[e] = dict(stats, terminal=True)
            self._reset_env(e)

        rewards = np.repeat(team_r[:, None, None], N, axis=1).astype(np.float32)
        dones = np.repeat(done_env[:, None], N, axis=1)
        obs, agent_id, node_obs, adj = self._observe()
        return obs, agent_id, node_obs, adj, rewards, dones, infos

    # ------------------------------------------------------------------ observations
    def _observe(self):
        E, N, F, K, G, s = self.E, self.N, self.F, self.K, self.G, self.sight
        pos = np.concatenate([self.agent_pos, self.food_pos], axis=1)  # (E, K, 2)

        # ego vector
        r, c = self.agent_pos[..., 0], self.agent_pos[..., 1]
        walls = np.stack([r, G - 1 - r, c, G - 1 - c], -1).clip(max=s) / s
        remain = np.repeat((1.0 - self.t / self.max_steps)[:, None, None], N, axis=1)
        obs = np.concatenate([walls, remain], -1).astype(np.float32)

        # adjacency: Chebyshev distance, no edges beyond sight or to loaded food
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = np.abs(diff).max(-1).astype(np.float32)  # (E, K, K)
        alive = np.concatenate([np.ones((E, N), bool), self.food_alive], 1)
        dist[~alive[:, :, None] | ~alive[:, None, :]] = FAR
        dist[dist > s] = FAR
        adj = np.repeat(dist[:, None], N, axis=1)  # same graph for every ego agent

        # node features relative to each ego agent
        rel = (pos[:, None, :, :] - self.agent_pos[:, :, None, :]).astype(np.float32) / s
        level = np.concatenate([np.ones((E, N)), np.full((E, F), float(self.kappa))], 1)
        etype = np.concatenate([np.full((E, N), ENT_AGENT), np.where(self.food_alive, ENT_FOOD, ENT_LOADED)], 1)
        node_obs = np.concatenate(
            [rel, np.repeat(level[:, None, :, None], N, 1), np.repeat(etype[:, None, :, None], N, 1)], -1
        ).astype(np.float32)
        node_obs[..., :2] *= alive[:, None, :, None]  # loaded food carries no position

        agent_id = np.repeat(np.arange(N)[None, :, None], E, 0)
        return obs, agent_id, node_obs, adj

    # ------------------------------------------------------------------ gym-style spaces for the InforMARL runner
    def spaces(self):
        from gym import spaces

        def box(shape):
            return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        n = self.N
        return dict(
            observation_space=[box((5,))] * n,
            share_observation_space=[box((5 * n,))] * n,
            node_observation_space=[box((self.K, 4))] * n,
            adj_observation_space=[box((self.K, self.K))] * n,
            edge_observation_space=[box((1,))] * n,
            agent_id_observation_space=[box((1,))] * n,
            share_agent_id_observation_space=[box((n,))] * n,
            action_space=[spaces.Discrete(N_ACTIONS)] * n,
        )

    def render_ascii(self, e: int = 0) -> str:
        g = [["." for _ in range(self.G)] for _ in range(self.G)]
        for f in range(self.F):
            if self.food_alive[e, f]:
                r, c = self.food_pos[e, f]
                g[r][c] = "F"
        for i in range(self.N):
            r, c = self.agent_pos[e, i]
            g[r][c] = str(i % 10)
        return "\n".join("".join(row) for row in g)


class GraphLBFVecEnv(VecLBF):
    """VecLBF with the attribute interface the InforMARL runner and buffer expect."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        for k, v in self.spaces().items():
            setattr(self, k, v)
        self.num_envs = self.E

    def close(self):
        pass
