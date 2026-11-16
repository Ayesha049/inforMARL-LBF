"""Rule tests for lbf/vec_lbf.py. Run: python -m pytest tests/ (or python tests/test_vec_lbf.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lbf.vec_lbf import EAST, LOAD, NONE, NORTH, SOUTH, WEST, VecLBF, default_grid  # noqa: E402


def make(n=2, kappa=1, F=1, G=6, E=1, **kw):
    env = VecLBF(E, n, kappa, grid=G, n_food=F, **kw)
    env.reset()
    return env


def place(env, agents, foods, e=0):
    env.food_id[e] = -1
    for f, (r, c) in enumerate(foods):
        env.food_pos[e, f] = (r, c)
        env.food_id[e, r, c] = f
    env.food_alive[e] = True
    for i, (r, c) in enumerate(agents):
        env.agent_pos[e, i] = (r, c)
    env.t[e] = 0
    env.ep_return[e] = 0


def act(env, *a):
    return env.step(np.array([a]))


def test_constant_density():
    for n in (2, 4, 6, 8, 12, 16):
        env = VecLBF(4, n, 1)
        assert env.F == n and env.G == default_grid(n, 16)
        assert abs(env.G ** 2 / n - 16) / 16 < 0.25  # cells per agent stays near 16


def test_reset_placement():
    env = VecLBF(64, 8, 2, seed=1)
    env.reset()
    for e in range(env.E):
        fp = env.food_pos[e]
        assert ((fp >= 1) & (fp <= env.G - 2)).all()  # no food on the border
        d = np.abs(fp[:, None] - fp[None]).max(-1) + np.eye(env.F, dtype=int) * 99
        assert d.min() >= 2  # no food in another food's 8-neighbourhood
        cells = {tuple(p) for p in env.agent_pos[e]}
        assert len(cells) == env.N  # agents on distinct cells
        assert not cells & {tuple(p) for p in fp}  # and not on food


def test_moves_walls_and_food():
    env = make(G=5)
    place(env, [(0, 0), (2, 3)], [(2, 2)])
    act(env, NORTH, WEST)  # wall; food cell
    assert env.agent_pos[0].tolist() == [[0, 0], [2, 3]]
    act(env, SOUTH, NORTH)
    assert env.agent_pos[0].tolist() == [[1, 0], [1, 3]]


def test_same_target_both_stay_and_swap_allowed():
    env = make(G=6, F=1)
    place(env, [(1, 1), (1, 3)], [(4, 4)])
    act(env, EAST, WEST)  # both target (1, 2)
    assert env.agent_pos[0].tolist() == [[1, 1], [1, 3]]
    place(env, [(1, 1), (1, 2)], [(4, 4)])
    act(env, EAST, WEST)  # swap
    assert env.agent_pos[0].tolist() == [[1, 2], [1, 1]]


def test_chained_conflict_never_overlaps():
    # agent 0 at (1,1) -> (1,2) conflicts with agent 2 at (1,3) -> (1,2); agent 1 at (2,1) -> (1,1)
    env = make(n=3, G=6, F=1)
    place(env, [(1, 1), (2, 1), (1, 3)], [(4, 4)])
    act(env, EAST, NORTH, WEST)
    pos = env.agent_pos[0].tolist()
    assert pos[0] == [1, 1] and pos[2] == [1, 3] and pos[1] == [2, 1]  # 1 is blocked by 0 staying
    assert len({tuple(p) for p in pos}) == 3


def test_kappa1_single_loader():
    env = make(n=2, kappa=1, F=2, G=7)
    place(env, [(1, 2), (5, 5)], [(2, 2), (4, 1)])
    _, _, _, _, r, d, _ = act(env, LOAD, NONE)  # agent 0 is north of food 0
    assert np.allclose(r, 0.5) and not env.food_alive[0, 0] and env.food_alive[0, 1] and not d.any()


def test_kappa2_needs_two_loaders():
    env = make(n=3, kappa=2, F=1, G=6)
    place(env, [(1, 2), (2, 1), (4, 4)], [(2, 2)])
    _, _, _, _, r, d, _ = act(env, LOAD, NONE, NONE)
    assert np.allclose(r, 0) and env.food_alive[0, 0]
    _, _, _, _, r, d, infos = act(env, LOAD, LOAD, NONE)
    assert np.allclose(r, 1.0) and d.all()  # all food loaded -> episode ends, every agent gets the team reward
    assert infos[0]["success"] == 1.0 and infos[0]["food_frac"] == 1.0 and infos[0]["length"] == 2
    assert env.t[0] == 0 and env.food_alive[0].all()  # auto-reset


def test_loader_not_adjacent_does_nothing():
    env = make(n=2, kappa=1, F=1, G=6)
    place(env, [(1, 1), (4, 4)], [(2, 2)])  # diagonal is not adjacent
    _, _, _, _, r, _, _ = act(env, LOAD, LOAD)
    assert np.allclose(r, 0) and env.food_alive[0, 0]


def test_time_limit():
    env = make(n=2, kappa=2, F=1, G=6, max_steps=3)
    place(env, [(0, 0), (5, 5)], [(2, 2)])
    for k in range(3):
        _, _, _, _, r, d, infos = act(env, NONE, NONE)
    assert d.all() and infos[0]["success"] == 0.0 and infos[0]["food_frac"] == 0.0 and infos[0]["length"] == 3


def test_observation_graph():
    env = make(n=2, kappa=2, F=2, G=10, sight=2)
    place(env, [(1, 1), (8, 8)], [(2, 3), (6, 6)])
    obs, aid, node, adj = env._observe()
    K = env.K
    assert obs.shape == (1, 2, 5) and node.shape == (1, 2, K, 4) and adj.shape == (1, 2, K, K)
    assert aid[0, :, 0].tolist() == [0, 1]
    A = adj[0, 0]
    assert A[0, 2] == 2 and A[0, 1] > 2 and A[1, 3] == 2  # agent0-food0 within sight, agents far apart
    assert np.allclose(A, A.T) and np.allclose(adj[0, 0], adj[0, 1])
    # node features relative to ego 0: food0 at (+1, +2) / sight
    assert np.allclose(node[0, 0, 2], [0.5, 1.0, 2.0, 1.0])
    assert np.allclose(node[0, 1, 0, :2], [(1 - 8) / 2, (1 - 8) / 2])
    # walls for agent 0 at (1,1): N=1, S=8->clip 2, W=1, E=8->clip 2, divided by sight
    assert np.allclose(obs[0, 0, :4], [0.5, 1, 0.5, 1]) and obs[0, 0, 4] == 1.0
    # loaded food: no edges, type 2, zero position
    env.food_alive[0, 0] = False
    _, _, node, adj = env._observe()
    assert (adj[0, 0, 2] >= 1e3).all() and node[0, 0, 2, 3] == 2 and np.allclose(node[0, 0, 2, :2], 0)


def test_seed_determinism():
    def roll(seed):
        env = VecLBF(8, 4, 2, seed=seed)
        env.reset()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(120):
            o, _, n, a, r, d, _ = env.step(rng.integers(0, 6, size=(8, 4)))
            out.append((n.sum(), r.sum(), d.sum()))
        return out
    assert roll(3) == roll(3) and roll(3) != roll(4)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
