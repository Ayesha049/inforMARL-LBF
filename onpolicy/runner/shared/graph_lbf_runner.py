"""InforMARL runner for the vectorized LBF env.

Reuses GMPERunner (collect / insert / compute / train, unchanged) and only replaces the
training loop's logging: LBF episode statistics come from ``envs.completed`` (episodes that
ended during the rollout) instead of the Target task's per-agent info dicts. It also keeps an
actor snapshot every ``snapshot_every`` env steps (``models/snapshots/actor_<steps>.pt``);
``models/actor.pt`` stays the final model.
"""
import os
import time

import numpy as np
import torch

from onpolicy.runner.shared.graph_mpe_runner import GMPERunner


class GLBFRunner(GMPERunner):
    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        snap_every = int(getattr(self.all_args, "snapshot_every", 0))
        snap_dir = os.path.join(str(self.save_dir), "snapshots")
        if snap_every > 0:
            os.makedirs(snap_dir, exist_ok=True)
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                obs, agent_id, node_obs, adj, rewards, dones, infos = self.envs.step(actions_env)
                self.insert((obs, agent_id, node_obs, adj, agent_id, rewards, dones, infos, values, actions,
                             action_log_probs, rnn_states, rnn_states_critic))
            self.compute()
            train_infos = self.train()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()
            prev_steps = total_num_steps - self.episode_length * self.n_rollout_threads
            if snap_every > 0 and total_num_steps // snap_every > prev_steps // snap_every:
                torch.save(self.trainer.policy.actor.state_dict(),
                           os.path.join(snap_dir, f"actor_{total_num_steps}.pt"))

            if episode % self.log_interval == 0 or episode == episodes - 1:
                done = self.envs.completed
                self.envs.completed = []
                stats = {k: float(np.mean([d[k] for d in done])) if done else float("nan")
                         for k in ("team_return", "food_frac", "success", "length")}
                fps = total_num_steps / (time.time() - start)
                print(
                    f"Episode return {stats['team_return']:.3f} food {stats['food_frac']:.3f} "
                    f"success {stats['success']:.3f} length {stats['length']:.1f} ({len(done)} eps) \t"
                    f"Total timesteps: {total_num_steps} \t FPS {fps:.0f} \t"
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}",
                    flush=True,
                )
                train_infos["average_episode_rewards"] = stats["team_return"]
                self.log_train(train_infos, total_num_steps)
                self.log_env({f"lbf/{k}": [v] for k, v in stats.items()}, total_num_steps)
