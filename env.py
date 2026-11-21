import numpy as np
import gymnasium as gym
from gymnasium import spaces


class HomeostaticForagingEnv(gym.Env):
    """
    2D Gridworld implementing Homeostatic Reinforcement Learning
    (Keramati & Gutkin, 2014) alongside a fixed-reward baseline.

    Observation:
        Dict with:
        - 'grid': (3, view_size, view_size) float32 egocentric tensor
                  Channels: [0: walls/OOB, 1: food, 2: toxin]
        - 'energy': (1,) float32 scalar in [0.0, 1.0]

    Actions:
        0: Up, 1: Down, 2: Left, 3: Right, 4: Stay
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 11,
        view_size: int = 5,
        n_food: int = 4,
        n_toxins: int = 3,
        max_steps: int = 300,
        # Homeostatic parameters (normalized to [0, 1])
        set_point: float = 0.7,
        init_energy: float = 0.5,
        metabolic_cost: float = 0.01,
        food_energy: float = 0.25,
        toxin_energy: float = -0.25,
        # Drive function D(H) = (|H* - H|^n)^(1/m)
        drive_n: float = 2.0,
        drive_m: float = 1.0,
        reward_scale: float = 10.0,
        # 'homeostatic' or 'fixed'
        reward_mode: str = "homeostatic",
        respawn_items: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert view_size % 2 == 1, "view_size must be odd for a centered egocentric view."
        assert reward_mode in ("homeostatic", "fixed"), "reward_mode must be 'homeostatic' or 'fixed'."

        self.grid_size = grid_size
        self.view_size = view_size
        self.half_view = view_size // 2
        self.n_food = n_food
        self.n_toxins = n_toxins
        self.max_steps = max_steps

        self.set_point = set_point
        self.init_energy = init_energy
        self.metabolic_cost = metabolic_cost
        self.food_energy = food_energy
        self.toxin_energy = toxin_energy
        self.drive_n = drive_n
        self.drive_m = drive_m
        self.reward_scale = reward_scale
        self.reward_mode = reward_mode
        self.respawn_items = respawn_items
        self.render_mode = render_mode

        # Action deltas: Up, Down, Left, Right, Stay
        self._action_to_delta = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0),
        }

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, view_size, view_size),
                    dtype=np.float32,
                ),
                "energy": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        # Internal state (initialized in reset)
        self.agent_pos = (0, 0)
        self.energy = self.init_energy
        self.food_positions: set[tuple[int, int]] = set()
        self.toxin_positions: set[tuple[int, int]] = set()
        self.steps = 0

    def _compute_drive(self, energy: float) -> float:
        """Keramati & Gutkin (2014) drive function: D(H) = (|H* - H|^n)^(1/m)."""
        return float((abs(self.set_point - energy) ** self.drive_n) ** (1.0 / self.drive_m))

    def _sample_empty_cell(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        while True:
            r = int(self.np_random.integers(0, self.grid_size))
            c = int(self.np_random.integers(0, self.grid_size))
            if (r, c) not in occupied:
                return (r, c)

    def _get_obs(self) -> dict[str, np.ndarray]:
        view = np.zeros((3, self.view_size, self.view_size), dtype=np.float32)
        ar, ac = self.agent_pos

        for dr in range(-self.half_view, self.half_view + 1):
            for dc in range(-self.half_view, self.half_view + 1):
                vr = dr + self.half_view
                vc = dc + self.half_view
                gr, gc = ar + dr, ac + dc

                if not (0 <= gr < self.grid_size and 0 <= gc < self.grid_size):
                    view[0, vr, vc] = 1.0  # Wall / out-of-bounds
                else:
                    if (gr, gc) in self.food_positions:
                        view[1, vr, vc] = 1.0
                    if (gr, gc) in self.toxin_positions:
                        view[2, vr, vc] = 1.0

        return {
            "grid": view,
            "energy": np.array([self.energy], dtype=np.float32),
        }

    def _nearest_manhattan_dist(self, items: set[tuple[int, int]]) -> float:
        if not items:
            return float("nan")
        ar, ac = self.agent_pos
        return float(min(abs(ar - r) + abs(ac - c) for r, c in items))

    def _get_info(self, ate_food: bool = False, ate_toxin: bool = False) -> dict:
        return {
            "energy": self.energy,
            "drive": self._compute_drive(self.energy),
            "ate_food": ate_food,
            "ate_toxin": ate_toxin,
            "dist_to_food": self._nearest_manhattan_dist(self.food_positions),
            "dist_to_toxin": self._nearest_manhattan_dist(self.toxin_positions),
            "agent_pos": self.agent_pos,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}

        self.steps = 0
        self.energy = float(options.get("init_energy", self.init_energy))
        self.respawn_items = bool(options.get("respawn_items", self.respawn_items))

        if "agent_pos" in options:
            self.agent_pos = tuple(options["agent_pos"])
        else:
            self.agent_pos = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
            )

        occupied = {self.agent_pos}

        if "food_positions" in options:
            self.food_positions = {tuple(p) for p in options["food_positions"]}
            occupied |= self.food_positions
        else:
            self.food_positions = set()
            for _ in range(self.n_food):
                pos = self._sample_empty_cell(occupied)
                self.food_positions.add(pos)
                occupied.add(pos)

        if "toxin_positions" in options:
            self.toxin_positions = {tuple(p) for p in options["toxin_positions"]}
            occupied |= self.toxin_positions
        else:
            self.toxin_positions = set()
            for _ in range(self.n_toxins):
                pos = self._sample_empty_cell(occupied)
                self.toxin_positions.add(pos)
                occupied.add(pos)

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        self.steps += 1
        prev_energy = self.energy
        prev_drive = self._compute_drive(prev_energy)

        # 1. Move agent (clip to grid boundaries)
        dr, dc = self._action_to_delta[int(action)]
        nr = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
        nc = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))
        self.agent_pos = (nr, nc)

        # 2. Apply metabolic decay
        self.energy -= self.metabolic_cost

        # 3. Check item interactions
        ate_food = False
        ate_toxin = False
        occupied = {self.agent_pos} | self.food_positions | self.toxin_positions

        if self.agent_pos in self.food_positions:
            ate_food = True
            self.energy += self.food_energy
            self.food_positions.remove(self.agent_pos)
            if self.respawn_items:
                self.food_positions.add(self._sample_empty_cell(occupied))

        elif self.agent_pos in self.toxin_positions:
            ate_toxin = True
            self.energy += self.toxin_energy
            self.toxin_positions.remove(self.agent_pos)
            if self.respawn_items:
                self.toxin_positions.add(self._sample_empty_cell(occupied))

        # Clamp energy to [0.0, 1.0]
        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        # 4. Compute reward
        if self.reward_mode == "homeostatic":
            new_drive = self._compute_drive(self.energy)
            reward = self.reward_scale * (prev_drive - new_drive)
        else:  # 'fixed' baseline
            reward = (1.0 if ate_food else 0.0) + (-1.0 if ate_toxin else 0.0)

        # 5. Termination & truncation
        terminated = bool(self.energy <= 0.0)
        truncated = bool(self.steps >= self.max_steps)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(ate_food, ate_toxin)

    def render(self):
        if self.render_mode != "ansi":
            return None
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U1")
        for r, c in self.food_positions:
            grid[r, c] = "F"
        for r, c in self.toxin_positions:
            grid[r, c] = "T"
        ar, ac = self.agent_pos
        grid[ar, ac] = "A"
        header = f"Step: {self.steps} | Energy: {self.energy:.2f} (Set point: {self.set_point:.2f})\n"
        return header + "\n".join(" ".join(row) for row in grid)


if __name__ == "__main__":
    # Quick sanity check for both reward modes
    for mode in ("homeostatic", "fixed"):
        env = HomeostaticForagingEnv(reward_mode=mode, render_mode="ansi")
        obs, info = env.reset(seed=42)
        total_r = 0.0
        for _ in range(20):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            total_r += r
            if term or trunc:
                break
        print(f"Mode: {mode:11s} | 20-step return: {total_r:+.3f} | Final energy: {info['energy']:.2f}")

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

from env import HomeostaticForagingEnv


class RecurrentActorCritic(nn.Module):
    def __init__(self, view_size: int = 5, action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out_dim = 32 * view_size * view_size

        self.gru = nn.GRU(input_size=cnn_out_dim + 1, hidden_size=hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization for stable PPO training
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, grid, energy, h_in, dones=None):
        """
        grid:   (B, T, 3, H, W)
        energy: (B, T, 1)
        h_in:   (1, B, hidden_dim)
        dones:  (B, T) float tensor; 1.0 indicates a new episode starts at step t
        """
        B, T = grid.shape[:2]
        feat = self.cnn(grid.reshape(B * T, *grid.shape[2:])).reshape(B, T, -1)
        x = torch.cat([feat, energy], dim=-1)

        # Step through time to mask hidden states cleanly at episode boundaries
        outputs = []
        h = h_in
        for t in range(T):
            if dones is not None:
                mask = (1.0 - dones[:, t]).view(1, B, 1)
                h = h * mask
            out_t, h = self.gru(x[:, t : t + 1, :], h)
            outputs.append(out_t)

        out = torch.cat(outputs, dim=1)  # (B, T, hidden_dim)
        logits = self.actor(out)
        values = self.critic(out).squeeze(-1)
        return logits, values, h


def make_env(reward_mode: str, seed: int):
    def _init():
        env = HomeostaticForagingEnv(reward_mode=reward_mode)
        env.reset(seed=seed)
        return env
    return _init


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training '{args.mode}' agent on {device} for {args.updates} updates...")

    # Vectorized environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.mode, args.seed + i) for i in range(args.num_envs)]
    )

    model = RecurrentActorCritic(view_size=5, action_dim=5, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Storage buffers for rollouts: shape (T, N, ...)
    T, N = args.num_steps, args.num_envs
    obs_grid = torch.zeros((T, N, 3, 5, 5), device=device)
    obs_energy = torch.zeros((T, N, 1), device=device)
    actions = torch.zeros((T, N), dtype=torch.long, device=device)
    log_probs = torch.zeros((T, N), device=device)
    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device)
    values = torch.zeros((T, N), device=device)
    # Store hidden state at every timestep so any chunk start has its exact h_0
    hidden_states = torch.zeros((T, N, args.hidden_dim), device=device)

    # Initialize environments and hidden state
    obs, _ = envs.reset(seed=args.seed)
    next_grid = torch.tensor(obs["grid"], dtype=torch.float32, device=device)
    next_energy = torch.tensor(obs["energy"], dtype=torch.float32, device=device)
    next_done = torch.zeros(N, dtype=torch.float32, device=device)
    h = torch.zeros((1, N, args.hidden_dim), device=device)

    start_time = time.time()

    for update in range(1, args.updates + 1):
        # 1. Collect rollouts
        model.eval()
        for t in range(T):
            obs_grid[t] = next_grid
            obs_energy[t] = next_energy
            dones[t] = next_done
            hidden_states[t] = h.squeeze(0)

            with torch.no_grad():
                logits, val, h = model(
                    next_grid.unsqueeze(1),
                    next_energy.unsqueeze(1),
                    h,
                    dones=next_done.unsqueeze(1),
                )
                dist = Categorical(logits=logits.squeeze(1))
                act = dist.sample()
                log_p = dist.log_prob(act)

            values[t] = val.squeeze(1)
            actions[t] = act
            log_probs[t] = log_p

            obs, rew, term, trunc, infos = envs.step(act.cpu().numpy())
            rewards[t] = torch.tensor(rew, dtype=torch.float32, device=device)
            next_done = torch.tensor(np.logical_or(term, trunc), dtype=torch.float32, device=device)
            next_grid = torch.tensor(obs["grid"], dtype=torch.float32, device=device)
            next_energy = torch.tensor(obs["energy"], dtype=torch.float32, device=device)

        # 2. Compute GAE
        with torch.no_grad():
            _, next_val, _ = model(
                next_grid.unsqueeze(1),
                next_energy.unsqueeze(1),
                h,
                dones=next_done.unsqueeze(1),
            )
            next_val = next_val.squeeze(1)

            advantages = torch.zeros_like(rewards)
            last_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_val
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Chunk rollouts into sequences of length seq_len for TBPTT
        assert T % args.seq_len == 0, "num_steps must be divisible by seq_len"
        num_chunks_per_env = T // args.seq_len
        total_chunks = N * num_chunks_per_env

        def to_chunks(tensor):
            # (T, N, ...) -> (N, T, ...) -> (N * num_chunks, seq_len, ...)
            return tensor.transpose(0, 1).reshape(total_chunks, args.seq_len, *tensor.shape[2:])

        b_grid = to_chunks(obs_grid)
        b_energy = to_chunks(obs_energy)
        b_actions = to_chunks(actions)
        b_log_probs = to_chunks(log_probs)
        b_advantages = to_chunks(advantages)
        b_returns = to_chunks(returns)
        b_dones = to_chunks(dones)
        # Initial hidden state for each chunk: (total_chunks, hidden_dim)
        b_h0 = to_chunks(hidden_states)[:, 0, :]

        # 4. PPO optimization epochs
        model.train()
        chunk_indices = np.arange(total_chunks)

        for _ in range(args.epochs):
            np.random.shuffle(chunk_indices)
            for start in range(0, total_chunks, args.batch_chunks):
                idx = chunk_indices[start : start + args.batch_chunks]

                logits, new_values, _ = model(
                    b_grid[idx],
                    b_energy[idx],
                    b_h0[idx].unsqueeze(0),
                    dones=b_dones[idx],
                )
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_log_probs[idx])
                adv = b_advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((new_values - b_returns[idx]) ** 2).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if update % 20 == 0 or update == 1:
            mean_rew = rewards.mean().item()
            mean_energy = obs_energy.mean().item()
            elapsed = time.time() - start_time
            print(
                f"Update {update:4d}/{args.updates} | "
                f"Mean Step Rew: {mean_rew:+.4f} | "
                f"Mean Energy: {mean_energy:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

    save_path = f"model_{args.mode}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")
    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="homeostatic", choices=["homeostatic", "fixed"])
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_chunks", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
