import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HomeostaticForagingEnv(gym.Env):
    """
    Partially observable 2D foraging gridworld with:
    - Movement-dependent metabolic energy decay
    - Variable food energy yields
    - Optional noisy interoceptive energy observation
    - Three reward modes: 'homeostatic', 'fixed', 'fixed_death'
    """

    def __init__(
        self,
        reward_mode: str = "homeostatic",
        grid_size: int = 13,
        view_size: int = 5,
        n_food: int = 8,
        n_toxins: int = 4,
        max_steps: int = 200,
        set_point: float = 0.7,
        basal_decay: float = 0.004,
        move_cost: float = 0.002,
        energy_noise: float = 0.0,
    ):
        super().__init__()
        assert reward_mode in ("homeostatic", "fixed", "fixed_death")
        self.reward_mode = reward_mode
        self.grid_size = grid_size
        self.view_size = view_size
        self.half_view = view_size // 2
        self.n_food = n_food
        self.n_toxins = n_toxins
        self.max_steps = max_steps
        self.set_point = set_point
        self.basal_decay = basal_decay
        self.move_cost = move_cost
        self.energy_noise = energy_noise

        # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, shape=(3, view_size, view_size), dtype=np.float32),
                "energy": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        self.agent_pos = (grid_size // 2, grid_size // 2)
        self.energy = set_point
        self.food_map = {}   # {(r, c): energy_yield}
        self.toxins = set()
        self.respawn_items = True
        self.steps = 0

    def _random_empty_cell(self):
        while True:
            pos = (int(self.np_random.integers(0, self.grid_size)), int(self.np_random.integers(0, self.grid_size)))
            if pos != self.agent_pos and pos not in self.food_map and pos not in self.toxins:
                return pos

    def _sample_food_yield(self):
        return float(self.np_random.uniform(0.18, 0.35))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        self.steps = 0
        self.respawn_items = options.get("respawn_items", True)
        self.agent_pos = options.get(
            "agent_pos",
            (int(self.np_random.integers(0, self.grid_size)), int(self.np_random.integers(0, self.grid_size))),
        )
        # Initial-energy curriculum: sample across hungry and sated states
        self.energy = float(options.get("init_energy", self.np_random.uniform(0.25, 0.90)))

        self.food_map = {}
        self.toxins = set()

        if "food_positions" in options:
            for pos in options["food_positions"]:
                self.food_map[tuple(pos)] = 0.25
        else:
            for _ in range(self.n_food):
                self.food_map[self._random_empty_cell()] = self._sample_food_yield()

        if "toxin_positions" in options:
            for pos in options["toxin_positions"]:
                self.toxins.add(tuple(pos))
        else:
            for _ in range(self.n_toxins):
                self.toxins.add(self._random_empty_cell())

        return self._get_obs(), self._get_info(ate_food=False, ate_toxin=False)

    def _get_obs(self):
        grid = np.zeros((3, self.view_size, self.view_size), dtype=np.float32)
        ar, ac = self.agent_pos

        for dr in range(-self.half_view, self.half_view + 1):
            for dc in range(-self.half_view, self.half_view + 1):
                r, c = ar + dr, ac + dc
                vr, vc = dr + self.half_view, dc + self.half_view
                if r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size:
                    grid[2, vr, vc] = 1.0  # Wall channel
                else:
                    if (r, c) in self.food_map:
                        # Intensity reflects food yield
                        grid[0, vr, vc] = min(1.0, self.food_map[(r, c)] / 0.35)
                    if (r, c) in self.toxins:
                        grid[1, vr, vc] = 1.0

        obs_e = self.energy
        if self.energy_noise > 0.0:
            obs_e = float(np.clip(obs_e + self.np_random.normal(0.0, self.energy_noise), 0.0, 1.0))

        return {"grid": grid, "energy": np.array([obs_e], dtype=np.float32)}

    def _get_info(self, ate_food: bool, ate_toxin: bool):
        if self.food_map:
            ar, ac = self.agent_pos
            dist = min(abs(ar - fr) + abs(ac - fc) for (fr, fc) in self.food_map)
        else:
            dist = self.grid_size * 2
        return {
            "energy": self.energy,
            "dist_to_food": dist,
            "ate_food": ate_food,
            "ate_toxin": ate_toxin,
        }

    def step(self, action: int):
        self.steps += 1
        prev_energy = self.energy

        # Movement
        deltas = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        dr, dc = deltas[int(action)]
        nr = int(np.clip(self.agent_pos[0] + dr, 0, self.grid_size - 1))
        nc = int(np.clip(self.agent_pos[1] + dc, 0, self.grid_size - 1))
        moved = (nr, nc) != self.agent_pos
        self.agent_pos = (nr, nc)

        # Movement-dependent energy decay
        decay = self.basal_decay + (self.move_cost if moved else 0.0)
        self.energy = max(0.0, self.energy - decay)

        ate_food, ate_toxin = False, False
        if self.agent_pos in self.food_map:
            ate_food = True
            food_yield = self.food_map.pop(self.agent_pos)
            self.energy = min(1.0, self.energy + food_yield)
            if self.respawn_items:
                self.food_map[self._random_empty_cell()] = self._sample_food_yield()
        elif self.agent_pos in self.toxins:
            ate_toxin = True
            self.toxins.remove(self.agent_pos)
            self.energy = max(0.0, self.energy - 0.30)
            if self.respawn_items:
                self.toxins.add(self._random_empty_cell())

        terminated = self.energy <= 0.0
        truncated = self.steps >= self.max_steps

        # Reward computation
        if self.reward_mode == "homeostatic":
            prev_drive = abs(prev_energy - self.set_point)
            curr_drive = abs(self.energy - self.set_point)
            reward = (prev_drive - curr_drive) * 5.0
            if terminated:
                reward -= 2.0
        elif self.reward_mode == "fixed":
            reward = (1.0 if ate_food else 0.0) - (1.0 if ate_toxin else 0.0) - 0.01
        else:  # fixed_death
            reward = (1.0 if ate_food else 0.0) - (1.0 if ate_toxin else 0.0) - 0.01
            if terminated:
                reward -= 5.0

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(ate_food, ate_toxin)
