import glob
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from env import HomeostaticForagingEnv
from train_ppo import RecurrentActorCritic

torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = "runs"


def load_model(ckpt_path: str, hidden_dim: int) -> RecurrentActorCritic:
    model = RecurrentActorCritic(view_size=5, action_dim=5, hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def step_model(model, obs, h, hide_energy: bool, steer_vec=None, alpha: float = 0.0):
    grid = torch.tensor(obs["grid"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    energy = torch.tensor(obs["energy"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    if hide_energy:
        energy = torch.zeros_like(energy)

    feat = model.cnn(grid.reshape(1, *grid.shape[2:])).reshape(1, 1, -1)
    x = torch.cat([feat, energy], dim=-1)
    out, h_next = model.gru(x, h)

    if steer_vec is not None and alpha != 0.0:
        delta = alpha * steer_vec.view(1, 1, -1)
        h_next = h_next + delta
        out = out + delta

    logits = model.actor(out).squeeze(0).squeeze(0)
    action = int(torch.argmax(logits).item())
    return action, h_next


def collect_episodes(model, hide_energy: bool, n_episodes: int = 16, max_steps: int = 150, seed_offset: int = 1000):
    """Collect hidden states and true energy grouped by episode."""
    env = HomeostaticForagingEnv(reward_mode="homeostatic", max_steps=max_steps)
    H_eps, E_eps, devs = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        h = torch.zeros((1, 1, model.hidden_dim), device=DEVICE)
        h_ep, e_ep = [], []
        done = False
        while not done:
            action, h = step_model(model, obs, h, hide_energy=hide_energy)
            h_ep.append(h.squeeze(0).squeeze(0).cpu().numpy())
            e_ep.append(info["energy"])
            devs.append(abs(info["energy"] - env.set_point))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        H_eps.append(np.array(h_ep))
        E_eps.append(np.array(e_ep))

    return H_eps, E_eps, float(np.mean(devs))


def episode_split_probe(H_eps, E_eps):
    """Train Ridge probe on first half of episodes, evaluate R^2 on held-out second half."""
    mid = len(H_eps) // 2
    H_train = np.concatenate(H_eps[:mid], axis=0)
    E_train = np.concatenate(E_eps[:mid], axis=0)
    H_test = np.concatenate(H_eps[mid:], axis=0)
    E_test = np.concatenate(E_eps[mid:], axis=0)

    probe = Ridge(alpha=1.0).fit(H_train, E_train)
    r2 = float(r2_score(E_test, probe.predict(H_test)))

    # Unit vector pointing in the direction of INCREASING energy
    w = probe.coef_.astype(np.float32)
    norm = np.linalg.norm(w)
    w_hat = w / (norm + 1e-8)
    return r2, w_hat


def run_steering_dose_response(model, hide_energy: bool, w_hat: np.ndarray, alphas, trials: int = 12, max_steps: int = 30):
    """
    Place agent at center with energy near set point (0.65) and food at distance 3.
    Negative alpha steers hidden state toward LOW energy (should increase foraging).
    Positive alpha steers hidden state toward HIGH energy (should suppress foraging).
    """
    env = HomeostaticForagingEnv(reward_mode="homeostatic", grid_size=11, max_steps=max_steps)
    steer_vec = torch.tensor(w_hat, dtype=torch.float32, device=DEVICE)
    probs = []

    for alpha in alphas:
        ate = 0
        for t in range(trials):
            food_pos = (5, 8) if t % 2 == 0 else (2, 5)
            obs, _ = env.reset(
                seed=2000 + t,
                options={
                    "agent_pos": (5, 5),
                    "init_energy": 0.65,
                    "food_positions": [food_pos],
                    "toxin_positions": [],
                    "respawn_items": False,
                },
            )
            h = torch.zeros((1, 1, model.hidden_dim), device=DEVICE)
            for _ in range(max_steps):
                action, h = step_model(model, obs, h, hide_energy=hide_energy, steer_vec=steer_vec, alpha=alpha)
                obs, _, term, trunc, info = env.step(action)
                if info["ate_food"]:
                    ate += 1
                    break
                if term or trunc:
                    break
        probs.append(ate / trials)
    return np.array(probs)


def mean_ci95(values):
    arr = np.array(values, dtype=float)
    mean = np.mean(arr, axis=0)
    ci = 1.96 * np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(mean)
    return mean, ci


def main():
    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "*")))
    if not run_dirs:
        print("No runs found in 'runs/'. Wait for run_matrix.py to finish.")
        return

    # 1. Final checkpoint evaluation across the 2x2x2 matrix
    matrix_r2 = {}
    matrix_dev = {}
    steering_curves = {}
    alphas = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    print("Evaluating final checkpoints across all completed runs...")
    for rdir in run_dirs:
        final_ckpt = os.path.join(rdir, "final.pt")
        cfg_path = os.path.join(rdir, "config.json")
        if not (os.path.exists(final_ckpt) and os.path.exists(cfg_path)):
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        key = (cfg["mode"], int(not cfg["hide_energy"]), cfg["hidden_dim"])
        model = load_model(final_ckpt, cfg["hidden_dim"])
        H_eps, E_eps, mean_dev = collect_episodes(model, cfg["hide_energy"], n_episodes=16)
        r2, w_hat = episode_split_probe(H_eps, E_eps)

        matrix_r2.setdefault(key, []).append(r2)
        matrix_dev.setdefault(key, []).append(mean_dev)

        # Run causal steering on h=128, energy_input=1 (the matched-input core comparison)
        if cfg["hidden_dim"] == 128 and not cfg["hide_energy"]:
            probs = run_steering_dose_response(model, cfg["hide_energy"], w_hat, alphas)
            steering_curves.setdefault(cfg["mode"], []).append(probs)

    # Print summary table
    print("\n=== Episode-Split Linear Probe R² (Mean ± 95% CI) ===")
    for key in sorted(matrix_r2.keys()):
        mode, e_in, h_dim = key
        m_r2, ci_r2 = mean_ci95(matrix_r2[key])
        m_dev, ci_dev = mean_ci95(matrix_dev[key])
        print(
            f"{mode:11s} | energy_in={e_in} | h={h_dim:3d} (n={len(matrix_r2[key])}) | "
            f"Probe R²: {m_r2:.3f} ± {ci_r2:.3f} | |E - H*|: {m_dev:.3f} ± {ci_dev:.3f}"
        )

    # --- FIGURE 1: 2x2x2 Matrix Bar Chart ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, h_dim in zip(axes, [32, 128]):
        labels = ["Homeo\n(E=1)", "Fixed\n(E=1)", "Homeo\n(E=0)", "Fixed\n(E=0)"]
        keys = [
            ("homeostatic", 1, h_dim),
            ("fixed", 1, h_dim),
            ("homeostatic", 0, h_dim),
            ("fixed", 0, h_dim),
        ]
        means, cis = [], []
        for k in keys:
            if k in matrix_r2:
                m, c = mean_ci95(matrix_r2[k])
            else:
                m, c = 0.0, 0.0
            means.append(m)
            cis.append(c)

        colors = ["#2b8cbe", "#f03b20", "#7bccc4", "#feb24c"]
        ax.bar(labels, means, yerr=cis, capsize=5, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"Hidden Size = {h_dim}")
        ax.set_ylim(-0.1, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Held-Out Episode Probe R² (95% CI)")
    fig.suptitle("Internal Energy Decodability Across Objectives & Inputs")
    plt.tight_layout()
    plt.savefig("fig1_matrix_probes.png", dpi=150)
    print("\nSaved fig1_matrix_probes.png")

    # --- FIGURE 2: Training-Time Emergence (h=128, energy_in=1) ---
    print("Evaluating training-time emergence across checkpoints (h=128, energy_in=1)...")
    ckpt_steps = [0, 100, 200, 300, 400, 500]
    emergence_r2 = {"homeostatic": [], "fixed": []}
    emergence_dev = {"homeostatic": [], "fixed": []}

    for mode in ["homeostatic", "fixed"]:
        pattern = os.path.join(RUNS_DIR, f"{mode}_energy1_h128_s*")
        for rdir in sorted(glob.glob(pattern)):
            r2_traj, dev_traj = [], []
            valid = True
            for step in ckpt_steps:
                cpath = os.path.join(rdir, f"ckpt_{step:04d}.pt")
                if not os.path.exists(cpath):
                    valid = False
                    break
                model = load_model(cpath, 128)
                H_eps, E_eps, m_dev = collect_episodes(model, hide_energy=False, n_episodes=10)
                r2, _ = episode_split_probe(H_eps, E_eps)
                r2_traj.append(r2)
                dev_traj.append(m_dev)
            if valid:
                emergence_r2[mode].append(r2_traj)
                emergence_dev[mode].append(dev_traj)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for mode, color in [("homeostatic", "#2b8cbe"), ("fixed", "#f03b20")]:
        if emergence_r2[mode]:
            m_r2, ci_r2 = mean_ci95(emergence_r2[mode])
            m_dev, ci_dev = mean_ci95(emergence_dev[mode])
            ax1.plot(ckpt_steps, m_r2, "o-", color=color, label=mode.capitalize())
            ax1.fill_between(ckpt_steps, m_r2 - ci_r2, m_r2 + ci_r2, color=color, alpha=0.2)
            ax2.plot(ckpt_steps, m_dev, "o-", color=color, label=mode.capitalize())
            ax2.fill_between(ckpt_steps, m_dev - ci_dev, m_dev + ci_dev, color=color, alpha=0.2)

    ax1.set_title("Representation Emergence")
    ax1.set_xlabel("PPO Updates")
    ax1.set_ylabel("Held-Out Energy Probe R²")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_title("Behavioral Regulation")
    ax2.set_xlabel("PPO Updates")
    ax2.set_ylabel("Mean |Energy - Set Point| (Lower = Better)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig2_emergence.png", dpi=150)
    print("Saved fig2_emergence.png")

    # --- FIGURE 3: Causal Steering Dose-Response ---
    plt.figure(figsize=(6.5, 4.5))
    for mode, color in [("homeostatic", "#2b8cbe"), ("fixed", "#f03b20")]:
        if mode in steering_curves and steering_curves[mode]:
            m_p, ci_p = mean_ci95(steering_curves[mode])
            plt.plot(alphas, m_p, "o-", color=color, label=f"{mode.capitalize()} (n={len(steering_curves[mode])})")
            plt.fill_between(alphas, m_p - ci_p, m_p + ci_p, color=color, alpha=0.2)

    plt.axvline(0.0, color="gray", linestyle="--", alpha=0.6)
    plt.xlabel("Steering Magnitude α Along Energy Axis\n(← Steer Hungrier | Steer More Sated →)")
    plt.ylabel("Foraging Probability (Food at Dist = 3)")
    plt.title("Causal Steering Dose-Response Curve (h=128, E=1)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig3_steering.png", dpi=150)
    print("Saved fig3_steering.png")

    # Save raw arrays for future replotting
    np.savez(
        "eval_results.npz",
        alphas=alphas,
        ckpt_steps=np.array(ckpt_steps),
        homeo_steer=np.array(steering_curves.get("homeostatic", [])),
        fixed_steer=np.array(steering_curves.get("fixed", [])),
    )
    print("Saved raw data to eval_results.npz. All done!")


if __name__ == "__main__":
    main()
