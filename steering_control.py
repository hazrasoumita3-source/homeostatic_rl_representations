import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from evaluate import (
    load_model,
    collect_episodes,
    episode_split_probe,
    run_steering_dose_response,
    mean_ci95,
    RUNS_DIR,
)

if __name__ == "__main__":
    alphas = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    homeo_energy, homeo_random, fixed_energy = [], [], []
    rng = np.random.default_rng(42)

    print("Running controlled steering (Energy Axis vs. Random Direction)...")
    for mode in ["homeostatic", "fixed"]:
        pattern = os.path.join(RUNS_DIR, f"{mode}_energy1_h128_s*")
        for rdir in sorted(glob.glob(pattern)):
            ckpt = os.path.join(rdir, "final.pt")
            if not os.path.exists(ckpt):
                continue

            model = load_model(ckpt, 128)
            H_eps, E_eps, _ = collect_episodes(model, hide_energy=False, n_episodes=16)
            _, w_hat = episode_split_probe(H_eps, E_eps)

            probs = run_steering_dose_response(model, False, w_hat, alphas)
            if mode == "homeostatic":
                homeo_energy.append(probs)
                # Random unit vector of the exact same norm
                r_vec = rng.normal(size=w_hat.shape).astype(np.float32)
                r_vec /= np.linalg.norm(r_vec) + 1e-8
                homeo_random.append(run_steering_dose_response(model, False, r_vec, alphas))
            else:
                fixed_energy.append(probs)

    plt.figure(figsize=(7, 4.5))
    curves = [
        (homeo_energy, "#2b8cbe", "o-", "Homeostatic (Energy Axis)"),
        (homeo_random, "#636363", "s--", "Homeostatic (Random Direction Control)"),
        (fixed_energy, "#f03b20", "^-", "Fixed-Reward (Energy Axis)"),
    ]
    for data, color, fmt, label in curves:
        m, ci = mean_ci95(data)
        plt.plot(alphas, m, fmt, color=color, label=f"{label} (n={len(data)})")
        plt.fill_between(alphas, m - ci, m + ci, color=color, alpha=0.18)

    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
    plt.xlabel("Steering Magnitude α\n(← Steer Hungrier | Steer More Sated →)")
    plt.ylabel("Foraging Probability (Food at Dist = 3)")
    plt.title("Causal Steering Dose-Response with Random-Direction Control")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig3_steering_controlled.png", dpi=150)
    print("Saved fig3_steering_controlled.png")
