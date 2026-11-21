import glob
import json
import os
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.exceptions import ConvergenceWarning

from env import HomeostaticForagingEnv
from train_ppo import RecurrentActorCritic

warnings.filterwarnings("ignore", category=ConvergenceWarning)
torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = "runs_v2"

CONDITIONS = [
    ("gru_homeostatic_clean", "GRU\nHomeo"),
    ("gru_homeostatic_n15", "GRU\nHomeo\n(Noisy)"),
    ("gru_fixed_clean", "GRU\nFixed"),
    ("gru_fixed_death_clean", "GRU\nFixed+\nDeath"),
    ("lstm_homeostatic_clean", "LSTM\nHomeo"),
    ("lstm_fixed_clean", "LSTM\nFixed"),
]


def load_model(ckpt_path: str, hidden_dim: int, arch: str) -> RecurrentActorCritic:
    model = RecurrentActorCritic(view_size=5, action_dim=5, hidden_dim=hidden_dim, arch=arch).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def step_model(model, obs, state, ablate_vec=None, patch_h=None):
    grid = torch.tensor(obs["grid"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    energy = torch.tensor(obs["energy"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    feat = model.cnn(grid.reshape(1, *grid.shape[2:])).reshape(1, 1, -1)
    x = torch.cat([feat, energy], dim=-1)
    out, next_state = model.rnn(x, state)

    # Project out a unit direction from hidden state (Direction Ablation)
    if ablate_vec is not None:
        u = ablate_vec.view(1, 1, -1)
        out = out - (out * u).sum(dim=-1, keepdim=True) * u
        if model.arch == "lstm":
            h_new = next_state[0] - (next_state[0] * u).sum(dim=-1, keepdim=True) * u
            next_state = (h_new, next_state[1])
        else:
            next_state = next_state - (next_state * u).sum(dim=-1, keepdim=True) * u

    # Replace hidden state with a patched vector (Hungry <-> Sated Patching)
    if patch_h is not None:
        p = patch_h.view(1, 1, -1)
        out = p
        if model.arch == "lstm":
            next_state = (p, next_state[1])
        else:
            next_state = p

    logits = model.actor(out).squeeze(0).squeeze(0)
    action = int(torch.argmax(logits).item())
    h_np = out.squeeze(0).squeeze(0).cpu().numpy()
    return action, next_state, h_np


def collect_episodes(
    model,
    energy_noise: float = 0.0,
    grid_size: int = 13,
    n_food: int = 8,
    n_toxins: int = 4,
    n_episodes: int = 12,
    max_steps: int = 150,
    seed_offset: int = 1000,
    ablate_vec=None,
):
    env = HomeostaticForagingEnv(
        reward_mode="homeostatic",
        grid_size=grid_size,
        n_food=n_food,
        n_toxins=n_toxins,
        max_steps=max_steps,
        energy_noise=energy_noise,
    )
    H_eps, E_eps, devs = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_offset + ep)
        state = model.init_state(1, DEVICE)
        h_ep, e_ep = [], []
        done = False
        while not done:
            action, state, h_np = step_model(model, obs, state, ablate_vec=ablate_vec)
            h_ep.append(h_np)
            e_ep.append(info["energy"])
            devs.append(abs(info["energy"] - env.set_point))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        H_eps.append(np.array(h_ep))
        E_eps.append(np.array(e_ep))

    return H_eps, E_eps, float(np.mean(devs))


def episode_split_probes(H_eps, E_eps):
    mid = len(H_eps) // 2
    H_tr = np.concatenate(H_eps[:mid], axis=0)
    E_tr = np.concatenate(E_eps[:mid], axis=0)
    H_te = np.concatenate(H_eps[mid:], axis=0)
    E_te = np.concatenate(E_eps[mid:], axis=0)

    # 1. Linear probe
    lin = Ridge(alpha=1.0).fit(H_tr, E_tr)
    r2_lin = float(r2_score(E_te, lin.predict(H_te)))
    w = lin.coef_.astype(np.float32)
    w_hat = w / (np.linalg.norm(w) + 1e-8)

    # 2. Small-MLP probe
    mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=200, early_stopping=True, random_state=0)
    mlp.fit(H_tr, E_tr)
    r2_mlp = float(r2_score(E_te, mlp.predict(H_te)))

    return r2_lin, r2_mlp, w_hat


def participation_ratio(H: np.ndarray) -> float:
    Hc = H - H.mean(axis=0, keepdims=True)
    cov = np.cov(Hc, rowvar=False)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    return float((eigvals.sum() ** 2) / (np.sum(eigvals ** 2) + 1e-12))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (norm_x * norm_y + 1e-12))


def build_reference_sequence(length: int = 300):
    """Fixed observation sequence used to compute cross-seed CKA on matched inputs."""
    env = HomeostaticForagingEnv(reward_mode="homeostatic", grid_size=13, max_steps=length)
    obs, _ = env.reset(seed=777)
    seq = []
    for _ in range(length):
        seq.append(obs)
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    return seq


@torch.no_grad()
def run_on_reference(model, seq):
    state = model.init_state(1, DEVICE)
    H = []
    for obs in seq:
        _, state, h_np = step_model(model, obs, state)
        H.append(h_np)
    return np.array(H)


def run_patching_experiment(model, trials: int = 12, max_steps: int = 30):
    """
    Compare foraging probability at distance 3 under:
    - Hungry baseline (E=0.25)
    - Hungry patched with Sated hidden state (should suppress foraging)
    - Sated baseline (E=0.85)
    - Sated patched with Hungry hidden state (should induce foraging)
    """
    env = HomeostaticForagingEnv(reward_mode="homeostatic", grid_size=13, max_steps=max_steps)

    def get_context_h(init_e):
        obs, _ = env.reset(
            seed=3000,
            options={"agent_pos": (6, 6), "init_energy": init_e, "food_positions": [(6, 9)], "toxin_positions": [], "respawn_items": False},
        )
        state = model.init_state(1, DEVICE)
        for _ in range(3):
            _, state, h_np = step_model(model, obs, state)
        return torch.tensor(h_np, dtype=torch.float32, device=DEVICE)

    h_hungry = get_context_h(0.25)
    h_sated = get_context_h(0.85)

    def eval_forage(init_e, patch_h=None):
        ate = 0
        for t in range(trials):
            food_pos = (6, 9) if t % 2 == 0 else (3, 6)
            obs, _ = env.reset(
                seed=4000 + t,
                options={"agent_pos": (6, 6), "init_energy": init_e, "food_positions": [food_pos], "toxin_positions": [], "respawn_items": False},
            )
            state = model.init_state(1, DEVICE)
            for step_i in range(max_steps):
                # Patch for the first 10 steps to test causal override
                p = patch_h if (patch_h is not None and step_i < 10) else None
                action, state, _ = step_model(model, obs, state, patch_h=p)
                obs, _, term, trunc, info = env.step(action)
                if info["ate_food"]:
                    ate += 1
                    break
                if term or trunc:
                    break
        return ate / trials

    return (
        eval_forage(0.25, None),
        eval_forage(0.25, h_sated),
        eval_forage(0.85, None),
        eval_forage(0.85, h_hungry),
    )


def mean_ci95(vals):
    arr = np.array(vals, dtype=float)
    m = np.mean(arr, axis=0)
    ci = 1.96 * np.std(arr, axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(m)
    return m, ci


def main():
    ref_seq = build_reference_sequence(300)
    rng = np.random.default_rng(42)

    results = {
        prefix: {
            "r2_lin": [], "r2_mlp": [], "pr": [], "ref_H": [],
            "dev_base": [], "dev_abl_e": [], "dev_abl_r": [],
            "patch": [], "ood_r2": [], "ood_dev": [],
        }
        for prefix, _ in CONDITIONS
    }

    print("Running Phases 3–5 analysis across all runs_v2 models...")
    for prefix, label in CONDITIONS:
        run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, f"{prefix}_h128_s*")))
        print(f"  Condition {prefix} ({len(run_dirs)} seeds)...")

        for rdir in run_dirs:
            ckpt = os.path.join(rdir, "final.pt")
            cfg_path = os.path.join(rdir, "config.json")
            if not (os.path.exists(ckpt) and os.path.exists(cfg_path)):
                continue

            with open(cfg_path) as f:
                cfg = json.load(f)

            model = load_model(ckpt, cfg["hidden_dim"], cfg["arch"])
            noise = cfg["energy_noise"]

            # --- Phase 3: Probes, PR, Reference H for CKA ---
            H_eps, E_eps, dev_base = collect_episodes(model, energy_noise=noise, n_episodes=12)
            r2_lin, r2_mlp, w_hat = episode_split_probes(H_eps, E_eps)
            H_all = np.concatenate(H_eps, axis=0)
            pr = participation_ratio(H_all)
            ref_H = run_on_reference(model, ref_seq)

            # --- Phase 4: Direction Ablation & Patching ---
            u_e = torch.tensor(w_hat, dtype=torch.float32, device=DEVICE)
            r_vec = rng.normal(size=w_hat.shape).astype(np.float32)
            r_vec /= np.linalg.norm(r_vec) + 1e-8
            u_r = torch.tensor(r_vec, dtype=torch.float32, device=DEVICE)

            _, _, dev_abl_e = collect_episodes(model, energy_noise=noise, n_episodes=10, ablate_vec=u_e)
            _, _, dev_abl_r = collect_episodes(model, energy_noise=noise, n_episodes=10, ablate_vec=u_r)
            patch_res = run_patching_experiment(model, trials=10)

            # --- Phase 5: Zero-Shot OOD (19x19 grid, higher food/toxin count) ---
            H_ood, E_ood, ood_dev = collect_episodes(
                model, energy_noise=noise, grid_size=19, n_food=10, n_toxins=6, n_episodes=10, seed_offset=5000
            )
            ood_r2, _, _ = episode_split_probes(H_ood, E_ood)

            res = results[prefix]
            res["r2_lin"].append(r2_lin)
            res["r2_mlp"].append(r2_mlp)
            res["pr"].append(pr)
            res["ref_H"].append(ref_H)
            res["dev_base"].append(dev_base)
            res["dev_abl_e"].append(dev_abl_e)
            res["dev_abl_r"].append(dev_abl_r)
            res["patch"].append(patch_res)
            res["ood_r2"].append(ood_r2)
            res["ood_dev"].append(ood_dev)

    # Compute pairwise cross-seed CKA per condition
    cka_scores = {}
    for prefix, _ in CONDITIONS:
        Hs = results[prefix]["ref_H"]
        pairs = [linear_cka(Hs[i], Hs[j]) for i in range(len(Hs)) for j in range(i + 1, len(Hs))]
        cka_scores[prefix] = pairs

    # Print concise summary table
    print("\n=== Summary (Mean ± 95% CI) ===")
    for prefix, _ in CONDITIONS:
        r = results[prefix]
        m_lin, c_lin = mean_ci95(r["r2_lin"])
        m_mlp, _ = mean_ci95(r["r2_mlp"])
        m_pr, _ = mean_ci95(r["pr"])
        m_cka, _ = mean_ci95(cka_scores[prefix])
        m_ood, _ = mean_ci95(r["ood_r2"])
        print(
            f"{prefix:24s} | Lin R²: {m_lin:.3f}±{c_lin:.3f} | MLP R²: {m_mlp:.3f} | "
            f"PR: {m_pr:4.1f} | CKA: {m_cka:.3f} | OOD R²: {m_ood:.3f}"
        )

    labels = [lbl for _, lbl in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    # --- FIGURE 4: Phase 3 (Probes, Participation Ratio, Cross-Seed CKA) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    w_bar = 0.35

    lin_m, lin_c = zip(*[mean_ci95(results[p]["r2_lin"]) for p, _ in CONDITIONS])
    mlp_m, mlp_c = zip(*[mean_ci95(results[p]["r2_mlp"]) for p, _ in CONDITIONS])
    axes[0].bar(x - w_bar / 2, lin_m, w_bar, yerr=lin_c, capsize=3, label="Linear Probe", color="#2b8cbe")
    axes[0].bar(x + w_bar / 2, mlp_m, w_bar, yerr=mlp_c, capsize=3, label="MLP Probe", color="#a6bddb")
    axes[0].set_title("Linear vs. MLP Probe R²")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    pr_m, pr_c = zip(*[mean_ci95(results[p]["pr"]) for p, _ in CONDITIONS])
    axes[1].bar(x, pr_m, yerr=pr_c, capsize=4, color="#31a354", alpha=0.85)
    axes[1].set_title("Effective Dimensionality (Participation Ratio)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    cka_m, cka_c = zip(*[mean_ci95(cka_scores[p]) for p, _ in CONDITIONS])
    axes[2].bar(x, cka_m, yerr=cka_c, capsize=4, color="#756bb1", alpha=0.85)
    axes[2].set_title("Cross-Seed Representational Similarity (CKA)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig4_phase3_repr.png", dpi=150)
    print("\nSaved fig4_phase3_repr.png")

    # --- FIGURE 5: Phase 4 (Direction Ablation & Hungry <-> Sated Patching) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    w_bar = 0.25

    b_m, b_c = zip(*[mean_ci95(results[p]["dev_base"]) for p, _ in CONDITIONS])
    e_m, e_c = zip(*[mean_ci95(results[p]["dev_abl_e"]) for p, _ in CONDITIONS])
    r_m, r_c = zip(*[mean_ci95(results[p]["dev_abl_r"]) for p, _ in CONDITIONS])

    ax1.bar(x - w_bar, b_m, w_bar, yerr=b_c, capsize=3, label="Baseline", color="#969696")
    ax1.bar(x, e_m, w_bar, yerr=e_c, capsize=3, label="Ablate Energy Axis", color="#de2d26")
    ax1.bar(x + w_bar, r_m, w_bar, yerr=r_c, capsize=3, label="Ablate Random Axis", color="#3182bd")
    ax1.set_title("Direction Ablation: Mean |E - H*| (Lower = Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Patching comparison on GRU Homeo vs GRU Fixed
    patch_labels = ["Hungry\n(Base)", "Hungry +\nSated Patch", "Sated\n(Base)", "Sated +\nHungry Patch"]
    px = np.arange(4)
    hm, hc = mean_ci95(results["gru_homeostatic_clean"]["patch"])
    fm, fc = mean_ci95(results["gru_fixed_clean"]["patch"])
    ax2.bar(px - 0.18, hm, 0.36, yerr=hc, capsize=4, label="GRU Homeostatic", color="#2b8cbe")
    ax2.bar(px + 0.18, fm, 0.36, yerr=fc, capsize=4, label="GRU Fixed-Reward", color="#f03b20")
    ax2.set_title("Hidden-State Patching (Foraging Prob at Dist=3)")
    ax2.set_xticks(px)
    ax2.set_xticklabels(patch_labels, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig5_phase4_causal.png", dpi=150)
    print("Saved fig5_phase4_causal.png")

    # --- FIGURE 6: Phase 5 (In-Distribution 13x13 vs. OOD 19x19) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    w_bar = 0.35

    ood_r2_m, ood_r2_c = zip(*[mean_ci95(results[p]["ood_r2"]) for p, _ in CONDITIONS])
    ax1.bar(x - w_bar / 2, lin_m, w_bar, yerr=lin_c, capsize=3, label="In-Dist (13x13)", color="#2b8cbe")
    ax1.bar(x + w_bar / 2, ood_r2_m, w_bar, yerr=ood_r2_c, capsize=3, label="OOD (19x19)", color="#7bccc4")
    ax1.set_title("Zero-Shot OOD Transfer: Linear Probe R²")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    ood_dev_m, ood_dev_c = zip(*[mean_ci95(results[p]["ood_dev"]) for p, _ in CONDITIONS])
    ax2.bar(x - w_bar / 2, b_m, w_bar, yerr=b_c, capsize=3, label="In-Dist (13x13)", color="#2b8cbe")
    ax2.bar(x + w_bar / 2, ood_dev_m, w_bar, yerr=ood_dev_c, capsize=3, label="OOD (19x19)", color="#7bccc4")
    ax2.set_title("Zero-Shot OOD Transfer: Mean |E - H*|")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig6_phase5_ood.png", dpi=150)
    print("Saved fig6_phase5_ood.png")
    print("All Phases 3–5 analyses complete!")


if __name__ == "__main__":
    main()
