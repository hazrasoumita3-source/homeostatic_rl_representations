import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from env import HomeostaticForagingEnv
from train_ppo import RecurrentActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: str, hidden_dim: int = 128) -> RecurrentActorCritic:
    model = RecurrentActorCritic(view_size=5, action_dim=5, hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def step_model(model, obs, h, ablate_idx=None, ablate_vals=None):
    grid = torch.tensor(obs["grid"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
    energy = torch.tensor(obs["energy"], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

    # Forward through CNN + GRU
    feat = model.cnn(grid.reshape(1, *grid.shape[2:])).reshape(1, 1, -1)
    x = torch.cat([feat, energy], dim=-1)
    out, h_next = model.gru(x, h)

    # Optional unit ablation (clamping specific GRU units)
    if ablate_idx is not None and ablate_vals is not None:
        h_next[:, :, ablate_idx] = ablate_vals
        out[:, :, ablate_idx] = ablate_vals

    logits = model.actor(out).squeeze(0).squeeze(0)
    action = int(torch.argmax(logits).item())
    return action, h_next


# ---------------------------------------------------------
# 1. BEHAVIORAL PROBE: Foraging Probability vs. Energy & Distance
# ---------------------------------------------------------
def probe_foraging(model, energies, distances, trials=10, max_steps=40):
    env = HomeostaticForagingEnv(reward_mode="homeostatic", grid_size=11, max_steps=max_steps)
    results = np.zeros((len(energies), len(distances)))

    for i, e0 in enumerate(energies):
        for j, dist in enumerate(distances):
            successes = 0
            for t in range(trials):
                # Place agent at center (5, 5) and one food item at Manhattan distance `dist`
                food_pos = (5, min(10, 5 + dist)) if t % 2 == 0 else (max(0, 5 - dist), 5)
                obs, _ = env.reset(
                    seed=100 + t,
                    options={
                        "agent_pos": (5, 5),
                        "init_energy": float(e0),
                        "food_positions": [food_pos],
                        "toxin_positions": [],
                        "respawn_items": False,
                    },
                )
                h = torch.zeros((1, 1, model.hidden_dim), device=DEVICE)
                for _ in range(max_steps):
                    action, h = step_model(model, obs, h)
                    obs, _, term, trunc, info = env.step(action)
                    if info["ate_food"]:
                        successes += 1
                        break
                    if term or trunc:
                        break
            results[i, j] = successes / trials
    return results


# ---------------------------------------------------------
# 2. HIDDEN REPRESENTATION LOGGING & PCA
# ---------------------------------------------------------
def collect_hidden_states(model, n_episodes=30):
    env = HomeostaticForagingEnv(reward_mode="homeostatic")
    H_list, E_list, D_food_list = [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=500 + ep)
        h = torch.zeros((1, 1, model.hidden_dim), device=DEVICE)
        done = False
        while not done:
            action, h = step_model(model, obs, h)
            H_list.append(h.squeeze(0).squeeze(0).cpu().numpy())
            E_list.append(info["energy"])
            D_food_list.append(info["dist_to_food"])
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc

    return np.array(H_list), np.array(E_list), np.array(D_food_list)


# ---------------------------------------------------------
# 3. TARGETED VS. RANDOM UNIT ABLATION
# ---------------------------------------------------------
def evaluate_ablation(model, H_states, energies, k_values=(0, 4, 8, 16, 32), n_episodes=20):
    # Rank GRU units by absolute Pearson correlation with internal energy
    corrs = np.array([
        abs(np.corrcoef(H_states[:, i], energies)[0, 1])
        if np.std(H_states[:, i]) > 1e-6 else 0.0
        for i in range(H_states.shape[1])
    ])
    ranked_units = np.argsort(corrs)[::-1].copy()
    mean_act = torch.tensor(H_states.mean(axis=0), dtype=torch.float32, device=DEVICE)

    def run_eval(ablate_idx):
        env = HomeostaticForagingEnv(reward_mode="homeostatic")
        abs_devs = []
        ablate_vals = mean_act[ablate_idx] if len(ablate_idx) > 0 else None

        for ep in range(n_episodes):
            obs, info = env.reset(seed=900 + ep)
            h = torch.zeros((1, 1, model.hidden_dim), device=DEVICE)
            done = False
            while not done:
                action, h = step_model(
                    model, obs, h,
                    ablate_idx=ablate_idx if len(ablate_idx) > 0 else None,
                    ablate_vals=ablate_vals,
                )
                obs, _, term, trunc, info = env.step(action)
                abs_devs.append(abs(info["energy"] - env.set_point))
                done = term or trunc
        return float(np.mean(abs_devs))

    targeted_devs, random_devs = [], []
    rng = np.random.default_rng(42)

    for k in k_values:
        t_idx = ranked_units[:k]
        r_idx = rng.choice(H_states.shape[1], size=k, replace=False) if k > 0 else np.array([], dtype=int)
        targeted_devs.append(run_eval(t_idx))
        random_devs.append(run_eval(r_idx))

    return targeted_devs, random_devs, corrs


if __name__ == "__main__":
    print("Loading trained models...")
    m_homeo = load_model("model_homeostatic.pt")
    m_fixed = load_model("model_fixed.pt")

    # --- 1. Behavioral Foraging Comparison ---
    energies = [0.2, 0.4, 0.6, 0.75, 0.9]
    distances = [1, 2, 3, 4, 5]
    print("Running behavioral foraging probes...")
    res_homeo = probe_foraging(m_homeo, energies, distances)
    res_fixed = probe_foraging(m_fixed, energies, distances)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, res, title in zip(axes, [res_homeo, res_fixed], ["Homeostatic Agent", "Fixed-Reward Agent"]):
        im = ax.imshow(res, vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(distances)
        ax.set_yticks(range(len(energies)))
        ax.set_yticklabels(energies)
        ax.set_xlabel("Manhattan Distance to Food")
        ax.set_title(title)
    axes[0].set_ylabel("Initial Energy (Set Point = 0.7)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Foraging Probability")
    plt.savefig("foraging_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved foraging_comparison.png")

    # --- 2. PCA & Linear Probe ---
    print("Collecting GRU hidden states & running PCA...")
    H_homeo, E_homeo, _ = collect_hidden_states(m_homeo)
    H_fixed, E_fixed, _ = collect_hidden_states(m_fixed)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, H, E, title in zip(
        axes, [H_homeo, H_fixed], [E_homeo, E_fixed], ["Homeostatic GRU", "Fixed-Reward GRU"]
    ):
        pca = PCA(n_components=2)
        Z = pca.fit_transform(H)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=E, cmap="coolwarm", s=4, alpha=0.6, vmin=0, vmax=1)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

        # Linear probe R^2 for decoding energy from hidden state
        probe = Ridge(alpha=1.0).fit(H, E)
        r2 = r2_score(E, probe.predict(H))
        ax.set_title(f"{title} (Energy Probe R² = {r2:.3f})")
        print(f"{title} | Linear Energy Decoding R²: {r2:.3f}")

    fig.colorbar(sc, ax=axes.ravel().tolist(), label="Internal Energy")
    plt.savefig("pca_representations.png", dpi=150, bbox_inches="tight")
    print("Saved pca_representations.png")

    # --- 3. Unit Ablation (Homeostatic Agent) ---
    print("Running targeted vs. random unit ablation...")
    k_vals = [0, 4, 8, 16, 32]
    targ_devs, rand_devs, corrs = evaluate_ablation(m_homeo, H_homeo, E_homeo, k_values=k_vals)

    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, targ_devs, "o-", label="Top-k Energy-Selective Units")
    plt.plot(k_vals, rand_devs, "s--", label="k Random Units")
    plt.xlabel("Number of Ablated GRU Units (k)")
    plt.ylabel("Mean |Energy - Set Point| (Lower is Better)")
    plt.title("Causal Impact of Ablating Energy-Selective Units")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("ablation_results.png", dpi=150, bbox_inches="tight")
    print("Saved ablation_results.png")
    print("Done! Open the three .png files in your folder to view the results.")
