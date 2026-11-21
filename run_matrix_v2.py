import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# 6 conditions x 8 seeds = 48 runs (~1.5 - 2 hours on 4 parallel workers)
CONDITIONS = [
    ("gru", "homeostatic", 0.0),    # Core homeostatic
    ("gru", "homeostatic", 0.15),   # Noisy interoception control
    ("gru", "fixed", 0.0),          # Fixed reward baseline
    ("gru", "fixed_death", 0.0),    # Fixed reward + death penalty control
    ("lstm", "homeostatic", 0.0),   # Architecture control (LSTM)
    ("lstm", "fixed", 0.0),         # Architecture control (LSTM)
]
SEEDS = list(range(8))
HIDDEN_DIM = 128
UPDATES = 1000
SAVE_EVERY = 200
MAX_WORKERS = 4
OUT_DIR = "runs_v2"


def run_job(job):
    arch, mode, noise, seed = job
    noise_tag = f"n{int(noise * 100)}" if noise > 0 else "clean"
    run_name = f"{arch}_{mode}_{noise_tag}_h{HIDDEN_DIM}_s{seed}"
    final_path = os.path.join(OUT_DIR, run_name, "final.pt")

    if os.path.exists(final_path):
        return f"SKIPPED: {run_name}"

    cmd = [
        sys.executable, "train_ppo.py",
        "--arch", arch,
        "--mode", mode,
        "--energy_noise", str(noise),
        "--hidden_dim", str(HIDDEN_DIM),
        "--seed", str(seed),
        "--updates", str(UPDATES),
        "--save_every", str(SAVE_EVERY),
        "--out_dir", OUT_DIR,
        "--threads", "1",
        "--quiet", "--ent_coef", "0.02",
    ]
    subprocess.run(cmd, check=True)
    return f"COMPLETED: {run_name}"


if __name__ == "__main__":
    jobs = [(arch, mode, noise, s) for (arch, mode, noise) in CONDITIONS for s in SEEDS]
    print(f"Launching {len(jobs)} v2 runs with {MAX_WORKERS} parallel workers...")
    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, j) for j in jobs]
        for fut in as_completed(futures):
            done += 1
            print(f"[{done:2d}/{len(jobs)}] {fut.result()}", flush=True)
    print("All v2 runs complete!")
