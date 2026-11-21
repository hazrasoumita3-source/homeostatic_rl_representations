import itertools
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

MODES = ["homeostatic", "fixed"]
HIDE_ENERGY = [False, True]
HIDDEN_DIMS = [32, 128]
SEEDS = list(range(8))

UPDATES = 500        # 500 * 8 * 128 = 512,000 steps per run (~2 hrs total for 64 runs)
SAVE_EVERY = 100     # Saves ckpt_0000, ckpt_0100, ..., ckpt_0500
MAX_WORKERS = 4      # 4 parallel single-threaded jobs on your 18-thread CPU
OUT_DIR = "runs"


def run_job(job):
    mode, hide_energy, hidden_dim, seed = job
    energy_tag = 0 if hide_energy else 1
    run_name = f"{mode}_energy{energy_tag}_h{hidden_dim}_s{seed}"
    final_path = os.path.join(OUT_DIR, run_name, "final.pt")

    if os.path.exists(final_path):
        return f"SKIPPED (already done): {run_name}"

    cmd = [
        sys.executable, "train_ppo.py",
        "--mode", mode,
        "--hidden_dim", str(hidden_dim),
        "--seed", str(seed),
        "--updates", str(UPDATES),
        "--save_every", str(SAVE_EVERY),
        "--out_dir", OUT_DIR,
        "--threads", "1",
        "--quiet",
    ]
    if hide_energy:
        cmd.append("--hide_energy")

    subprocess.run(cmd, check=True)
    return f"COMPLETED: {run_name}"


if __name__ == "__main__":
    jobs = list(itertools.product(MODES, HIDE_ENERGY, HIDDEN_DIMS, SEEDS))
    print(f"Launching {len(jobs)} runs with {MAX_WORKERS} parallel workers...")

    done = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for fut in as_completed(futures):
            done += 1
            print(f"[{done:2d}/{len(jobs)}] {fut.result()}", flush=True)

    print("All 64 runs complete!")
