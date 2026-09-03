"""
Standalone replot of LP-MILP verification paths from existing JSON.
No dependency on verify_lp_milp_improvement (avoids torch_geometric etc).
Generates: LP_MILP_verification_paths.png (absolute), LP_MILP_verification_paths_ratio.png (ratio).
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(package_root, "artifacts", "appendix_e_final")
json_path = os.path.join(output_dir, "LP_MILP_verification_results.json")

if not os.path.exists(json_path):
    print(f"Not found: {json_path}. Run verify_lp_milp_improvement.py first.")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    all_results = json.load(f)

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

by_dataset = {}
for res in all_results:
    name = res["dataset_name"]
    if name not in by_dataset:
        by_dataset[name] = []
    by_dataset[name].append(res)

dataset_order = [
    "m10_n10_sample_100", "m20_n10_sample_100", "m30_n10_sample_100",
    "test_BSP_m10n15", "test_BSP_m10n20", "test_BSP_m10n25",
]


def plot_paths(use_absolute=True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, dataset_name in enumerate(dataset_order):
        if dataset_name not in by_dataset:
            continue
        ax = axes[idx]
        samples = by_dataset[dataset_name]
        max_len = max(len(s["lp_path"]) for s in samples)
        opt_rev_default = 1.0

        lp_padded_list = []
        milp_padded_list = []
        for sample in samples:
            lp_path = sample["lp_path"]
            milp_path = sample["milp_path"]
            lp_padded = lp_path + [lp_path[-1]] * (max_len - len(lp_path)) if len(lp_path) < max_len else lp_path
            milp_padded = milp_path + [milp_path[-1]] * (max_len - len(milp_path)) if len(milp_path) < max_len else milp_path
            if use_absolute:
                opt_rev = sample.get("opt_rev", opt_rev_default)
                lp_padded_list.append([v * opt_rev for v in lp_padded])
                milp_padded_list.append([v * opt_rev for v in milp_padded])
            else:
                lp_padded_list.append(lp_padded)
                milp_padded_list.append(milp_padded)

        for i in range(len(samples)):
            steps = list(range(max_len))
            ax.plot(steps, lp_padded_list[i], color="C0", alpha=0.2, linewidth=1)
            ax.plot(steps, milp_padded_list[i], color="C1", alpha=0.2, linewidth=1, linestyle="--")

        lp_avg = [np.mean([lp_padded_list[i][step] for i in range(len(samples))]) for step in range(max_len)]
        milp_avg = [np.mean([milp_padded_list[i][step] for i in range(len(samples))]) for step in range(max_len)]
        lp_std = [np.std([lp_padded_list[i][step] for i in range(len(samples))]) for step in range(max_len)]
        milp_std = [np.std([milp_padded_list[i][step] for i in range(len(samples))]) for step in range(max_len)]

        steps_avg = list(range(max_len))
        ax.plot(steps_avg, lp_avg, "o-", color="C0", linewidth=3, markersize=6, label="LP (avg)")
        ax.plot(steps_avg, milp_avg, "s--", color="C1", linewidth=3, markersize=6, label="MILP (avg)")
        ax.fill_between(steps_avg, np.array(lp_avg) - np.array(lp_std), np.array(lp_avg) + np.array(lp_std), color="C0", alpha=0.1)
        ax.fill_between(steps_avg, np.array(milp_avg) - np.array(milp_std), np.array(milp_avg) + np.array(milp_std), color="C1", alpha=0.1)
        ax.set_xlabel("Improvement Step")
        ax.set_ylabel("Profit (absolute)" if use_absolute else "Profit Ratio")
        ax.set_title(f"{dataset_name} (n={len(samples)} samples, padded)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = "LP_MILP_verification_paths.png" if use_absolute else "LP_MILP_verification_paths_ratio.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


plot_paths(use_absolute=True)
plot_paths(use_absolute=False)
print("Done.")
