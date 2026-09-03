import matplotlib.pyplot as plt
import numpy as np


# --- Plot: Varying n (m=10), updated FCP / PCP data ---

n_values = np.array([20, 30, 40, 50])

# FCP Data (m=10), updated from the latest evaluation output.
fcp_n_rev_mean = np.array([1.1362, 1.1174, 1.0756, 1.0204])
fcp_n_rev_std = np.array([0.0404, 0.0387, 0.0474, 0.0489])
fcp_n_time_mean = np.array([0.0209, 0.0245, 0.0282, 0.0195])

# PCP Data (m=10), updated from the latest PCP-cp evaluation output.
pcp_n_rev_mean = np.array([1.1618, 1.1819, 1.1956, 1.1948])
pcp_n_rev_std = np.array([0.0419, 0.0367, 0.0367, 0.0346])
pcp_n_time_mean = np.array([0.6719, 2.0597, 6.0393, 27.2775])

# BSP baseline data (m=10), recomputed from dataset running_time.
bsp_n_rev_mean = np.ones_like(n_values, dtype=float)
bsp_n_time_mean = np.array([0.2025, 0.3970, 0.6144, 0.8370])

# Colorblind-friendly, publication-style palette.
FCP_COLOR = "#0072B2"
PCP_COLOR = "#D55E00"
BSP_COLOR = "#4D4D4D"
SHADE_ALPHA = 0.16
GRID_COLOR = "#D9D9D9"
MARKER_FACE_COLOR = "white"
MARKER_EDGE_WIDTH = 1.5

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Profit Ratio with standard deviation shading.
axes[0].plot(
    n_values,
    fcp_n_rev_mean,
    marker="o",
    markersize=6,
    linewidth=2.2,
    label="FCP",
    color=FCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[0].fill_between(
    n_values,
    fcp_n_rev_mean - fcp_n_rev_std,
    fcp_n_rev_mean + fcp_n_rev_std,
    alpha=SHADE_ALPHA,
    color=FCP_COLOR,
    linewidth=0,
)
axes[0].plot(
    n_values,
    pcp_n_rev_mean,
    marker="s",
    markersize=5.8,
    linewidth=2.2,
    label="PCP",
    color=PCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[0].fill_between(
    n_values,
    pcp_n_rev_mean - pcp_n_rev_std,
    pcp_n_rev_mean + pcp_n_rev_std,
    alpha=SHADE_ALPHA,
    color=PCP_COLOR,
    linewidth=0,
)
axes[0].plot(
    n_values,
    bsp_n_rev_mean,
    marker="^",
    markersize=6,
    linewidth=2.0,
    label="BSP",
    linestyle="--",
    color=BSP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)

axes[0].set_xlabel("Number of Products ($n_{test}$)")
axes[0].set_ylabel("Profit Ratio")
axes[0].set_title("Profit Ratio vs. Number of Products ($m_{test}=10$)")
axes[0].set_xticks(n_values)
axes[0].set_ylim(0.98, 1.3)
axes[0].set_yticks(np.arange(1.00, 1.31, 0.05))
axes[0].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[0].legend(frameon=False, handlelength=2.2, handletextpad=0.7, numpoints=1)

# Right: Inference Time on a linear scale.
axes[1].plot(
    n_values,
    fcp_n_time_mean,
    marker="o",
    markersize=6,
    linewidth=2.2,
    label="FCP",
    color=FCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[1].plot(
    n_values,
    pcp_n_time_mean,
    marker="s",
    markersize=5.8,
    linewidth=2.2,
    label="PCP",
    color=PCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[1].plot(
    n_values,
    bsp_n_time_mean,
    marker="^",
    markersize=6,
    linewidth=2.0,
    label="BSP",
    linestyle="--",
    color=BSP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)

axes[1].set_xlabel("Number of Products ($n_{test}$)")
axes[1].set_ylabel("Time (s)")
axes[1].set_title("Inference Time vs. Number of Products ($m_{test}=10$)")
axes[1].set_xticks(n_values)
axes[1].set_ylim(bottom=0)
axes[1].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[1].legend(frameon=False, handlelength=2.2, handletextpad=0.7, numpoints=1)

fig.tight_layout()
fig.savefig("scalability_n_updated.pdf", bbox_inches="tight")
fig.savefig("scalability_n_updated.png", dpi=300, bbox_inches="tight")
plt.close(fig)
