import matplotlib.pyplot as plt
import numpy as np


# --- Plot: Iterative improvement over n (m=10) ---

n_values = np.array([20, 30, 40, 50])

# Baseline data, matching plot_scalability_n_new.py.
fcp_rev_mean = np.array([1.1362, 1.1174, 1.0756, 1.0204])
fcp_rev_std = np.array([0.0404, 0.0387, 0.0474, 0.0489])
fcp_time_mean = np.array([0.0209, 0.0245, 0.0282, 0.0195])

pcp_rev_mean = np.array([1.1618, 1.1819, 1.1956, 1.1948])
pcp_rev_std = np.array([0.0419, 0.0367, 0.0367, 0.0346])
pcp_time_mean = np.array([0.6719, 2.0597, 6.0393, 27.2775])

bsp_rev_mean = np.ones_like(n_values, dtype=float)
bsp_time_mean = np.array([0.2025, 0.3970, 0.6144, 0.8370])

# Iterative-improvement data.
fcp_i_rev_mean = np.array([1.1301, 1.1673, 1.1891, 1.1889])
fcp_i_rev_std = np.array([0.0403, 0.0356, 0.0361, 0.0335])
fcp_i_time_mean = np.array([0.0179, 0.0180, 0.0191, 0.0184])

pcp_i_rev_mean = np.array([1.1315, 1.1727, 1.2020, 1.2167])
pcp_i_rev_std = np.array([0.0405, 0.0370, 0.0371, 0.0362])
pcp_i_time_mean = np.array([0.1384, 0.3160, 0.9711, 2.1535])

# Colorblind-friendly, publication-style palette.
FCP_COLOR = "#0072B2"
PCP_COLOR = "#D55E00"
BSP_COLOR = "#4D4D4D"
SHADE_ALPHA = 0.12
GRID_COLOR = "#D9D9D9"
DASHED_STYLE = (0, (6.0, 2.4))
SOLID_STYLE = "-"
MARKER_FACE_COLOR = "white"
MARKER_EDGE_WIDTH = 1.5

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Profit Ratio.
series_rev = [
    ("FCP", fcp_rev_mean, fcp_rev_std, FCP_COLOR, "o", SOLID_STYLE),
    ("FCP-I", fcp_i_rev_mean, fcp_i_rev_std, FCP_COLOR, "o", DASHED_STYLE),
    ("PCP", pcp_rev_mean, pcp_rev_std, PCP_COLOR, "s", SOLID_STYLE),
    ("PCP-I", pcp_i_rev_mean, pcp_i_rev_std, PCP_COLOR, "s", DASHED_STYLE),
]

for label, mean, std, color, marker, linestyle in series_rev:
    axes[0].plot(
        n_values,
        mean,
        marker=marker,
        markersize=6,
        linewidth=2.2,
        linestyle=linestyle,
        label=label,
        color=color,
        markerfacecolor=MARKER_FACE_COLOR,
        markeredgewidth=MARKER_EDGE_WIDTH,
        dash_capstyle="butt",
    )
    axes[0].fill_between(
        n_values,
        mean - std,
        mean + std,
        alpha=SHADE_ALPHA,
        color=color,
        linewidth=0,
    )

axes[0].plot(
    n_values,
    bsp_rev_mean,
    marker="^",
    markersize=6,
    linewidth=2.0,
    linestyle=DASHED_STYLE,
    label="BSP",
    color=BSP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
    dash_capstyle="butt",
)

axes[0].set_xlabel("Number of Products ($n_{test}$)")
axes[0].set_ylabel("Profit Ratio")
axes[0].set_title("Profit Ratio vs. Number of Products ($m_{test}=10$)")
axes[0].set_xticks(n_values)
axes[0].set_ylim(0.98, 1.3)
axes[0].set_yticks(np.arange(1.00, 1.31, 0.05))
axes[0].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[0].legend(frameon=False, ncol=2, handlelength=2.2, handletextpad=0.7, numpoints=1)

# Right: Inference Time on a linear scale.
series_time = [
    ("FCP", fcp_time_mean, FCP_COLOR, "o", SOLID_STYLE),
    ("FCP-I", fcp_i_time_mean, FCP_COLOR, "o", DASHED_STYLE),
    ("PCP", pcp_time_mean, PCP_COLOR, "s", SOLID_STYLE),
    ("PCP-I", pcp_i_time_mean, PCP_COLOR, "s", DASHED_STYLE),
    ("BSP", bsp_time_mean, BSP_COLOR, "^", DASHED_STYLE),
]

for label, mean, color, marker, linestyle in series_time:
    axes[1].plot(
        n_values,
        mean,
        marker=marker,
        markersize=6,
        linewidth=2.2 if label != "BSP" else 2.0,
        linestyle=linestyle,
        label=label,
        color=color,
        markerfacecolor=MARKER_FACE_COLOR,
        markeredgewidth=MARKER_EDGE_WIDTH,
        dash_capstyle="butt",
    )

axes[1].set_xlabel("Number of Products ($n_{test}$)")
axes[1].set_ylabel("Time (s)")
axes[1].set_title("Inference Time vs. Number of Products ($m_{test}=10$)")
axes[1].set_xticks(n_values)
axes[1].set_ylim(bottom=0)
axes[1].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[1].legend(frameon=False, ncol=2, handlelength=2.2, handletextpad=0.7, numpoints=1)

fig.tight_layout()
fig.savefig("iterative_self_improvement_updated_exact_colors.pdf", bbox_inches="tight")
fig.savefig("iterative_self_improvement_updated_exact_colors.png", dpi=300, bbox_inches="tight")
plt.close(fig)
