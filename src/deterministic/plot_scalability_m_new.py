import matplotlib.pyplot as plt
import numpy as np


# --- Plot: Varying m (n=20), updated FCP data ---

m_values = np.array([20, 40, 60, 80])
sample_counts = np.array([100, 100, 50, 29])

# FCP Data (n=20), updated from the latest evaluation output.
fcp_m_rev_mean = np.array([1.1614, 1.1739, 1.1801, 1.1862])
fcp_m_rev_std = np.array([0.0322, 0.0301, 0.0271, 0.0204])
fcp_m_time_mean = np.array([0.1754, 2.4466, 32.4148, 287.3423])

# BSP baseline data (n=20).
bsp_m_rev_mean = np.ones_like(m_values, dtype=float)
bsp_m_time_mean = np.array([1.3581, 11.8397, 137.6513, 457.8839])

# Colorblind-friendly, publication-style palette.
FCP_COLOR = "#0072B2"
FCP_SHADE_ALPHA = 0.18
BSP_COLOR = "#4D4D4D"
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
    m_values,
    fcp_m_rev_mean,
    marker="o",
    markersize=6,
    linewidth=2.2,
    label="FCP",
    color=FCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[0].fill_between(
    m_values,
    fcp_m_rev_mean - fcp_m_rev_std,
    fcp_m_rev_mean + fcp_m_rev_std,
    alpha=FCP_SHADE_ALPHA,
    color=FCP_COLOR,
    linewidth=0,
)
axes[0].plot(
    m_values,
    bsp_m_rev_mean,
    marker="^",
    markersize=6,
    linewidth=2.0,
    label="BSP",
    linestyle="--",
    color=BSP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)

axes[0].set_xlabel("Number of Customers ($m_{test}$)")
axes[0].set_ylabel("Profit Ratio")
axes[0].set_title("Profit Ratio vs. Number of Customers ($n_{test}=20$)")
axes[0].set_xticks(m_values)
axes[0].set_ylim(0.98, 1.3)
axes[0].set_yticks(np.arange(1.00, 1.31, 0.05))
axes[0].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[0].legend(frameon=False, handlelength=2.2, handletextpad=0.7, numpoints=1)

# Right: Inference Time on a linear scale.
axes[1].plot(
    m_values,
    fcp_m_time_mean,
    marker="o",
    markersize=6,
    linewidth=2.2,
    label="FCP",
    color=FCP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)
axes[1].plot(
    m_values,
    bsp_m_time_mean,
    marker="^",
    markersize=6,
    linewidth=2.0,
    label="BSP",
    linestyle="--",
    color=BSP_COLOR,
    markerfacecolor=MARKER_FACE_COLOR,
    markeredgewidth=MARKER_EDGE_WIDTH,
)

axes[1].set_xlabel("Number of Customers ($m_{test}$)")
axes[1].set_ylabel("Time (s)")
axes[1].set_title("Inference Time vs. Number of Customers ($n_{test}=20$)")
axes[1].set_xticks(m_values)
axes[1].set_ylim(bottom=0)
axes[1].grid(True, linestyle="--", linewidth=0.8, color=GRID_COLOR)
axes[1].legend(frameon=False, handlelength=2.2, handletextpad=0.7, numpoints=1)

fig.tight_layout()
fig.savefig("scalability_m_updated_extracted_colors.pdf", bbox_inches="tight")
fig.savefig("scalability_m_updated_extracted_colors.png", dpi=300, bbox_inches="tight")
plt.close(fig)
