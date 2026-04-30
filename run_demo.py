import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from event_simulator import simulate
from observations import extract_observations
from hmm import HMM

# ── HMM parameters ────────────────────────────────────────────────────────────
# state order: 0=LEFT, 1=REST, 2=RIGHT
# obs order:   0=LEFT_FLOW, 1=NO_FLOW, 2=RIGHT_FLOW

A = np.array([
    [0.90, 0.08, 0.02],
    [0.05, 0.90, 0.05],
    [0.02, 0.08, 0.90],
])
B = np.array([
    [0.75, 0.20, 0.05],
    [0.15, 0.70, 0.15],
    [0.05, 0.20, 0.75],
])
pi = np.array([1/3, 1/3, 1/3])

# ── data ──────────────────────────────────────────────────────────────────────
modes = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20
events, gt = simulate(modes)
obs = extract_observations(events, num_bins=len(modes))

# ── inference ─────────────────────────────────────────────────────────────────
hmm        = HMM(A, B, pi)
_, ll, filtered = hmm.forward(obs)

hmm_u      = HMM(np.full((3, 3), 1/3), B, pi)
_, ll_u, _ = hmm_u.forward(obs)

print(f"structured HMM  log-likelihood : {ll:.1f}")
print(f"uniform HMM     log-likelihood : {ll_u:.1f}")
print(f"delta                          : {ll - ll_u:.1f} nats  (structured is better)")

# ── plot ──────────────────────────────────────────────────────────────────────
MODE_INT  = {"LEFT": 0, "REST": 1, "RIGHT": 2}
STATE_COL = ["#e05c5c", "#888888", "#5c8fe0"]  # red / grey / blue
OBS_COL   = ["#e05c5c", "#888888", "#5c8fe0"]
STATE_NAMES = ["LEFT", "REST", "RIGHT"]

T = len(modes)
t = np.arange(T)

gt_int  = np.array([MODE_INT[m] for m in gt])
obs_int = obs.astype(float)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0.08)

# panel 1 — ground truth
ax = axes[0]
for i, (col, name) in enumerate(zip(STATE_COL, STATE_NAMES)):
    mask = gt_int == i
    ax.fill_between(t, 0, 1, where=mask, color=col, alpha=0.85, linewidth=0)
ax.set_yticks([])
ax.set_ylabel("ground truth", fontsize=9)
ax.set_ylim(0, 1)

# panel 2 — discretised observations
ax = axes[1]
for i, col in enumerate(OBS_COL):
    mask = obs_int == i
    ax.fill_between(t, 0, 1, where=mask, color=col, alpha=0.85, linewidth=0)
ax.set_yticks([])
ax.set_ylabel("observation", fontsize=9)
ax.set_ylim(0, 1)

# panel 3 — filtered posterior
ax = axes[2]
for i, (col, name) in enumerate(zip(STATE_COL, STATE_NAMES)):
    ax.plot(t, filtered[:, i], color=col, lw=1.5, label=name)
ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("P(state | obs)", fontsize=9)
ax.set_xlabel("time bin (10 ms)", fontsize=9)
ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
ax.axhline(0.5, color="k", lw=0.5, ls="--", alpha=0.3)

# shared legend for top two panels
patches = [mpatches.Patch(color=c, label=n) for c, n in zip(STATE_COL, STATE_NAMES)]
axes[0].legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.7)

fig.suptitle(
    f"HMM forward algorithm on synthetic DVS events\n"
    f"structured Δlog-likelihood vs uniform: {ll - ll_u:+.1f} nats",
    fontsize=10, y=0.98
)

plt.savefig("plots/demo.png", dpi=150, bbox_inches="tight")
print("saved plots/demo.png")
plt.show()
