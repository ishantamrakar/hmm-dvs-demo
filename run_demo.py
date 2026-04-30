import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from event_simulator import simulate, random_mode_sequence
from observations import extract_observations
from hmm import HMM

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
pi  = np.array([1/3, 1/3, 1/3])
BIN_MS = 10
MAX_EVENTS_PER_BIN = 80  # cap for JSON size

# ── simulate 30 s ─────────────────────────────────────────────────────────────
modes  = random_mode_sequence(n_bins=3000, seed=42)
events, gt = simulate(modes, bin_ms=BIN_MS)
obs    = extract_observations(events, num_bins=len(modes), bin_ms=BIN_MS)

# ── inference ─────────────────────────────────────────────────────────────────
_, ll,   filtered = HMM(A, B, pi).forward(obs)
_, ll_u, _        = HMM(np.full((3,3), 1/3), B, pi).forward(obs)

print(f"structured  log-likelihood : {ll:.1f}")
print(f"uniform     log-likelihood : {ll_u:.1f}")
print(f"delta                      : {ll - ll_u:.1f} nats")

# ── build per-bin event lists for animation ───────────────────────────────────
bin_us = BIN_MS * 1000
T = len(modes)
bin_events = [[] for _ in range(T)]
for t_us, x, y, pol in events:
    k = t_us // bin_us
    if k < T:
        bin_events[k].append((x, y, pol))

# subsample each bin so JSON stays small
rng = np.random.default_rng(0)
for k in range(T):
    ev = bin_events[k]
    if len(ev) > MAX_EVENTS_PER_BIN:
        idx = rng.choice(len(ev), MAX_EVENTS_PER_BIN, replace=False)
        bin_events[k] = [ev[i] for i in idx]

# reconstruct bar position per bin (re-simulate positions only)
_VX = {"LEFT": -2, "REST": 0, "RIGHT": 2}
bar_positions = []
x = 64 // 2
for mode in modes:
    bar_positions.append(x)
    x = (x + _VX[mode]) % (64 - 5)

# ── export JSON ───────────────────────────────────────────────────────────────
payload = {
    "bin_ms"    : BIN_MS,
    "frame_size": 64,
    "bar_width" : 5,
    "T"         : T,
    "gt"        : gt,
    "obs"       : obs.tolist(),
    "filtered"  : [[round(p, 4) for p in row] for row in filtered.tolist()],
    "bar_x"     : bar_positions,
    "events"    : bin_events,
    "ll"        : round(ll, 2),
    "ll_uniform": round(ll_u, 2),
}

with open("plots/data.json", "w") as f:
    json.dump(payload, f, separators=(",", ":"))

import os
size_kb = os.path.getsize("plots/data.json") / 1024
print(f"saved plots/data.json  ({size_kb:.0f} KB)")

# ── static summary plot (first 300 bins for readability) ──────────────────────
PLOT_T = 300
STATE_COL   = ["#e05c5c", "#888888", "#5c8fe0"]
STATE_NAMES = ["LEFT", "REST", "RIGHT"]
MODE_INT    = {"LEFT": 0, "REST": 1, "RIGHT": 2}

t      = np.arange(PLOT_T)
gt_int = np.array([MODE_INT[m] for m in gt[:PLOT_T]])

fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
fig.subplots_adjust(hspace=0.08)

ax = axes[0]
for i, (col, name) in enumerate(zip(STATE_COL, STATE_NAMES)):
    ax.fill_between(t, 0, 1, where=(gt_int == i), color=col, alpha=0.85, linewidth=0)
ax.set_yticks([])
ax.set_ylabel("ground truth", fontsize=9)
patches = [mpatches.Patch(color=c, label=n) for c, n in zip(STATE_COL, STATE_NAMES)]
ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.7)

ax = axes[1]
for i, col in enumerate(STATE_COL):
    ax.fill_between(t, 0, 1, where=(obs[:PLOT_T] == i), color=col, alpha=0.85, linewidth=0)
ax.set_yticks([])
ax.set_ylabel("observation", fontsize=9)

ax = axes[2]
for i, (col, name) in enumerate(zip(STATE_COL, STATE_NAMES)):
    ax.plot(t, filtered[:PLOT_T, i], color=col, lw=1.5, label=name)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color="k", lw=0.5, ls="--", alpha=0.3)
ax.set_ylabel("P(state | obs)", fontsize=9)
ax.set_xlabel("time bin (10 ms)", fontsize=9)
ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

fig.suptitle(
    f"HMM forward algorithm on synthetic DVS events (first {PLOT_T} bins of 3000)\n"
    f"structured Δlog-likelihood vs uniform: {ll - ll_u:+.1f} nats",
    fontsize=10, y=0.98
)
plt.savefig("plots/demo.png", dpi=150, bbox_inches="tight")
print("saved plots/demo.png")
plt.show()
