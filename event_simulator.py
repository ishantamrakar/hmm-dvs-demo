import numpy as np

_VX = {"LEFT": -2, "REST": 0, "RIGHT": 2}


def _wrap(x, frame_size, bar_width):
    return x % (frame_size - bar_width)


def _moved_cols(x_old, x_new, vx, frame_size, bar_width):
    """Return (on_cols, off_cols) accounting for toroidal wrap."""
    W = frame_size - bar_width  # effective wrap width

    if vx == 0:
        return [], []

    # number of columns actually crossed (capped at bar_width to avoid overlap)
    delta = abs(x_new - x_old)
    if delta == 0 and x_old != x_new:  # wrapped around
        delta = abs(vx)

    if vx > 0:
        on_cols  = [(x_old + bar_width + i) % frame_size for i in range(delta)]
        off_cols = [(x_old + i)              % frame_size for i in range(delta)]
    else:
        on_cols  = [(x_new + i)              % frame_size for i in range(delta)]
        off_cols = [(x_new + bar_width + i)  % frame_size for i in range(delta)]

    return on_cols, off_cols


def simulate(mode_sequence, frame_size=64, bar_width=5, bin_ms=10, noise_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    bin_us = bin_ms * 1000
    events = []
    x = frame_size // 2

    for k, mode in enumerate(mode_sequence):
        t0 = k * bin_us
        vx = _VX[mode]
        x_new = _wrap(x + vx, frame_size, bar_width)

        on_cols, off_cols = _moved_cols(x, x_new, vx, frame_size, bar_width)

        for col in on_cols:
            ts = t0 + rng.integers(0, bin_us, size=frame_size)
            for y, t in enumerate(ts):
                events.append((int(t), col, y, 1))

        for col in off_cols:
            ts = t0 + rng.integers(0, bin_us, size=frame_size)
            for y, t in enumerate(ts):
                events.append((int(t), col, y, -1))

        n_noise = int(frame_size * frame_size * noise_rate)
        t_n  = t0 + rng.integers(0, bin_us, size=n_noise)
        xs   = rng.integers(0, frame_size, size=n_noise)
        ys   = rng.integers(0, frame_size, size=n_noise)
        pols = rng.choice([-1, 1], size=n_noise)
        for t, nx, ny, p in zip(t_n, xs, ys, pols):
            events.append((int(t), int(nx), int(ny), int(p)))

        x = x_new

    events.sort(key=lambda e: e[0])
    return events, list(mode_sequence)


def random_mode_sequence(n_bins, seed=42):
    """Generate a random mode sequence with realistic dwell times."""
    rng = np.random.default_rng(seed)
    modes = []
    while len(modes) < n_bins:
        mode = rng.choice(["LEFT", "REST", "RIGHT"])
        dwell = int(rng.integers(10, 40))
        modes.extend([mode] * dwell)
    return modes[:n_bins]


if __name__ == "__main__":
    modes = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20
    events, _ = simulate(modes)

    bin_us = 10_000
    print("fixed sequence — flow per bin:")
    for b in [19, 20, 55, 60, 61]:
        ev  = [e for e in events if b*bin_us <= e[0] < (b+1)*bin_us]
        on  = [e[1] for e in ev if e[3] ==  1]
        off = [e[1] for e in ev if e[3] == -1]
        flow = np.mean(on) - np.mean(off) if on and off else float("nan")
        print(f"  bin {b:3d} ({modes[b]:5s})  flow={flow:+.2f}  ON={len(on)}  OFF={len(off)}")

    print("\nrandom sequence (300 bins):")
    rand_modes = random_mode_sequence(300)
    events2, _ = simulate(rand_modes)
    print(f"  events: {len(events2)},  t=[{events2[0][0]}, {events2[-1][0]}] µs")
    # spot check a few transitions
    transitions = [i for i in range(1, len(rand_modes)) if rand_modes[i] != rand_modes[i-1]][:5]
    for t in transitions:
        ev  = [e for e in events2 if t*bin_us <= e[0] < (t+1)*bin_us]
        on  = [e[1] for e in ev if e[3] ==  1]
        off = [e[1] for e in ev if e[3] == -1]
        flow = np.mean(on) - np.mean(off) if on and off else float("nan")
        print(f"  bin {t:3d} ({rand_modes[t-1]:5s}->{rand_modes[t]:5s})  flow={flow:+.2f}")
