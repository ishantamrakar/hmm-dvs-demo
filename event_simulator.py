import numpy as np

_VX = {"LEFT": -2, "REST": 0, "RIGHT": 2}


def simulate(mode_sequence, frame_size=64, bar_width=5, bin_ms=10, noise_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    bin_us = bin_ms * 1000
    events = []
    x = frame_size // 2  # bar left edge

    for k, mode in enumerate(mode_sequence):
        t0 = k * bin_us
        vx = _VX[mode]
        x_new = int(np.clip(x + vx, 0, frame_size - bar_width))

        if vx > 0:
            on_cols  = range(x + bar_width, x_new + bar_width)
            off_cols = range(x, x_new)
        elif vx < 0:
            on_cols  = range(x_new, x)
            off_cols = range(x_new + bar_width, x + bar_width)
        else:
            on_cols = off_cols = range(0)

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


if __name__ == "__main__":
    modes = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20
    events, _ = simulate(modes)

    print(f"events: {len(events)},  t=[{events[0][0]}, {events[-1][0]}] µs")

    bin_us = 10_000
    for b in [19, 20, 60, 61]:
        ev = [e for e in events if b*bin_us <= e[0] < (b+1)*bin_us]
        on  = [e[1] for e in ev if e[3] ==  1]
        off = [e[1] for e in ev if e[3] == -1]
        flow = np.mean(on) - np.mean(off) if on and off else float("nan")
        print(f"  bin {b:3d} ({modes[b]:5s})  flow={flow:+.2f}  ON={len(on)}  OFF={len(off)}")
