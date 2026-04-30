import numpy as np


def simulate(mode_sequence, frame_size=64, bar_width=5, bin_ms=10, noise_rate=0.05, seed=0):
    """
    Simulate a moving bar on a DVS sensor.

    Returns:
        events: list of (t_us, x, y, polarity) tuples, sorted by t_us
        gt_modes: list of mode strings, length len(mode_sequence)
    """
    rng = np.random.default_rng(seed)

    # velocity in pixels per bin for each mode
    VELOCITY = {"LEFT": -2, "REST": 0, "RIGHT": 2}

    events = []
    bar_x = frame_size // 2  # leading (left) edge of bar, starts at center
    bin_us = bin_ms * 1_000  # 10 ms in microseconds

    for bin_idx, mode in enumerate(mode_sequence):
        t_start = bin_idx * bin_us
        vx = VELOCITY[mode]
        new_bar_x = int(np.clip(bar_x + vx, 0, frame_size - bar_width))

        # columns newly entered by the bar (ON events)
        if vx > 0:
            entered_cols = range(bar_x + bar_width, new_bar_x + bar_width)
            left_cols    = range(bar_x,             new_bar_x)
        elif vx < 0:
            entered_cols = range(new_bar_x,          bar_x)
            left_cols    = range(new_bar_x + bar_width, bar_x + bar_width)
        else:
            entered_cols = range(0, 0)  # empty
            left_cols    = range(0, 0)

        # emit signal events for every pixel in the bar's vertical extent
        for col in entered_cols:
            for y in range(frame_size):
                t = int(t_start + rng.integers(0, bin_us))
                events.append((t, int(col), y, 1))   # ON

        for col in left_cols:
            for y in range(frame_size):
                t = int(t_start + rng.integers(0, bin_us))
                events.append((t, int(col), y, -1))  # OFF

        # noise: ~noise_rate fraction of pixels fire a random event
        n_noise = int(frame_size * frame_size * noise_rate)
        for _ in range(n_noise):
            t   = int(t_start + rng.integers(0, bin_us))
            nx  = int(rng.integers(0, frame_size))
            ny  = int(rng.integers(0, frame_size))
            pol = int(rng.choice([-1, 1]))
            events.append((t, nx, ny, pol))

        bar_x = new_bar_x

    events.sort(key=lambda e: e[0])
    return events, list(mode_sequence)


if __name__ == "__main__":
    mode_sequence = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20

    events, gt_modes = simulate(mode_sequence)

    print(f"Total events : {len(events)}")
    print(f"Time range   : {events[0][0]} µs → {events[-1][0]} µs")
    print(f"First 5 events (t_us, x, y, pol):")
    for e in events[:5]:
        print(f"  {e}")

    # sanity check: count ON vs OFF events in the first RIGHT bin (bin 20)
    bin_ms = 10
    bin_us = bin_ms * 1_000
    right_start = 20 * bin_us
    right_end   = 21 * bin_us
    bin_events  = [e for e in events if right_start <= e[0] < right_end]
    on_events   = [e for e in bin_events if e[3] == 1]
    off_events  = [e for e in bin_events if e[3] == -1]
    print(f"\nBin 20 (first RIGHT bin):")
    print(f"  ON events  : {len(on_events)}  mean_x = {np.mean([e[1] for e in on_events]):.1f}" if on_events else "  ON events: 0")
    print(f"  OFF events : {len(off_events)}  mean_x = {np.mean([e[1] for e in off_events]):.1f}" if off_events else "  OFF events: 0")
