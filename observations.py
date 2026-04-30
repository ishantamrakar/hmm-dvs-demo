import numpy as np


def extract_observations(events, num_bins, bin_ms=10, threshold=0.5):
    """Returns int array of observation indices (0=LEFT_FLOW, 1=NO_FLOW, 2=RIGHT_FLOW), shape (num_bins,)."""
    bin_us = bin_ms * 1000
    obs = np.ones(num_bins, dtype=int)  # default NO_FLOW

    for k in range(num_bins):
        ev = [e for e in events if k*bin_us <= e[0] < (k+1)*bin_us]
        on_x  = [e[1] for e in ev if e[3] ==  1]
        off_x = [e[1] for e in ev if e[3] == -1]

        if not on_x or not off_x:
            continue

        flow = np.mean(on_x) - np.mean(off_x)
        if   flow >  threshold: obs[k] = 2
        elif flow < -threshold: obs[k] = 0

    return obs


if __name__ == "__main__":
    from event_simulator import simulate

    modes = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20
    events, gt = simulate(modes)
    obs = extract_observations(events, num_bins=len(modes))

    label = {0: "LEFT_FLOW", 1: "NO_FLOW", 2: "RIGHT_FLOW"}
    print(f"{'bin':>4}  {'gt':>6}  {'obs'}")
    for k in range(len(modes)):
        mark = "<--" if gt[k] != label[obs[k]].replace("_FLOW","").replace("NO","REST") else ""
        print(f"  {k:3d}  {gt[k]:>6}  {label[obs[k]]}  {mark}")
