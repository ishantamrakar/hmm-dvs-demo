import numpy as np
from scipy.special import logsumexp


class HMM:
    def __init__(self, A, B, pi):
        self.log_A  = np.log(A)
        self.log_B  = np.log(B)
        self.log_pi = np.log(pi)

    def forward(self, obs):
        T, N = len(obs), len(self.log_pi)
        log_alpha = np.empty((T, N))

        log_alpha[0] = self.log_pi + self.log_B[:, obs[0]]
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = logsumexp(log_alpha[t-1] + self.log_A[:, j]) + self.log_B[j, obs[t]]

        ll = logsumexp(log_alpha[-1])
        filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        return log_alpha, ll, filtered


if __name__ == "__main__":
    from event_simulator import simulate
    from observations import extract_observations

    modes = ["REST"]*20 + ["RIGHT"]*40 + ["REST"]*10 + ["LEFT"]*30 + ["REST"]*20
    events, gt = simulate(modes)
    obs = extract_observations(events, num_bins=len(modes))

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

    hmm = HMM(A, B, pi)
    _, ll, filtered = hmm.forward(obs)

    hmm_uniform = HMM(np.full((3,3), 1/3), B, pi)
    _, ll_uniform, _ = hmm_uniform.forward(obs)

    print(f"log-likelihood  structured : {ll:.1f}")
    print(f"log-likelihood  uniform    : {ll_uniform:.1f}")
    print(f"delta                      : {ll - ll_uniform:.1f} nats\n")

    state_names = ["LEFT", "REST", "RIGHT"]
    print(f"{'bin':>4}  {'gt':>6}  {'MAP state':>10}  {'P(LEFT)':>8}  {'P(REST)':>8}  {'P(RIGHT)':>8}")
    for t in range(len(modes)):
        map_state = state_names[np.argmax(filtered[t])]
        mark = "" if map_state == gt[t] else "<--"
        print(f"  {t:3d}  {gt[t]:>6}  {map_state:>10}  {filtered[t,0]:.3f}     {filtered[t,1]:.3f}     {filtered[t,2]:.3f}  {mark}")
