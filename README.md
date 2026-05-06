# HMM Forward Algorithm on Synthetic DVS Events

A small demo that recovers hidden motion states from noisy event-camera data using the HMM forward algorithm — no real event-camera hardware required.

The idea: a 5-pixel-wide vertical bar drifts across a 64×64 sensor, switching between LEFT, REST, and RIGHT motion modes. Each mode fires a characteristic pattern of ON/OFF events. The forward algorithm reads those events bin-by-bin and maintains a running belief over which mode is active.

![demo plot](plots/demo.png)

## How it works

**Event simulation** — the bar moves at ±2 px per 10 ms bin. When it crosses a column boundary it emits ON events on the leading edge and OFF events on the trailing edge, mimicking how a real DVS sensor responds to contrast changes. Random noise events (~5% of pixels per bin) are sprinkled on top.

**Observation extraction** — within each bin, net horizontal flow is estimated as `mean_x(ON) − mean_x(OFF)`. A rightward-moving bar pushes ON events ahead of OFF events, giving a positive flow signal. The result is discretized into three symbols: LEFT_FLOW, NO_FLOW, RIGHT_FLOW.

**Inference** — a three-state HMM (LEFT / REST / RIGHT) with hand-set transition and emission matrices runs the forward algorithm in log space. The filtered posterior `P(state | observations so far)` tracks mode switches with a small lag of a few bins.

**Sanity check** — the same observations are also fed to a baseline HMM with uniform transitions. The structured model wins by a wide margin in log-likelihood, confirming the transition structure is doing real work.

The posterior closely follows the ground truth, lagging by a few bins at transitions. That lag is expected: the forward algorithm is causal and needs a handful of observations to shift its belief after a mode change.

## Forward trellis visualization

The interactive demo includes a forward trellis panel that shows the recursion happening step by step. Here is how to read it.

**Layout** — columns are time steps, rows are the three hidden states (LEFT at top, REST in the middle, RIGHT at bottom). The window slides forward, keeping the 5 most recent bins visible. The rightmost column is always the step just computed.

**Filled circles** — each circle represents one state at one time step. The filled area inside the circle encodes `α(t, s)`, the normalized forward probability for that state at that time. A nearly full circle means the algorithm is highly confident the system is in that state; a tiny sliver means it considers that state unlikely. The circle border is always fixed size so you can compare fills across nodes at a glance.

**Dashed rings** — the dashed ring around each node in the rightmost column encodes `B[s, obs]`, the emission probability of the current observation given that state. A large ring means this state would readily explain what the sensor just saw; a small ring means the observation is surprising under that state. This is the scaling factor applied after the sum in the recursion.

**Arrows** — each arrow from state `i` at `t−1` to state `j` at `t` represents one term inside the summation `Σᵢ α(t−1, i) · A[i,j]`. Arrow thickness encodes the magnitude of that term relative to the largest term in the same column. Thick arrows are the dominant paths contributing probability mass to the destination node; thin arrows are negligible paths. Historical arrows (not the most recent step) are dimmed so the active computation stands out.

**B= numbers** — the value printed below each dashed ring is the exact emission probability `B[s, obs]` for that state given the current observation. It is the factor that multiplies the incoming sum to produce the final `α(t, s)`.

**α numbers** — the value printed inside each node in the two rightmost columns is the normalized `α(t, s)`. The three values in any column sum to 1. Watching these numbers shift across a mode transition shows the algorithm gradually moving probability mass from one state to another over several bins.

**What to look for** — at a mode transition, the dominant state's node starts shrinking while another state's node grows. This happens gradually because the self-transition probability is high (0.90), so the algorithm needs several consistent observations before it is willing to commit to a new state. That visible lag is the forward algorithm working exactly as intended.

## Parameters

The HMM matrices are hand-set rather than learned. The transition matrix encodes a strong self-transition prior (0.90) with small probability of switching modes. The emission matrix reflects that LEFT motion mostly produces LEFT_FLOW but occasionally produces noise observations, and so on.

```
A (transitions)            
LEFT  → [0.90 0.08 0.02]   
REST  → [0.05 0.90 0.05]   
RIGHT → [0.02 0.08 0.90]   

B (emissions: L / N / R flow)
LEFT  → [0.75 0.20 0.05]
REST  → [0.15 0.70 0.15]
RIGHT → [0.05 0.20 0.75]
```

