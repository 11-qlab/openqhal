# AOD Simulation: Loss Tolerance of the 5×5 RL Model

## Setup
- Model: sb3_5x5_final.zip (Colab float32, 91.2% on 1000-episode eval)
- Metric: F1 score (precision × recall over target sites)
- Problems: 100 random 3-move and 6-move rearrangements
- Loss model: per-atom-per-move, uniform random

## Results (scramble=6)

| Loss/move | F1    | Atoms lost |
|-----------|-------|------------|
| 0.000     | 1.000 | 0.00       |
| 0.010     | 0.997 | 0.04       |
| 0.020     | 0.994 | 0.08       |
| 0.050     | 0.990 | 0.16       |
| 0.100     | 0.980 | 0.30       |

## Finding
At the 2% per-move loss rate characteristic of current neutral-atom
hardware (QuEra, Pasqal, Infleqtion), the model's plans achieve
99.4% F1. Even at 5× current loss rates, F1 remains above 98%.

The plans are loss-tolerant because the row/column action space
does not depend on which specific atoms survive — moving an
entire row is unaffected by loss at individual sites.

## Implication
The model's output is deployable in principle. The remaining barrier
is the Move → AOD pulse conversion, which no public hardware exposes.
