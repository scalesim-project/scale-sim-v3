# Cost table for PE energy and area (Project 3)

Every per-operation energy and area used in `energy.py` comes from this file. All values are for a **45 nm** process, with an **INT8 baseline** (8-bit activations and weights, 32-bit accumulator). Last updated: 2026-11-15.

Rule for the team: never change a number in `energy.py` without changing it here too, with its source.

## 1. Primitive operation costs (45 nm)

| Operation | Energy (pJ) | Area (µm²) | Source |
|---|---|---|---|
| 8-bit integer add | 0.03 | 36 | [1] (via [2]) |
| 16-bit integer add | 0.05 | 67 | [1] (via [2]) |
| 32-bit integer add | 0.1 | 137 | [1] (via [2], [3]) |
| 8-bit integer multiply | 0.2 | 282 | [1] (via [2]) |
| 32-bit integer multiply | 3.1 | 3495 | [1] (via [2], [3]) |
| 16-bit FP add | 0.4 | 1360 | [1] (via [2]) |
| 16-bit FP multiply | 1.1 | 1640 | [1] (via [2]) |
| 8-bit shift | 0.024 | 34 | [4] |
| 16-bit shift | 0.057 | 73 | [4] |
| 32-bit shift | 0.13 | 157 | [4] |
| 32-bit SRAM read | 5 | n/a | [1] (via [2], [3]) |
| 32-bit DRAM read | 640 | n/a | [1] (via [2], [3]) |

**Mitchell vs. exact multiplier (32 nm, Synopsys DC, 250 MHz)** [5]:

| Design | 8-bit energy (pJ) | 8-bit area (µm²) | 16-bit energy (pJ) | 16-bit area (µm²) |
|---|---|---|---|---|
| Exact multiplier | 0.29 | 403 | 2.77 | 1681 |
| Mitchell log multiplier | 0.22 | 312 | 1.27 | 909 |
| **Ratio (Mitchell / exact)** | **0.759** | **0.774** | 0.458 | 0.541 |

Energy in [5] is computed as power × critical-path delay.

## 2. Derived cost per PE (what `energy.py` uses)

Each PE = its multiply step + one 32-bit accumulate add (0.1 pJ, 137 µm²). Pipeline registers, control and wires are **not** included, for any PE.

| PE type | Formula | Energy / MAC (pJ) | vs. baseline | Area (µm²) | vs. baseline | Weight bits stored |
|---|---|---|---|---|---|---|
| `baseline` (INT8) | 8b mult + 32b add | 0.300 | 1.00× | 419 | 1.00× | 8 |
| `pot` | 16b shift + 32b add | 0.157 | 0.52× | 210 | 0.50× | 4 |
| `apot` | 2 × 16b shift + 16b add + 32b add | 0.264 | 0.88× | 350 | 0.84× | 8 |
| `mitchell` | 8b mult × 0.759 [5] + 32b add | 0.252 | 0.84× | 355 | 0.85× | 8 |
| `bitserial` | 8 × 16b add + 32b add | 0.500 | 1.67× | 204 | 0.49× | 8 |
| `baseline_fp16` (reference only) | FP16 mult + FP16 add | 1.500 | 5.00× | 3000 | 7.16× | 16 |

Why these formulas:

- **PoT:** an 8-bit activation shifted by up to 7 places needs a 15-bit result, so we use the 16-bit shifter cost.
- **APoT:** matches `shiftmac.py`, which sums two 4-bit PoT terms (8 bits stored per weight). The APoT paper's own encoding can reach 4 bits; if you switch to it, change "Weight bits stored" to 4.
- **Mitchell:** [5] is in 32 nm, so we apply its exact-to-Mitchell **ratio** to the 45 nm 8-bit multiplier from [1], not its absolute numbers.
- **Bit-serial:** one 16-bit shift-add per weight bit; the 1-bit shift itself is wiring. It saves area, but costs **more** energy per MAC than the baseline. That is a real result worth reporting, not a bug.

## 3. Memory costs (what `energy.py` uses)

| Item | Value | How it was derived |
|---|---|---|
| SRAM energy per bit | 0.156 pJ/bit | 5 pJ per 32-bit read [1] ÷ 32 |
| DRAM energy per bit | 20 pJ/bit | 640 pJ per 32-bit read [1] ÷ 32 |
| Activation / output word | 8 bits | INT8 baseline |
| Weight word | 4 or 8 bits | per PE type, table 2 |

SCALE-Sim reports accesses in words. `energy.py` multiplies filter accesses by the PE's weight bits, and ifmap/ofmap accesses by 8 bits. Writes are charged the same as reads.

## 4. Iso-area array sizes

Baseline 32×32 array MAC area = 1024 × 419 µm² = 429,056 µm². With the same area:

| PE type | PEs that fit | Square array to simulate |
|---|---|---|
| `baseline` | 1024 | 32 × 32 |
| `pot` | 2043 | 45 × 45 |
| `apot` | 1226 | 35 × 35 |
| `mitchell` | 1208 | 34 × 34 |
| `bitserial` | 2103 | 45 × 45 (with `PeCyclesPerMac : 8`) |

## 5. Sanity check: GEMM test (M=256, N=128, K=256, 32×32 WS)

| PE type | Total energy (µJ) | MAC share of energy |
|---|---|---|
| `baseline` | 92.35 | 2.7% |
| `pot` | 88.51 | 1.5% |
| `apot` | 92.05 | 2.4% |
| `mitchell` | 91.94 | 2.3% |
| `bitserial` (cycles not stretched) | 94.03 | 4.5% |

DRAM dominates this small GEMM, so PE choice barely moves total energy. PoT saves the most mainly because its 4-bit weights halve filter traffic. Expect larger MAC shares on conv layers with more on-chip reuse; check this before drawing conclusions.

## 6. Caveats to state in the paper

1. **Sources disagree on SRAM size.** [2] labels the 5 pJ SRAM read as an 8 KB array; [3] labels it 32 KB. SCALE-Sim's default buffers are 64 KB, so the real cost is likely higher. Sweep SRAM energy from 5 to 20 pJ per 32 bits.
2. **[4] differs from [1] on FP16.** [4] lists FP16 multiply at 0.9 pJ and FP32 add at 1.1 pJ; [1] via [2] gives 1.1 and 0.9. We use [1] for all add and multiply values, and [4] only for shifts.
3. **Mixed nodes.** The Mitchell ratio comes from 32 nm [5]; ratios transfer between nodes better than absolute values, but this is an approximation.
4. **What is left out:** pipeline registers, leakage, interconnect, and control. These add the same overhead to every PE type, so they shrink relative savings. Say so explicitly.
5. **Sensitivity sweep:** run shifter-based PE energies at 0.5×, 1×, and 2× the values above, and DRAM at 10–30 pJ/bit, so the conclusions do not hinge on one number.


## References

1. M. Horowitz, "Computing's energy problem (and what we can do about it)," IEEE ISSCC, 2014. The energy/area table is an image in the original PDF, so values were read from [2] and [3], which reproduce it.
2. V. Sze, MIT MTL seminar slides, 2020, slide "Power Dominated by Data Movement" (cites Horowitz, ISSCC 2014). https://eems.mit.edu/wp-content/uploads/2020/02/2020_mtl_seminar.pdf
3. S. Han et al., "EIE: Efficient Inference Engine on Compressed Deep Neural Network," ISCA, 2016, Table I. https://ar5iv.arxiv.org/html/1602.01528
4. H. You et al., "ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer," NeurIPS, 2023, Table 1 (values attributed to ShiftAddNet, NeurIPS 2020). https://arxiv.org/html/2306.06446v6
5. M. S. Kim, "Cost-Efficient Approximate Log Multipliers for Convolutional Neural Networks," Ph.D. dissertation, UC Irvine, 2020, Table 2.2. https://escholarship.org/content/qt3w4980x3/qt3w4980x3.pdf
