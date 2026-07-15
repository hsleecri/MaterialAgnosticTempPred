# Material-Agnostic Temperature Field Prediction for Metal Additive Manufacturing via a Parametric PINN Framework

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.addma.2026.105289-blue)](https://doi.org/10.1016/j.addma.2026.105289)

Hyeonsu Lee, Jihoon Jeong — Texas A&M University

Published in [*Additive Manufacturing* **127** (2026) 105289](https://doi.org/10.1016/j.addma.2026.105289)

> Accurate thermal modeling in metal additive manufacturing (AM) is essential for understanding the process–structure–performance relationship. This repository provides the official implementation of a **parametric Physics-Informed Neural Network (PINN)** framework that achieves material-agnostic temperature prediction across arbitrary metal alloys — without labeled data, retraining, or pre-training.

---

## Overview

The proposed framework consists of three key components:

| Component | Description |
|-----------|-------------|
| **Decoupled Parametric PINN** | FiLM-based architecture that separately encodes spatiotemporal coordinates and material properties, enabling multiplicative feature modulation aligned with the governing PDEs |
| **Physics-Guided Output Scaling** | Material-dependent temperature scale derived from Rosenthal's analytical solution, stabilizing training across materials with vastly different thermal properties |
| **Hybrid Optimization** | Adam (global exploration) → mini-batch L-BFGS (local refinement) strategy that achieves convergence within 4.4% of baseline training epochs |

The framework is evaluated on a bare-plate numerical benchmark in metal AM across five metal alloys — including two out-of-distribution (OOD) materials — achieving up to **64.2% reduction in relative L2 error** compared to the non-parametric baseline.

---

## Repository Structure

```
.
├── src/
│   ├── train.py          # Main training script
│   ├── model.py          # Proposed decoupled PINN architecture
│   └── utils.py          # Collocation point sampling utilities
├── conf/
│   └── config.yaml       # Hydra configuration (hyperparameters, domain, materials)
├── FEM_data/             # Ground-truth FEM simulation data
│   ├── Ti_6AL_4V_(4430,560,6.7)/data/data.npy
│   ├── Inconel718_(8220,435,11.4)/data/data.npy
│   ├── SS316L_(8000,500,16)/data/data.npy
│   ├── AlSi10Mg_(2670,950,150)/data/data.npy
│   └── Copper_(8960,385,401)/data/data.npy
├── fem_data_generation/  # JAX-AM scripts used to generate FEM_data/
│   ├── generate_fem_data.py
│   └── models_bareplate.py
├── requirements.txt
└── README.md
```

---

## Cloning (Git LFS Required)

The FEM ground-truth data files (`*.npy`, ~106 MB each) are stored with [Git LFS](https://git-lfs.github.com/). Install Git LFS before cloning:

```bash
git lfs install
git clone https://github.com/hsleecri/MaterialAgnosticTempPred.git
```

If you have already cloned without LFS, run `git lfs pull` to fetch the data files.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (experiments run on a single NVIDIA RTX 5090)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Training

All hyperparameters are managed via [Hydra](https://hydra.cc/). Run from the repository root:

```bash
python src/train.py
```

Key parameters can be overridden from the command line:

```bash
# Change GPU device and number of epochs
python src/train.py base.device=1 base.epoch_final=10000

# Change random seed
python src/train.py base.seed=3

```

All outputs (model weights, loss history, plots) are saved to `results/bareplate/proposed/<run_name>/`.

---

## Configuration

Edit `conf/config.yaml` to modify:

| Section | Key Parameters |
|---------|---------------|
| `base` | `device`, `seed`, `epoch_final` |
| `material` | Training material property bounds (ρ, C_p, k) |
| `laser` | Process parameters (power, speed, beam radius) |
| `schedule` | Adam/L-BFGS epochs, learning rates, mini-batch sizes |
| `collocation` | Sampling strategy (`manual` by default) |
| `validation` | Paths to FEM ground-truth data |

### Material Property Space

The training material property space M is defined in `conf/config.yaml`:

```yaml
material:
  rho_bounds_si: [3000.0, 10000.0]  # density [kg/m^3]
  cp_bounds_si:  [300.0,  1000.0]   # specific heat capacity [J/(kg·K)]
  k_bounds_si:   [3.0,    50.0]     # thermal conductivity [W/(m·K)]
```

In-distribution materials (Ti-6Al-4V, Inconel 718, SS 316L) and out-of-distribution materials (AlSi10Mg, Copper) are all included in `FEM_data/` for evaluation.

---

## FEM Ground-Truth Data

Ground-truth temperature fields were generated using [JAX-AM](https://github.com/CMSL-HKUST/jax-am), an open-source GPU-accelerated FEM solver for metal AM simulations. Each `.npy` file contains an array of shape `(N, 5)` with columns `[x, y, z, t, T]` (coordinates in mm, time in s, temperature in K).

| Material | ρ [kg/m³] | C_p [J/kg·K] | k [W/m·K] | Split |
|----------|-----------|--------------|-----------|-------|
| Ti-6Al-4V | 4430 | 560 | 6.7 | In-distribution |
| Inconel 718 | 8220 | 435 | 11.4 | In-distribution |
| SS 316L | 8000 | 500 | 16.0 | In-distribution |
| AlSi10Mg | 2670 | 950 | 150 | Out-of-distribution |
| Copper | 8960 | 385 | 401 | Out-of-distribution |

### Regenerating the FEM data

The scripts used to generate `FEM_data/` are provided in `fem_data_generation/`. They require [JAX-AM](https://github.com/CMSL-HKUST/jax-am) to be installed (the PINN training itself does **not** need JAX-AM). From `fem_data_generation/`:

```bash
# One of the five materials used in the paper
python generate_fem_data.py --material Copper

# Or an arbitrary material
python generate_fem_data.py --rho 4430 --cp 560 --k 6.7 --name MyAlloy
```

Each run simulates a 3-second single-track laser scan (see process parameters below), saves VTU snapshots every 10 steps, and exports `FEM_data/<name>_(rho,cp,k)/data/data.npy` in the `(N, 5)` format described above. Select the GPU with `CUDA_VISIBLE_DEVICES`.

---

## Process Parameters

| Parameter | Value |
|-----------|-------|
| Spatial domain | 40 × 10 × 6 mm³ |
| Laser power | 500 W |
| Laser absorptivity | 0.4 |
| Beam radius | 1.5 mm |
| Scanning speed | 10 mm/s |
| Scanning time | 3 s |
| Initial temperature | 300 K |
| Ambient temperature | 300 K |
| Convection coefficient | 50 W/(m²·K) |
| Surface emissivity | 0.3 |

---

## Citation

```bibtex
@article{lee2026material,
  title={Material-agnostic temperature field prediction for metal additive manufacturing via a parametric PINN framework},
  author={Lee, Hyeonsu and Jeong, Jihoon},
  journal={Additive Manufacturing},
  pages={105289},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This repository uses different licenses for different components:

- **Source code** (`src/`, `conf/`): [PolyForm Noncommercial 1.0.0](LICENSE) — free to use, modify, and redistribute for **noncommercial purposes**, including academic research. Commercial use is not permitted.
- **`fem_data_generation/`**: [GPL-3.0](fem_data_generation/LICENSE), as this code is adapted from [JAX-AM](https://github.com/CMSL-HKUST/jax-am) (GPL-3.0).
- **`FEM_data/` (simulation data)**: [CC BY-NC 4.0](FEM_data/LICENSE) — attribution required, noncommercial use only.

For commercial licensing inquiries, please contact hslee@tamu.edu.

---

## Acknowledgment

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-02216260) and a grant of the Basic Research Program funded by the Korea Institute of Machinery and Materials (grant number: NK254A, Project Title: Development of Core Technologies for Advanced Chiplet Packaging Equipment).

## Reference

A significant portion of this codebase is adapted from the following repositories. We are grateful for their great contributions to the community.

- Liao, Shuheng, et al. "Hybrid thermal modeling of additive manufacturing processes using physics-informed neural networks for temperature prediction and parameter identification." Computational Mechanics 72.3 (2023): 499-512. (https://github.com/ShuhengLiao/Physics_informed_AM?tab=readme-ov-file)
- JAX-AM: https://github.com/CMSL-HKUST/jax-am
