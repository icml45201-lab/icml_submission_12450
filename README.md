# Comprehensive Rebuttal Appendix: ICML 2026 Submission 12450
**Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting**

This anonymous repository serves as an extended, high-resolution appendix to our official author rebuttal. To address reviewer inquiries comprehensively, we have conducted extensive new ablation studies, evaluated 14 total baselines, tested adaptive ODE solvers, and formalized the spectral properties of our continuous-time operator. 

All figures, tables, and theoretical analyses enclosed below are directly referenced in our official rebuttal text.

---

## Table of Contents
1. [Expanded Benchmarking & $O(1)$ Inference Efficiency](#1-expanded-benchmarking--o1-inference-efficiency)
2. [Architectural Ablations: LoRA, MLP, and Weighting Schedules](#2-architectural-ablations-lora-mlp-and-weighting-schedules)
3. [Continuous-Time "Consistent Koopman" (Azencot) Ablation](#3-continuous-time-consistent-koopman-azencot-ablation)
4. [ODE Solver Stability & Adaptive Steps (Dopri5)](#4-ode-solver-stability--adaptive-steps-dopri5)
5. [The "Closure Problem": Spectral Bias & 1000-Step Stability](#5-the-closure-problem-spectral-bias--1000-step-stability)
6. [Spatial Error Localization (Difference Maps)](#6-spatial-error-localization-difference-maps)

---

## 1. Expanded Benchmarking & $O(1)$ Inference Efficiency
**Addressed to:** *Reviewers z2Gs, B4CM (Request for broader baseline comparisons beyond diffusion models).*

We expanded our evaluation to encompass 14 total models, representing the current state-of-the-art in spatial-temporal PDE surrogates, including Fourier Neural Operators (FNOs), U-Nets, and Graph/Transformer variants. 

**Key Takeaway:** The Continuous-Time KAE establishes a distinct Pareto frontier. While highly non-linear generative models (ACDM) capture slightly more high-frequency stochastic texture in the short term, the strict global linearity of the KAE's latent space allows for analytical matrix exponentiation. This yields an **inference speedup of >300$\times$** over diffusion models and **>5$\times$** over continuous U-Nets, while simultaneously achieving superior stability over long horizons ($Tra_{long}$).

| Architecture Paradigm | Model | Avg. Step Inference (ms) | Mean VRAM (MB) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral / Operator** | FNO-16 | $1.17 \pm 0.01$ | $184.1$ | $20.8 \pm 2.0$ |
| | FNO-32 | $1.17 \pm 0.00$ | $183.9$ | *Diverged* |
| **Convolutional** | ResNet-dil | $3.46 \pm 0.02$ | $178.6$ | $22.0 \pm 2.4$ |
| | U-Net (m8) | $6.16 \pm 0.01$ | $184.1$ | $22.2 \pm 3.6$ |
| **Attention / Graph** | TF-Enc | $0.60 \pm 0.25$ | $3448.6$ | $22.2 \pm 3.9$ |
| **Generative (Diffusion)**| ACDM | $41.77 \pm 0.01$ | $659.2$ | $22.6 \pm 4.0$ |
| **Continuous Koopman** | **KAE (Ours)** | **$0.00104 \pm 0.0001$** | $2751.3$ | **$14.9 \pm 1.3$** |

---

## 2. Architectural Ablations: LoRA, MLP, and Weighting Schedules
**Addressed to:** *Reviewer z2Gs, B4CM (Requests for MLP parameterization vs. LoRA and Cosine vs. Uniform empirical ablations).*

To empirically validate our architectural design, we ablated the LoRA parameterization against a full-rank MLP ($\mathbf{K}_{\text{cont}} = \text{MLP}(\phi)$) and compared our Cosine temporal weighting against a Uniform schedule.

**Key Takeaways:**
1. **LoRA prevents severe overfitting:** While the full-rank MLP successfully preserves the linear latent space, directly predicting a full $N_z \times N_z$ matrix from physical conditions heavily overfits the training regime. This severely degrades extrapolation performance (e.g., $Inc_{low}$ MSE spikes to $10.4 \times 10^{-4}$). LoRA acts as a critical structural regularizer by anchoring the dynamics to a globally stable base matrix $\mathbf{K}_0$.
2. **Cosine scheduling stabilizes chaotic rollouts:** On the highly chaotic Transonic dataset, the Cosine schedule reduces the 240-step $Tra_{long}$ MSE from $17.0$ to $14.9$. By heavily penalizing early-step errors, it enforces strict local phase alignment before optimizing for global asymptotic stability.

| Conditioning | Weighting | $Inc_{low}$ MSE ($\times 10^{-4}$) | $Tra_{ext}$ MSE ($\times 10^{-3}$) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- |
| **LoRA (Proposed)** | **Cosine** | **1.3 ± 1.7** | **2.2 ± 0.9** | **14.9 ± 1.3** |
| LoRA | Uniform | 1.3 ± 1.7 | 2.5 ± 0.8 | 17.0 ± 2.3 |
| MLP (Full-Rank) | Cosine | 10.4 ± 17.5 | 3.6 ± 1.0 | 15.1 ± 1.9 |
| Base (Unconditional) | Cosine | 116.5 ± 31.0 | 13.9 ± 0.8 | 18.1 ± 1.7 |

---

## 3. Continuous-Time "Consistent Koopman" (Azencot) Ablation
**Addressed to:** *Reviewer z2Gs (Comparison to Azencot et al., 2020), Reviewer RCnK (History encoder justification).*

We formalize our latent consistency loss as the exact continuous-time equivalent of the discrete Consistent Koopman Autoencoder (Azencot et al., 2020). We ablated this specific constraint, alongside our physics regularizers and dual-stream history encoder.

**Key Takeaway:** Enforcing the Azencot forward-backward consistency in continuous time, combined with our history encoder (Takens' delay embedding proxy), strictly improves performance across all trajectory forecasting tasks.

| Method | $Tra_{ext}$ MSE ($\times 10^{-3}$) | $Tra_{int}$ MSE ($\times 10^{-3}$) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- |
| **Proposed KAE (Full)** | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| w/o Physics | 2.5 ± 0.5 | 5.6 ± 2.3 | 19.3 ± 1.2 |
| w/o Azencot Consistency | 2.6 ± 0.6 | 5.8 ± 2.7 | 18.5 ± 1.4 |
| w/o History Encoder | 2.6 ± 0.8 | 5.9 ± 3.5 | 18.6 ± 0.5 |

---

## 4. ODE Solver Stability & Adaptive Steps (Dopri5)
**Addressed to:** *Reviewer B4CM (Request to evaluate adaptive-step ODE solvers like Dopri5).*

We evaluated our learned continuous-time generator across 7 different ODE solvers (including adaptive `Dopri5` and `Adaptive Heun`) across step sizes ranging from $\Delta t = 0.05s$ to extreme jumps of $\Delta t = 1.00s$.

**Key Takeaway:** The learned latent dynamics are highly robust to the choice of solver. `Dopri5` perfectly matches `RK4` performance across standard intervals. Crucially, as the step size increases to massive bounds ($\Delta t = 1.00s$), weak solvers like `Euler` and `Midpoint` diverge catastrophically, while `RK4` and adaptive `Dopri5` maintain strict stability.

| Step Size | Solver | $Tra_{ext}$ MSE | $Tra_{int}$ MSE | $Tra_{long}$ MSE |
| :--- | :--- | :--- | :--- | :--- |
| **$\Delta t = 0.10s$** | **RK4** | **2.1 ± 1.1** | **5.5 ± 3.3** | **14.6 ± 0.9** |
| | Dopri5 (Adaptive) | 2.0 ± 1.1 | 5.5 ± 3.3 | 14.6 ± 0.9 |
| | Euler | 2.9 ± 1.2 | 6.0 ± 3.3 | 19.0 ± 1.3 |
| **$\Delta t = 1.00s$** | **RK4** | **6.0 ± 0.5** | **9.7 ± 2.4** | **15.1 ± 2.0** |
| | Dopri5 (Adaptive) | 8.5 ± 1.1 | 8.1 ± 3.5 | 14.7 ± 1.4 |
| | Euler | 13.2 ± 0.6 | 20.0 ± 2.3 | *7.14e13 (Diverged)* |

---

## 5. The "Closure Problem": Spectral Bias & 1000-Step Stability
**Addressed to:** *Reviewer Ge7F (Inquiry regarding closure errors when truncating chaotic features into a linear operator).*

We rigorously analyze the spectral properties of our learned operator. The Continuous KAE acts as a **physical low-pass filter**. 

* **Spatial Domain:** The KAE exhibits a steeper energy drop-off at high wavenumbers compared to the stochastic baseline, mathematically smoothing out fine-scale turbulent textures.
* **Temporal Domain:** This high-frequency truncation allows the KAE to accurately lock onto the dominant macro-scale vortex shedding frequencies without interference from chaotic micro-structures.

![Spectral Analysis](figures/fig7_premultiplied_grid_longer_250.pdf)
*Figure 1: Temporal (left) and Spatial (right) frequency analysis. The KAE captures dominant dynamics while suppressing high-frequency noise.*

### Extreme 1000-Step Rollout Stability
To prove the value of this spectral bias, we subjected the models to an extreme 1000-step autoregressive stress test. 
* The **unconstrained diffusion baseline (ACDM)** hallucinates and compounds high-frequency errors until the physical structure collapses completely into numerical noise.
* The **Continuous KAE**, bounded by the dissipative eigenvalues of its linear operator, degrades gracefully into a stable, physically consistent limit cycle.

<p align="center">
  <img src="figures/spatial_correlation_longer.pdf" width="48%" alt="Spatial Correlation 1000 Steps" />
  <img src="figures/l2_error_rollout_longer.pdf" width="48%" alt="L2 Error 1000 Steps" />
</p>
*Figure 2: Spatial correlation (left) and Relative L2 Error (right) over 1000 steps. The KAE remains strictly bounded.*

---

## 6. Spatial Error Localization (Difference Maps)
**Addressed to:** *Reviewer z2Gs (Request for absolute spatial difference maps).*

To visually isolate where the models drift, we plot the absolute spatial difference ($|\text{Prediction} - \text{Ground Truth}|$). Because the KAE enforces smooth global structural alignment, its errors are deterministic and tightly localized along sharp spatial discontinuities (e.g., transonic shock fronts). The diffusion baseline exhibits diffuse, widespread stochastic noise across the entire fluid domain.

### Transonic Interpolation ($Tra_{int}$) & Extrapolation ($Tra_{ext}$)
<p align="center">
  <img src="figures/difference_maps_interp.pdf" width="48%" alt="Diff Map Interp" />
  <img src="figures/difference_maps_extrap.pdf" width="48%" alt="Diff Map Extrap" />
</p>
*Figure 3: Absolute error distribution. KAE errors are tightly bounded to the shock waves, preserving the wake structure.*

---
*End of Supplementary Rebuttal Appendix.*
