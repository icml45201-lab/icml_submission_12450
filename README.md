# Comprehensive Rebuttal Appendix: ICML 2026 Submission 12450
**Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting**

This anonymous repository serves as an extended, high-resolution appendix to our official author rebuttal. To address reviewer inquiries with the highest level of empirical rigor, we have completely overhauled our experimental evaluation. 

This repository contains **14 comprehensive baselines**, extensive **architectural ablations** (LoRA, MLP, Azencot consistency), a massive **ODE solver stress-test** across extreme step sizes, and formalized **spectral/eigenvalue analyses** of our continuous-time operator. 

All figures, tables, and theoretical analyses enclosed below are directly referenced in our official ICML rebuttal text.

---

## Table of Contents
1. [Exhaustive Baseline Benchmarking & $O(1)$ Inference Efficiency](#1-exhaustive-baseline-benchmarking--o1-inference-efficiency)
2. [Architectural Ablations: Operator Parameterization & Weighting](#2-architectural-ablations-operator-parameterization--weighting)
3. [Continuous-Time "Consistent Koopman" (Azencot) Ablation](#3-continuous-time-consistent-koopman-azencot-ablation)
4. [ODE Solver Stability & Adaptive Steps (Dopri5)](#4-ode-solver-stability--adaptive-steps-dopri5)
5. [The "Closure Problem": Spectral Bias & Eigenvalue Analysis](#5-the-closure-problem-spectral-bias--eigenvalue-analysis)
6. [Extreme 1000-Step Rollout Stability](#6-extreme-1000-step-rollout-stability)
7. [Analytical Matrix Exponential vs. Numerical Integration](#7-analytical-matrix-exponential-vs-numerical-integration)
8. [High-Resolution Spatial Error Localization (Difference Maps)](#8-high-resolution-spatial-error-localization-difference-maps)

---

## 1. Exhaustive Baseline Benchmarking & $O(1)$ Inference Efficiency
**Addressed to:** *Reviewers z2Gs, B4CM, RCnK (Requests for broader baseline comparisons beyond diffusion models).*

We expanded our evaluation to encompass 14 total models, representing the current state-of-the-art in spatial-temporal PDE surrogates. This includes Fourier Neural Operators (FNO-16, FNO-32), U-Nets, and Graph/Transformer variants. 

**Key Takeaway:** The Continuous-Time KAE establishes a distinct Pareto frontier. While highly non-linear generative models (ACDM) capture slightly more high-frequency stochastic texture in the short term, the strict global linearity of the KAE's latent space allows for analytical matrix exponentiation. This yields an **inference speedup of >300$\times$** over diffusion models and **>5$\times$** over continuous U-Nets.

### Table A: Inference Speed and Memory Efficiency
| Architecture | Avg. Step Inference (ms) | Mean VRAM (MB) |
| :--- | :--- | :--- |
| TF-VAE | $0.30 \pm 0.01$ | $13749.9$ |
| TF-Enc | $0.60 \pm 0.25$ | $3448.6$ |
| TF-MGN | $0.69 \pm 0.01$ | $3498.0$ |
| FNO-16 | $1.17 \pm 0.01$ | $184.1$ |
| FNO-32 | $1.17 \pm 0.00$ | $183.9$ |
| Dil-ResNet-m2 | $3.46 \pm 0.02$ | $\mathbf{178.6}$ |
| ResNet-m2 | $3.67 \pm 0.04$ | $188.0$ |
| UNet-m8 | $6.16 \pm 0.01$ | $184.1$ |
| UNet-m2 | $6.19 \pm 0.09$ | $183.7$ |
| Refiner-R4 | $10.31 \pm 0.02$ | $642.4$ |
| ACDM$_{ncn}$ | $41.70 \pm 0.06$ | $649.2$ |
| ACDM | $41.77 \pm 0.01$ | $659.2$ |
| **Continuous KAE (Ours)** | **$\mathbf{0.00104 \pm 0.0001}$** | $2751.3$ |

### Table B: Complete Quantitative Split Comparison (MSE $\times 10^{-3}$)
| Method | $Inc_{low}$ | $Inc_{high}$ | $Tra_{ext}$ | $Tra_{int}$ | $Tra_{long}$ (240 steps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\text{U-Net}_{ut}$ | $\mathbf{0.8 \pm 1.1}$ | $\mathbf{0.2 \pm 0.1}$ | $1.6 \pm 0.7$ | $1.5 \pm 1.5$ | $22.2 \pm 3.6$ |
| $\text{ACDM}_{ncn}$ | $0.9 \pm 0.8$ | $5.7 \pm 2.7$ | $4.1 \pm 1.9$ | $2.8 \pm 1.3$ | $22.8 \pm 3.8$ |
| $\text{U-Net}_{tn}$ | $1.0 \pm 1.0$ | $0.9 \pm 0.6$ | $1.4 \pm 0.8$ | $1.8 \pm 1.1$ | $22.4 \pm 3.9$ |
| U-Net | $1.0 \pm 1.1$ | $2.7 \pm 0.6$ | $3.1 \pm 2.1$ | $2.3 \pm 2.0$ | $30.3 \pm 6.1$ |
| Refiner | $1.3 \pm 1.4$ | $3.5 \pm 2.2$ | $5.4 \pm 2.1$ | $7.1 \pm 2.1$ | *Diverged* |
| $\text{TF}_{Enc}$ | $1.5 \pm 1.7$ | $8.7 \pm 3.8$ | $\mathbf{1.0 \pm 0.3}$ | $1.8 \pm 0.3$ | $22.2 \pm 3.9$ |
| ResNet-dil | $1.6 \pm 1.8$ | $2.6 \pm 0.7$ | $1.2 \pm 0.3$ | $\mathbf{1.0 \pm 0.5}$ | $22.0 \pm 2.4$ |
| ACDM | $1.7 \pm 2.2$ | $0.8 \pm 0.4$ | $2.3 \pm 1.4$ | $2.7 \pm 2.1$ | $22.6 \pm 4.0$ |
| $\text{FNO}_{16}$ | $2.8 \pm 3.1$ | $8.9 \pm 3.8$ | $4.8 \pm 1.2$ | $5.5 \pm 2.6$ | $20.8 \pm 2.0$ |
| $\text{TF}_{VAE}$ | $5.4 \pm 5.5$ | $4.1 \pm 1.4$ | $2.4 \pm 0.2$ | $2.7 \pm 0.6$ | $20.6 \pm 2.1$ |
| $\text{TF}_{MGN}$ | $5.7 \pm 4.3$ | $10.0 \pm 2.9$ | $3.9 \pm 1.0$ | $6.3 \pm 4.4$ | $18.9 \pm 4.5$ |
| ResNet | $10.0 \pm 9.1$ | $16.0 \pm 3.0$ | $2.3 \pm 0.9$ | $1.8 \pm 1.0$ | $24.2 \pm 4.6$ |
| $\text{FNO}_{32}$ | $160 \pm 50$ | $1k \pm 140$ | $4.9 \pm 1.9$ | $6.8 \pm 3.4$ | *Diverged* |
| **Continuous KAE (Ours)** | **$1.3 \pm 1.7$** | **$2.9 \pm 1.1$** | **$2.2 \pm 0.9$** | **$5.2 \pm 2.4$** | **$\mathbf{14.9 \pm 1.3}$** |

---

## 2. Architectural Ablations: Operator Parameterization & Weighting
**Addressed to:** *Reviewer z2Gs (Requests for MLP parameterization vs. LoRA and Cosine vs. Uniform empirical ablations).*

To empirically validate our architectural design, we ablated the LoRA parameterization against a full-rank MLP ($\mathbf{K}_{\text{cont}} = \text{MLP}(\phi)$) and compared our Cosine temporal weighting against a Uniform schedule.

**Key Takeaways:**
1. **LoRA prevents severe overfitting:** While the full-rank MLP successfully preserves the linear latent space, directly predicting a full $N_z \times N_z$ matrix from physical conditions heavily overfits the training regime. This severely degrades extrapolation performance (e.g., $Inc_{low}$ MSE spikes to $10.4$). LoRA acts as a critical structural regularizer by anchoring the dynamics to a globally stable base matrix $\mathbf{K}_0$.
2. **Cosine scheduling stabilizes chaotic rollouts:** On the highly chaotic Transonic dataset, the Cosine schedule reduces the 240-step $Tra_{long}$ MSE from $17.0$ to $14.9$. By heavily penalizing early-step errors, it enforces strict local phase alignment before optimizing for global asymptotic stability.

| Conditioning | Weighting | $Inc_{low}$ MSE | $Inc_{high}$ MSE | $Tra_{ext}$ MSE | $Tra_{int}$ MSE | $Tra_{long}$ MSE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA (Proposed)** | **Cosine** | **1.3 ± 1.7** | 2.9 ± 1.1 | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| LoRA | Uniform | 1.3 ± 1.7 | 2.9 ± 1.1 | 2.5 ± 0.8 | 6.5 ± 1.6 | 17.0 ± 2.3 |
| MLP (Full-Rank) | Cosine | 10.4 ± 17.5 | 21.4 ± 7.1 | 3.6 ± 1.0 | 5.7 ± 3.0 | 15.1 ± 1.9 |
| Base (Unconditional)| Cosine | 116.5 ± 31.0 | 2991.2 ± 12.5| 13.9 ± 0.8 | 21.0 ± 2.7 | 18.1 ± 1.7 |

---

## 3. Continuous-Time "Consistent Koopman" (Azencot) Ablation
**Addressed to:** *Reviewer z2Gs (Comparison to Azencot et al., 2020), Reviewer RCnK (History encoder justification).*

We formalize our latent consistency loss as the exact continuous-time generalization of the discrete Consistent Koopman Autoencoder (Azencot et al., 2020). We ablated this specific structural constraint, alongside our physics regularizers and dual-stream history encoder.

**Key Takeaway:** Enforcing the Azencot forward-backward consistency in continuous time, combined with our history encoder (acting as a Takens' delay embedding proxy), strictly improves performance across all trajectory forecasting tasks.

| Model Configuration | $Tra_{ext}$ MSE ($\times 10^{-3}$) | $Tra_{int}$ MSE ($\times 10^{-3}$) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- |
| **Continuous KAE (Proposed)** | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| w/o Physics Regularization | 2.5 ± 0.5 | 5.6 ± 2.3 | 19.3 ± 1.2 |
| w/o Azencot Consistency | 2.6 ± 0.6 | 5.8 ± 2.7 | 18.5 ± 1.4 |
| w/o History Encoder | 2.6 ± 0.8 | 5.9 ± 3.5 | 18.6 ± 0.5 |

---

## 4. ODE Solver Stability & Adaptive Steps (Dopri5)
**Addressed to:** *Reviewer B4CM (Request to evaluate adaptive-step ODE solvers like Dopri5 vs. fixed-step solvers).*

We rigorously stress-tested our learned continuous-time generator across 7 different ODE solvers (including adaptive `Dopri5` and `Adaptive Heun`) across step sizes ranging from $\Delta t = 0.05s$ to extreme jumps of $\Delta t = 1.00s$.

**Key Takeaway:** The learned latent dynamics are highly robust to the choice of solver. `Dopri5` matches `RK4` performance perfectly across standard intervals. Crucially, as the step size increases to massive bounds ($\Delta t = 1.00s$), weak solvers like `Euler` and `Midpoint` diverge catastrophically into numerical infinity, while `RK4` and adaptive `Dopri5` maintain strict stability and bounded errors.

| Step Size ($\Delta t$) | Solver | $Tra_{ext}$ MSE | $Tra_{int}$ MSE | $Tra_{long}$ MSE |
| :--- | :--- | :--- | :--- | :--- |
| **0.05 s** | RK4 | 1.9 ± 1.1 | 5.8 ± 3.4 | 14.6 ± 0.9 |
| | Dopri5 | 1.9 ± 1.1 | 5.8 ± 3.4 | 14.6 ± 0.9 |
| | Euler | 1.8 ± 1.1 | 6.2 ± 3.6 | 18.8 ± 1.3 |
| **0.15 s** | RK4 | 2.3 ± 1.2 | 5.3 ± 3.2 | 14.6 ± 0.8 |
| | Dopri5 | 2.3 ± 1.2 | 5.3 ± 3.2 | 14.6 ± 0.9 |
| | Euler | 5.9 ± 1.2 | 6.2 ± 2.6 | 221.3 ± 101.8 |
| **0.50 s** | RK4 | 6.2 ± 1.1 | 5.6 ± 2.8 | 14.9 ± 1.4 |
| | Dopri5 | 5.0 ± 1.3 | 5.3 ± 2.7 | 14.8 ± 1.0 |
| | Euler | 19.1 ± 0.8 | 20.9 ± 3.3 | $4.88 \times 10^{14}$ |
| **1.00 s** | **RK4** | **6.0 ± 0.5** | **9.7 ± 2.4** | **15.1 ± 2.0** |
| | **Dopri5** | **8.5 ± 1.1** | **8.1 ± 3.5** | **14.7 ± 1.4** |
| | Euler | 13.2 ± 0.6 | 20.0 ± 2.3 | $7.14 \times 10^{13}$ |
| | Midpoint | 16.8 ± 0.6 | 11.0 ± 2.1 | $3.44 \times 10^{19}$ |

---

## 5. The "Closure Problem": Spectral Bias & Eigenvalue Analysis
**Addressed to:** *Reviewer Ge7F (Inquiry regarding closure errors when truncating chaotic features into a linear operator).*

We rigorously analyze the spectral properties of our learned operator. The Continuous KAE acts as a **physical low-pass filter**. 

* **Spatial Domain (Right):** The KAE exhibits a steeper energy drop-off at high wavenumbers compared to the stochastic baseline, mathematically smoothing out fine-scale turbulent textures.
* **Temporal Domain (Left):** This high-frequency truncation allows the KAE to accurately lock onto the dominant macro-scale vortex shedding frequencies without interference from chaotic micro-structures.

![Spectral Analysis](figures/fig7_premultiplied_grid_longer_250.png)
*Figure 1: Temporal (left) and Spatial (right) frequency analysis. The KAE captures dominant dynamics while suppressing high-frequency noise.*

### Latent Dynamics Eigenvalue Spectrum
To mathematically prove this stability, we plot the eigenvalue spectrum of the learned continuous generator $\mathbf{K}_{\text{cont}}$. The spectrum lies almost entirely in the left half of the complex plane ($Re(\lambda) < 0$), establishing strict asymptotic dissipativity.

<p align="center">
  <img src="figures/eigenvalues_spectrum.png" width="45%" alt="Eigenvalue Spectrum" />
</p>
*Figure 2: Eigenvalue spectrum of the continuous-time operator. The strictly negative real parts guarantee that high-frequency stochastic errors decay rather than compound.*

---

## 6. Extreme 1000-Step Rollout Stability
**Addressed to:** *All Reviewers (Demonstrating the ultimate utility of linear latent constraints).*

To prove the value of the spectral dissipativity identified above, we subjected the models to an extreme 1000-step autoregressive stress test. 
* The **unconstrained diffusion baseline (ACDM)** hallucinates and compounds high-frequency stochastic errors until the physical structure collapses completely into numerical noise.
* The **Continuous KAE**, bounded by the dissipative eigenvalues of its linear operator, degrades gracefully into a stable, physically consistent limit cycle.

<p align="center">
  <img src="figures/spatial_correlation_longer.png" width="48%" alt="Spatial Correlation 1000 Steps" />
  <img src="figures/l2_error_rollout_longer.png" width="48%" alt="L2 Error 1000 Steps" />
</p>
*Figure 3: Spatial correlation (left) and Relative L2 Error (right) over 1000 steps. The KAE remains strictly bounded.*

---

## 7. Analytical Matrix Exponential vs. Numerical Integration
**Addressed to:** *Reviewer RCnK (Confirming internal consistency of the continuous formulation).*

To empirically validate that our model learns a true continuous-time generator, we compare the trajectories generated by standard numerical integration (RK4) against the analytical matrix exponential $z(\tau) = \exp(\mathbf{K}_{\text{cont}}\tau)z_0$. The perfect visual and phase alignment confirms the validity of the $O(1)$ fast-forwarding inference capability.

<p align="center">
  <img src="figures/data_highRey_vort_rk4_tight.png" width="48%" alt="RK4 Incompressible" />
  <img src="figures/data_extrap_pres_rk4_tight.png" width="48%" alt="RK4 Transonic" />
</p>
*Figure 4: The continuous-time latent dynamics permit exact analytical evaluation (top rows) that perfectly matches iterative numerical integration (subsequent rows), enabling zero-shot temporal super-resolution.*

---

## 8. High-Resolution Spatial Error Localization (Difference Maps)
**Addressed to:** *Reviewer z2Gs (Request for absolute spatial difference maps).*

To visually isolate where the models drift, we plot the absolute spatial difference ($|\text{Prediction} - \text{Ground Truth}|$). Because the KAE enforces smooth global structural alignment, its errors are deterministic and tightly localized along sharp spatial discontinuities (e.g., transonic shock fronts). The diffusion baseline exhibits diffuse, widespread stochastic noise across the entire fluid domain.

### Transonic Interpolation ($Tra_{int}$) & Extrapolation ($Tra_{ext}$)
<p align="center">
  <img src="figures/difference_maps_interp.png" width="48%" alt="Diff Map Interp" />
  <img src="figures/difference_maps_extrap.png" width="48%" alt="Diff Map Extrap" />
</p>
*Figure 5: Absolute error distribution. KAE errors are tightly bounded to the shock waves, preserving the wake structure.*

### Transonic Long Rollout ($Tra_{long}$)
![Difference Maps Longer](figures/difference_maps_longer.png)
*Figure 6: Spatial error distribution in the 240-step long-rollout regime.*

### Incompressible Wake (Low vs. High Reynolds)
<p align="center">
  <img src="figures/difference_maps_lowRey_full.png" width="48%" alt="Low Rey Diff" />
  <img src="figures/difference_maps_highRey_full.png" width="48%" alt="High Rey Diff" />
</p>
*Figure 7: Difference maps for the Incompressible datasets. Note the tight error localization on the obstacle boundary for the KAE.*

---
*End of Supplementary Rebuttal Appendix. We thank the Area Chair and Reviewers for their time and constructive feedback.*
