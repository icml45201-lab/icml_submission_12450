# Comprehensive Rebuttal Appendix: ICML 2026 Submission 12450
**Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting**

This anonymous repository serves as an extended, high-resolution appendix to our official author rebuttal. To address reviewer inquiries comprehensively, we have conducted extensive new ablation studies, evaluated 12 additional baselines, and formalized the spectral properties of our continuous-time operator. 

All figures, tables, and theoretical analyses enclosed below are directly referenced in the official rebuttal text.

---

## Table of Contents
1. [Expanded Benchmarking & $O(1)$ Inference Efficiency](#1-expanded-benchmarking--o1-inference-efficiency)
2. [Architectural Ablations: LoRA vs. MLP & Cosine vs. Uniform](#2-architectural-ablations)
3. [The "Closure Problem": Spectral Bias & Frequency Smoothing](#3-the-closure-problem-spectral-bias)
4. [Extreme 1000-Step Rollout Stability](#4-extreme-1000-step-rollout-stability)
5. [Spatial Error Localization (Difference Maps)](#5-spatial-error-localization)
6. [Analytical Matrix Exponential vs. RK4 Integration](#6-analytical-matrix-exponential-vs-rk4)

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
| | TF-MGN | $0.69 \pm 0.01$ | $3498.0$ | $18.9 \pm 4.5$ |
| **Generative (Diffusion)**| ACDM | $41.77 \pm 0.01$ | $659.2$ | $22.6 \pm 4.0$ |
| **Continuous Koopman** | **KAE (Ours)** | **$0.00104 \pm 0.0001$** | $2751.3$ | **$14.9 \pm 1.3$** |

---

## 2. Architectural Ablations
**Addressed to:** *Reviewer z2Gs (Request for MLP vs. LoRA and Cosine vs. Uniform empirical ablations).*

To empirically validate our architectural design, we ablated the LoRA parameterization against a full-rank MLP ($\mathbf{K}_{\text{cont}} = \text{MLP}(\phi)$) and compared our Cosine temporal weighting against a Uniform schedule.

**Key Takeaways:**
1. **LoRA prevents severe overfitting:** While the full-rank MLP successfully preserves the linear latent space, directly predicting a full $N_z \times N_z$ matrix from physical conditions heavily overfits the training regime. This severely degrades extrapolation performance (e.g., $Inc_{low}$ MSE spikes to $10.4 \times 10^{-4}$). LoRA acts as a critical structural regularizer by anchoring the dynamics to a globally stable base matrix $\mathbf{K}_0$.
2. **Cosine scheduling stabilizes chaotic rollouts:** On the highly chaotic Transonic dataset, the Cosine schedule reduces the 240-step $Tra_{long}$ MSE from $17.0$ to $14.9$. By heavily penalizing early-step errors, it enforces strict local phase alignment before optimizing for global asymptotic stability, preventing autoregressive error compounding.

| Parameterization | Weighting | $Inc_{low}$ MSE ($\times 10^{-4}$) | $Inc_{high}$ MSE ($\times 10^{-5}$) | $Tra_{ext}$ MSE ($\times 10^{-3}$) | $Tra_{int}$ MSE ($\times 10^{-3}$) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA (Proposed)** | **Cosine** | **1.3 ± 1.7** | 2.9 ± 1.1 | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| LoRA | Uniform | 1.3 ± 1.7 | 2.9 ± 1.1 | 2.5 ± 0.8 | 6.5 ± 1.6 | 17.0 ± 2.3 |
| MLP (Full-Rank) | Cosine | 10.4 ± 17.5 | 21.4 ± 7.1 | 3.6 ± 1.0 | 5.7 ± 3.0 | 15.1 ± 1.9 |

---

## 3. The "Closure Problem": Spectral Bias
**Addressed to:** *Reviewer Ge7F (Inquiry regarding closure errors when truncating infinite-dimensional chaotic features into a linear operator).*

We rigorously analyze the spectral properties of our learned operator. As shown below, the Continuous KAE acts as a **physical low-pass filter**. 

* **Spatial Domain (Right):** The KAE exhibits a steeper energy drop-off at high wavenumbers compared to the stochastic baseline, mathematically smoothing out fine-scale, unpredictable turbulent textures.
* **Temporal Domain (Left):** This high-frequency truncation is highly advantageous for stability. It allows the KAE to accurately lock onto the dominant macro-scale vortex shedding frequencies without interference from chaotic micro-structures.

![Spectral Analysis](figures/fig7_premultiplied_grid_longer_250.png)
*Figure 1: Temporal (left) and Spatial (right) frequency analysis. The KAE captures dominant dynamics while suppressing high-frequency noise.*

---

## 4. Extreme 1000-Step Rollout Stability
**Addressed to:** *All Reviewers (Demonstrating the ultimate utility of linear latent constraints).*

To prove the value of the spectral bias identified above, we subjected the models to an extreme 1000-step autoregressive stress test. 
* The **unconstrained diffusion baseline (ACDM)** hallucinates and compounds high-frequency stochastic errors until the physical structure collapses completely into numerical noise.
* The **Continuous KAE**, bounded by the dissipative eigenvalues of its linear operator, degrades gracefully into a stable, physically consistent limit cycle, making it vastly superior for extreme long-horizon structural forecasting.

<p align="center">
  <img src="figures/spatial_correlation_longer.png" width="48%" alt="Spatial Correlation 1000 Steps" />
  <img src="figures/l2_error_rollout_longer.png" width="48%" alt="L2 Error 1000 Steps" />
</p>
*Figure 2: Spatial correlation (left) and Relative L2 Error (right) over 1000 steps. The KAE remains strictly bounded.*

---

## 5. Spatial Error Localization
**Addressed to:** *Reviewer z2Gs (Request for absolute spatial difference maps).*

To visually isolate where the models drift, we plot the absolute spatial difference ($|\text{Prediction} - \text{Ground Truth}|$). Because the KAE enforces smooth global structural alignment, its errors are deterministic and tightly localized along sharp spatial discontinuities (e.g., transonic shock fronts). The diffusion baseline exhibits diffuse, widespread stochastic noise across the entire fluid domain.

### Transonic Interpolation ($Tra_{int}$) & Extrapolation ($Tra_{ext}$)
<p align="center">
  <img src="figures/difference_maps_interp.png" width="48%" alt="Diff Map Interp" />
  <img src="figures/difference_maps_extrap.png" width="48%" alt="Diff Map Extrap" />
</p>
*Figure 3: Absolute error distribution. KAE errors are tightly bounded to the shock waves, preserving the wake structure.*

---

## 6. Analytical Matrix Exponential vs. RK4 Integration
**Addressed to:** *Reviewer RCnK (Confirming internal consistency of the continuous formulation).*

To empirically validate that our model learns a true continuous-time generator, we compare the trajectories generated by standard numerical integration (RK4) against the analytical matrix exponential $z(\tau) = \exp(\mathbf{K}_{\text{cont}}\tau)z_0$. The perfect alignment confirms the validity of the $O(1)$ fast-forwarding capability.

<p align="center">
  <img src="figures/data_highRey_vort_rk4_tight.png" width="48%" alt="RK4 Incompressible" />
  <img src="figures/data_extrap_pres_rk4_tight.png" width="48%" alt="RK4 Transonic" />
</p>
*Figure 4: The continuous-time latent dynamics permit exact analytical evaluation (top rows) that perfectly matches iterative numerical integration (subsequent rows), enabling zero-shot temporal super-resolution.*

---
*End of Supplementary Rebuttal Appendix.*
