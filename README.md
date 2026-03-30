# Submission 12450: Continuous-Time Koopman Autoencoder

This anonymous repository contains supplementary experiments, ablation studies, and high-resolution visualizations prepared in response to reviewer feedback for Submission 12450. 

Our primary objective in this rebuttal phase was to expand our empirical baselines, rigorously stress-test the structural stability of the learned continuous-time generator, and provide fine-grained visualizations of the model's failure modes compared to generative baselines.

## Repository Contents

Please navigate to the detailed markdown files below for full experimental setups, quantitative tables, and visual proofs:

* **[1. Baseline Comparisons & Inference Efficiency](./1_baseline_comparisons.md)**
  * Expanded benchmarking against 13 spatial-temporal surrogate models (Diffusion, FNO, U-Net, ResNet, Transformer).
  * Profiling of the $O(1)$ inference speedup (>40,000x faster than ACDM).
* **[2. Ablation Studies: Architecture & Loss](./2_ablation_studies.md)**
  * Empirical justification for LoRA parameterization vs. Full-Rank MLP.
  * Isolation of the temporal cosine weighting schedule and Takens' delay embedding (history encoder).
  * Validation of continuous-time invertibility constraints.
* **[3. ODE Solvers & Temporal Generalization](./3_solver_generalization.md)**
  * Stress-testing the latent ODE across 7 numerical integrators (including adaptive `Dopri5`).
  * Demonstration of zero-shot temporal super-resolution and irregular sampling ($\Delta t = 0.05$s to $1.00$s).
* **[4. Long-Horizon Stability & Spectral Analysis](./4_spectral_long_horizon.md)**
  * FFT spectral analysis of the latent closure problem (spatial low-pass filtering vs. phase locking).
  * Extreme 1000-step autoregressive rollout comparisons (Limit cycle stability vs. Diffusion divergence).
* **[5. Spatial Error Maps & Distribution Analysis](./5_error_distributions.md)**
  * High-resolution absolute error difference fields ($|\hat{y} - y|$).
  * Statistical evaluation of heavy-tailed failure distributions across turbulence regimes.

---

## Executive Summary of New Findings

For convenience, the core empirical findings established in these new documents are summarized below:

1. **State-of-the-Art Inference Efficiency:** By strictly enforcing a linear continuous-time latent space, the model bypasses iterative solvers at inference via analytical matrix exponentiation ($z(\tau)=\exp(\mathbf{K}_{\mathrm{cont}}\tau)z_0$). This yields a measured per-step inference time of **0.00104 ms**, a **>40,000x** speedup over the autoregressive diffusion baseline (ACDM) and a **>5,000x** speedup over standard continuous U-Nets.
2. **Infinite-Horizon Bounded Stability:** Eigenvalue spectrum analysis proves the learned continuous operator is strictly dissipative ($Re(\lambda) < 0$). In extreme 1000-step stress tests, the KAE successfully degrades into a stable, physically consistent limit cycle, whereas unconstrained generative models (ACDM) and neural operators (FNO-32) suffer from compounding exponential error and collapse into structural noise.
3. **Zero-Shot Temporal Interpolation:** Numerical integration (`RK4`, `Dopri5`) perfectly aligns with the analytical closed-form solution. This allows the model to be evaluated zero-shot at untrained, irregular temporal resolutions (e.g., interpolating at $\Delta t=0.05$s from $\Delta t=0.10$s training data) without retraining or performance degradation.
4. **Structural Regularization:** Predicting full-rank matrices ($O(N_z^2)$) causes catastrophic out-of-distribution overfitting in physical forecasting. Ablations confirm that the proposed Low-Rank Adaptation (LoRA) bottleneck is strictly necessary as a structural regularizer to achieve out-of-distribution physical generalization (e.g., varying Reynolds/Mach numbers).
