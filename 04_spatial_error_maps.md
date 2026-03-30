# Spatial Error Maps & Distribution Analysis

This document provides the absolute spatial difference fields ($|\text{Prediction} - \text{Ground Truth}|$) and error distribution analysis requested by reviewers to visualize exact failure modes and model drift.

### 1. Spatial Error Morphology
**Addressed to:** Reviewer z2Gs (Plot absolute error fields instead of raw fields)

**Experiment:** We computed the absolute spatial difference fields for the Transonic dataset to isolate where the surrogate models fail to capture complex physics (e.g., shock wave interactions).

**Observations:**
* **Continuous KAE (Localized Errors):** KAE errors are tightly bounded and localized almost exclusively along sharp spatial discontinuities (the transonic shock fronts). The background fluid domain remains clean, indicating stable structural alignment.
* **ACDM (Diffuse Errors):** The autoregressive diffusion baseline exhibits widespread stochastic noise across the entire fluid domain. It fails to preserve global phase coherence, resulting in a scattered "salt-and-pepper" error distribution.

<br>

<p align="center">
  <img src="figures/difference_maps_interp.png" width="48%" alt="Diff Map Interp" />
  <img src="figures/difference_maps_extrap.png" width="48%" alt="Diff Map Extrap" />
</p>
<p align="center">
  <b>Figure 1:</b> Absolute error distribution in Transonic Interpolation (Left) and Extrapolation (Right). KAE errors concentrate at shock fronts; ACDM exhibits broader spatial noise.
</p>

<p align="center">
  <img src="figures/difference_maps_longer.png" width="90%" alt="Difference Maps Longer" />
</p>
<p align="center">
  <b>Figure 2:</b> Spatial error distribution in the 240-step rollout regime ($Tra_{long}$). KAE maintains structural bounds, while ACDM introduces stochastic noise across the wake.
</p>

---

### 2. Error Distribution & Variance Tracking
**Addressed to:** Reviewer z2Gs (Visualizing model failure and drift)

**Experiment:** We analyzed the statistical distribution of the field-wise MSE across all test trajectories in both low and high Reynolds number (Incompressible) regimes to evaluate reliability.

**Observations:**
* **Distribution Tails (Figure 3):** Both models perform comparably at low Reynolds numbers. In the highly turbulent High-Reynolds regime, ACDM exhibits a heavy-tailed error distribution, indicating larger prediction failures on specific trajectories. The KAE maintains a tightly compressed distribution with strictly controlled variance.
* **Temporal Error Growth (Figure 4):** Tracking MSE over time reveals that ACDM suffers from compounding error growth, while the KAE maintains stable, bounded error scaling across the rollout horizon.

<br>

<p align="center">
  <img src="figures/lowRey_violin_mse_distribution.png" width="48%" alt="Low Rey Violin" />
  <img src="figures/highRey_violin_mse_distribution.png" width="48%" alt="High Rey Violin" />
</p>
<p align="center">
  <b>Figure 3:</b> Error distributions under Low (left) and High (right) Reynolds number regimes. ACDM shows heavier error tails at higher Reynolds numbers, whereas KAE maintains bounded variance.
</p>

<p align="center">
  <img src="figures/highRey_temporal_mse_per_field.png" width="48%" alt="High Rey Temporal" />
  <img src="figures/highRey_line_mse_vs_Re_fieldwise.png" width="48%" alt="High Rey Fieldwise" />
</p>
<p align="center">
  <b>Figure 4:</b> Temporal evolution of field-wise MSE (left) and MSE scaling vs. Reynolds number (right). The Continuous KAE suppresses compounding errors over the temporal horizon.
</p>
