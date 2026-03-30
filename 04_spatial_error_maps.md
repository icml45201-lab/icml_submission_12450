# High-Resolution Spatial Error & Distribution Analysis

**Reviewer Comments Addressed:**
* **z2Gs**: Plot absolute error fields instead of raw fields to visualize where the model drifts or fails.

**Response & Empirical Evidence:**

**1. Spatial Error Morphology (Addressing z2Gs)**
* **Experiment**: We computed absolute spatial difference fields ($|\text{Prediction} - \text{Ground Truth}|$) to isolate failure points in the Transonic regime.
* **Result**: [See Figure: difference_maps_interp.png / difference_maps_extrap.png]. The Continuous KAE produces errors tightly localized along sharp spatial discontinuities (the shock fronts), maintaining a clean background fluid domain. The diffusion baseline (ACDM) exhibits widespread stochastic noise across the entire fluid domain over extended rollouts.

**2. Error Distribution Analysis (Addressing z2Gs)**
* **Experiment**: We analyzed the statistical distribution of field-wise MSE across all test trajectories via violin plots.
* **Result**: [See Figure: highRey_violin_mse_distribution.png]. In the highly turbulent High-Reynolds regime, ACDM exhibits heavy-tailed error distributions, corresponding to severe prediction failures on specific trajectories. The Continuous KAE maintains a compressed error distribution with bounded variance.
