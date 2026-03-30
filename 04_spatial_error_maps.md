# High-Resolution Spatial Error & Distribution Analysis

**Reviewer Comments Addressed:**
* **z2Gs**: Plot absolute error fields instead of raw fields to visualize where the model drifts or fails.


While raw aggregate MSE metrics can occasionally favor stochastic models in chaotic flows, relying solely on spatial averages completely obscures the true physical morphology of the predictive errors. To definitively answer Reviewer z2Gs's inquiry, we computed absolute spatial difference fields ($|\text{Prediction} - \text{Ground Truth}|$) and conducted a fine-grained distributional analysis. 

The visual and statistical evidence confirms a fundamental dichotomy: the Continuous KAE produces localized, deterministic boundary errors, whereas the generative baseline suffers from diffuse stochastic drift and catastrophic heavy-tailed failures.

### Observation A: Spatial Error Morphology (Transonic Regimes)
In the highly chaotic Transonic dataset, shock waves interact violently with the vortex street. We plotted the absolute error maps to isolate exactly where the surrogate models fail to capture this physics.

* **The Continuous KAE Signature:** Because our model enforces smooth, globally consistent structural alignment, its errors are entirely deterministic. As seen in Figures 9 and 10, KAE errors are tightly bounded and localized almost exclusively along sharp spatial discontinuities (e.g., the precise boundaries of the transonic shock fronts). The background fluid domain remains pristine.
* **The Diffusion Signature (ACDM):** In stark contrast, the autoregressive diffusion baseline exhibits diffuse, widespread stochastic noise across the entire fluid domain. It fails to preserve global phase coherence, resulting in a "salt-and-pepper" error distribution that physically degrades the entire flow field over long rollouts.

<p align="center">
  <img src="figures/difference_maps_interp.png" width="48%" alt="Diff Map Interp" />
  <img src="figures/difference_maps_extrap.png" width="48%" alt="Diff Map Extrap" />
  <b>Figure 9:</b> Absolute error distribution in Transonic Interpolation (Left) and Extrapolation (Right). KAE errors are concentrated precisely at the sharp shock fronts, whereas ACDM exhibits broad, unphysical spatial noise.
</p>

<p align="center">
  <img src="figures/difference_maps_longer.png" width="90%" alt="Difference Maps Longer" />
</p>
<p align="center">
  <b>Figure 10:</b> Spatial error distribution in the extreme long-rollout regime ($Tra\_{\text{long}}$). The KAE maintains structural stability with tightly localized errors, while ACDM's stochastic noise pollutes the entire wake.
</p>

### Observation B: Distributional Robustness & Heavy Tails (Incompressible Regimes)
To understand the reliability of the models across different turbulence levels, we analyzed the statistical distribution of the field-wise MSE across all test trajectories.

* **Heavy-Tailed Catastrophic Failures:** As shown in the violin plots (Figure 11), both models perform comparably at benign, low Reynolds numbers. However, the highly turbulent High-Reynolds regime exposes the fragility of unconstrained generative sampling. ACDM exhibits pronounced, heavy-tailed error distributions—these long upper tails correspond to severe, catastrophic prediction failures on specific trajectories. 
* **Strictly Controlled Variance:** Conversely, the Continuous KAE maintains a tightly compressed error distribution with strictly controlled variance. As shown in the temporal tracking (Figure 12), ACDM suffers from accelerated compounding error growth over time, while the KAE maintains mathematically stable error scaling. This proves our method possesses far superior robustness for critical engineering applications where worst-case failure bounds must be guaranteed.

<p align="center">
  <img src="figures/lowRey_violin_mse_distribution.png" width="48%" alt="Low Rey Violin" />
  <img src="figures/highRey_violin_mse_distribution.png" width="48%" alt="High Rey Violin" />
  <b>Figure 11:</b> Error distributions under Low (left) and High (right) Reynolds number regimes. Note the dangerous heavy tails in the stochastic baseline at higher Reynolds numbers, contrasting with the KAE's bounded variance.

</p>

<p align="center">
  <img src="figures/highRey_temporal_mse_per_field.png" width="48%" alt="High Rey Temporal" />
  <img src="figures/highRey_line_mse_vs_Re_fieldwise.png" width="48%" alt="High Rey Fieldwise" />
  <b>Figure 12:</b> Temporal evolution of field-wise MSE (left) and MSE scaling vs. Reynolds number (right). The Continuous KAE suppresses compounding errors, maintaining stable trajectory growth over long horizons.
