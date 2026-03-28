# Supplementary Material: ICML 2026 Submission 12450
**Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting**

This anonymous repository serves as an extended appendix to our official author rebuttal. Due to strict text-only formatting and character limits in the review portal, we provide our newly conducted ablation studies, expanded benchmarking tables, and high-resolution spatial/spectral analyses here. 

All data and figures enclosed directly address reviewer inquiries regarding theoretical boundaries, architectural design choices, and empirical baselines.

---

## 1. Operator Parameterization and Temporal Weighting Ablations
**Addressing Reviewer Feedback:** *Reviewer z2Gs requested empirical ablations justifying the use of the LoRA parameterization over a full-rank MLP, as well as the necessity of the cosine temporal weighting schedule.*

We evaluated our default architecture against a full-rank MLP parameterization ($\mathbf{K}_{\text{cont}} = \text{MLP}(\phi)$), which predicts the entire $N_z \times N_z$ generator matrix from the physical conditions. We also ablated the temporal rollout loss, comparing our decaying Cosine schedule against a Uniform weighting.

### Empirical Results
As demonstrated in Table 1, while the full-rank MLP successfully preserves the linear latent space required for $O(1)$ matrix exponentiation, it severely overfits the training regimes. This results in significantly degraded performance on extrapolation tasks (e.g., $Inc_{low}$ MSE increases from $1.3 \times 10^{-4}$ to $10.4 \times 10^{-4}$). 

Our proposed **LoRA parameterization** anchors the dynamics to a globally stable base matrix ($\mathbf{K}_0$). This acts as a powerful structural regularizer, enabling robust extrapolation across unseen Reynolds and Mach numbers while vastly reducing the parameter footprint to $O(2rN_z)$. 

Furthermore, the **Cosine weighting schedule** strictly outperforms uniform weighting on the highly chaotic Transonic dataset. By forcing the model to prioritize strict local phase alignment in early epochs, it prevents chaotic compounding errors and improves long-horizon $Tra_{long}$ MSE from $17.0 \times 10^{-3}$ to $14.9 \times 10^{-3}$.

**Table 1: Ablation Study on Operator Parameterization and Temporal Weighting**

| Parameterization | Weighting Schedule | $Inc_{low}$ MSE ($\times 10^{-4}$) | $Inc_{high}$ MSE ($\times 10^{-5}$) | $Tra_{ext}$ MSE ($\times 10^{-3}$) | $Tra_{int}$ MSE ($\times 10^{-3}$) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA (Proposed)** | **Cosine** | **1.3 ± 1.7** | 2.9 ± 1.1 | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| LoRA | Uniform | 1.3 ± 1.7 | 2.9 ± 1.1 | 2.5 ± 0.8 | 6.5 ± 1.6 | 17.0 ± 2.3 |
| MLP (Full-Rank) | Cosine | 10.4 ± 17.5 | 21.4 ± 7.1 | 3.6 ± 1.0 | 5.7 ± 3.0 | 15.1 ± 1.9 |

---

## 2. Expanded Benchmarking and Inference Efficiency
**Addressing Reviewer Feedback:** *Reviewers requested a broader comparison against standard spatial-temporal PDE surrogates beyond the autoregressive diffusion baseline (ACDM).*

We have expanded our evaluation to include 12 additional baselines, including Fourier Neural Operators (FNO-16, FNO-32), continuous U-Nets, and Graph/Transformer models. 

**Table 2: Inference Speed and Long-Horizon Stability Comparison**
The results below highlight the core Pareto frontier established by our Continuous-Time KAE. While it sacrifices some short-term stochastic expressivity compared to highly non-linear models, the strict linearity of the latent space enables analytical matrix exponentiation. This bypasses iterative ODE solvers, yielding an inference speedup of over 300$\times$ compared to diffusion models, while maintaining bounded, stable predictions over extreme horizons ($Tra_{long}$).

| Architecture Paradigm | Model | Avg. Step Inference (ms) | Mean VRAM (MB) | $Tra_{long}$ MSE ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- |
| **Spectral / Operator** | FNO-16 | 1.17 ± 0.01 | 184.1 | 20.8 ± 2.0 |
| | FNO-32 | 1.17 ± 0.00 | 183.9 | *Diverged* |
| **Convolutional** | ResNet-dil | 3.46 ± 0.02 | 178.6 | 22.0 ± 2.4 |
| | U-Net (m8) | 6.16 ± 0.01 | 184.1 | 22.2 ± 3.6 |
| **Attention / Graph** | TF-Enc | 0.60 ± 0.25 | 3448.6 | 22.2 ± 3.9 |
| | TF-MGN | 0.69 ± 0.01 | 3498.0 | 18.9 ± 4.5 |
| **Generative (Diffusion)**| ACDM | 41.77 ± 0.01 | 659.2 | 22.6 ± 4.0 |
| **Continuous Koopman** | **KAE (Ours)** | **0.00104 ± 0.0001** | 2751.3 | **14.9 ± 1.3** |

---

## 3. The "Closure Problem": Spectral Bias and Extreme Stability
**Addressing Reviewer Feedback:** *Reviewer Ge7F noted that enforcing a linear ODE in a truncated latent space introduces closure errors when modeling infinite-dimensional chaotic fluid systems.*

We provide a rigorous frequency spectrum analysis to explicitly visualize this structural boundary. The Continuous KAE acts as a physical low-pass filter. As shown in the spatial frequency plot, the KAE exhibits a steeper energy drop-off at high wavenumbers compared to the stochastic baseline, smoothing out fine-scale, unpredictable turbulent textures.

However, this spectral bias is the exact mechanism that guarantees our model's extreme stability. It successfully locks onto the dominant macro-scale vortex shedding frequencies (temporal plot) while shedding the chaotic noise that causes autoregressive models to diverge.

### Temporal and Spatial Frequency Analysis
![Spectral Analysis](figures/fig7_premultiplied_grid_longer_250.png)
*Figure 1: The KAE accurately captures the dominant vortex shedding frequencies (left) but exhibits a steeper energy drop-off at high spatial wavenumbers (right) compared to the stochastic baseline.*

### Extreme 1000-Step Rollout Stability
To stress-test this stability, we subjected the models to a 1000-step autoregressive rollout. As the unconstrained diffusion baseline (ACDM) compounds high-frequency stochastic errors, its spatial correlation collapses into unphysical numerical noise. Conversely, the Continuous KAE remains strictly bounded by its linear latent dynamics, degrading gracefully into a stable limit cycle.

<p align="center">
  <img src="figures/spatial_correlation_longer.png" width="48%" alt="Spatial Correlation 1000 Steps" />
  <img src="figures/l2_error_rollout_longer.png" width="48%" alt="L2 Error 1000 Steps" />
</p>
*Figure 2: Quantitative metrics over an extreme 1000-step rollout in the chaotic Transonic regime.*

---

## 4. Spatial Error Difference Maps
**Addressing Reviewer Feedback:** *Reviewer z2Gs requested absolute error fields to clearly visualize where the KAE drifts compared to the diffusion baseline.*

We plot the absolute spatial difference ($|\text{Prediction} - \text{Ground Truth}|$). Because the KAE enforces smooth global structural alignment, its errors are deterministic and localized almost exclusively along sharp spatial discontinuities (the transonic shock fronts). In contrast, the diffusion baseline exhibits diffuse, widespread stochastic noise across the entire fluid domain.

### Transonic Interpolation ($Tra_{int}$)
![Difference Maps Interp](figures/difference_maps_interp.png)
*Figure 3: Absolute spatial error distribution for the interpolation task. KAE errors are tightly bounded to the shock waves.*

### Transonic Extrapolation ($Tra_{ext}$)
![Difference Maps Extrap](figures/difference_maps_extrap.png)
*Figure 4: Absolute spatial error distribution for the low-Mach extrapolation task.*
