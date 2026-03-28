Version History & Changelog: Continuous-Time Koopman Autoencoders

This document outlines the major updates, structural changes, and new contributions introduced in the latest version of the paper "Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting". It serves as a comprehensive comparison guide between the initial double-blind submission and the updated, fully revised manuscript.

Note on Anonymity: This document, like the manuscript, remains entirely anonymous to comply with double-blind review standards.

1. Abstract & Introduction Framing

Shift in Focus to Trade-offs: The previous abstract focused heavily on the mechanics of replacing discrete-time rollouts with numerical integration schemes. The updated abstract and introduction pivot to emphasize the trade-offs between highly expressive generative models (specifically, autoregressive diffusion models like ACDM) and deterministic, structured continuous-time representations.

Stronger Evaluated Claims: Backed by new $T=1000$ extreme stress tests and rigorous efficiency metrics, the latest version explicitly claims "massive computational efficiency and extreme long-horizon stability" when compared to autoregressive neural operators and diffusion models.

2. Expanded Results & Experimental Rigor

The most significant updates to the manuscript are in the experimental section, which has been massively expanded to include extreme stress tests, ablation studies, and deep spectral analyses.

2.1. Extreme Long-Horizon Stability ($T=1000$)

A major addition is the extreme 1000-step stress test. The newly added results highlight that while diffusion models (ACDM) suffer from chaotic phase divergence and complete structural collapse over long horizons, the continuous-time KAE remains asymptotically bounded and degrades gracefully into a stable limit cycle.



Quantitative metrics over an extreme 1000-step rollout. ACDM exhibits severe instability and variance, while the Continuous KAE remains strictly bounded.


Visual snapshots: While the Continuous KAE smoothly diffuses the flow into a stable limit cycle, the unconstrained autoregressive diffusion baseline (ACDM) eventually compounds stochastic errors until the physical structure collapses into numerical noise.

2.2. Spectral Bias & Frequency Smoothing

The updated text formally acknowledges and investigates the "spectral bias" of the $L_2$-optimized KAE. Using new spatial and temporal frequency analyses, the results demonstrate how the KAE acts as a physical low-pass filter—smoothing high-frequency chaotic structures to maintain macroscopic phase stability and lock onto dominant shedding frequencies with near-zero variance.


The Continuous KAE successfully captures the dominant physical frequencies but attenuates high-frequency turbulent noise compared to the stochastic baseline.

2.3. Temporal Consistency and Super-Resolution

The latest manuscript highlights the continuous-time KAE's ability to generalize to arbitrary temporal resolutions without retraining.


Direct comparison of inference results using the matrix exponentiation and RK4 integrator at varying time steps ($\Delta t \in \{0.05s, 0.1s, 0.2s\}$).

2.4. Computational Efficiency (Table Added)

We added a comprehensive runtime and VRAM comparison to prove the "massive computational efficiency" claim. KAE inference via matrix exponentiation is substantially faster than iterative diffusion sampling.

Table: Runtime Comparison (240-step rollout)
| Architecture | Avg. Step (ms) | Mean VRAM (MB) |
| :--- | :--- | :--- |
| KAE (Ours) | $\mathbf{1.04 \times 10^{-3} \pm 10^{-4}}$ | $2751.3$ |
| FNO-16 | $1.17 \pm 0.01$ | $184.1$ |
| U-Net-m8 | $6.16 \pm 0.01$ | $184.1$ |
| ACDM | $41.77 \pm 0.01$ | $659.2$ |

2.5. Comprehensive Quantitative Baselines

The updated manuscript provides a massive benchmarking table comparing against ResNets, FNOs, U-Nets, TF-MGN, and Diffusion models.

Table: Quantitative Comparison across Regimes (MSE and LSiM)
| Method | $Inc_{low}$ (MSE $\times 10^{-4}$) | $Inc_{high}$ (MSE $\times 10^{-5}$) | $Tra_{ext}$ (MSE $\times 10^{-3}$) | $Tra_{long}$ (MSE $\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- |
| $\text{FNO}_{32}$ | $160 \pm 50$ | $1000 \pm 140$ | $4.9 \pm 1.9$ | Diverged |
| $\text{TF}_{MGN}$ | $5.7 \pm 4.3$ | $10 \pm 2.9$ | $3.9 \pm 1.0$ | $18.9 \pm 4.5$ |
| U-Net | $1.0 \pm 1.1$ | $2.7 \pm 0.6$ | $3.1 \pm 2.1$ | $30.3 \pm 6.1$ |
| ACDM | $1.7 \pm 2.2$ | $\mathbf{0.8 \pm 0.4}$ | $2.3 \pm 1.4$ | $22.6 \pm 4.0$ |
| Continuous KAE | $1.3 \pm 1.7$ | $2.9 \pm 1.1$ | $\mathbf{2.2 \pm 0.9}$ | $\mathbf{14.9 \pm 1.3}$ |
Note: The deterministic KAE smoothly outperforms stochastic baselines over the long horizon ($Tra_{long}$), degrading gracefully where other models diverge.



Qualitative validation rollouts for extrapolation regimes showing stable, physically consistent predictions.

3. Methodological Additions & Clarifications

The methodology section has been heavily revised to formally define the mathematical and structural additions that enable the reported stability and efficiency.

LoRA Parameterization Explicitly Defined: The new version formally introduces Low-Rank Adaptation (LoRA) to explain how the physics-conditioning network $\mathcal{N}_\psi(\phi)$ is parameterized. It details how this mechanism reduces the memory footprint from $O(N_z^2)$ to $O(2rN_z)$, preventing severe VRAM fragmentation during batched training while acting as a structural regularizer.

Gaussian Noise Regularization: Explicitly mentions the addition of injecting small Gaussian noise into the physical conditioning parameters ($\phi$) during training to improve robustness to sparse parameter sampling.

Cosine Weighting Schedule: Defines a decaying cosine schedule for temporal weights during the rollout prediction loss, which replaces the simpler uniform approach to prevent autoregressive error accumulation early in training.

Physics-Inspired Regularization: The newer version explicitly breaks down structural regularizations ($\mathcal{L}_{phys}$) into Temporal Sobolev Loss (velocity matching), Spatial Sobolev Loss (structure matching), and Spectral Consistency Loss (Fourier domain alignment).

4. Extensive Appendices & Deep Dives

The supplementary materials were transformed into a much more comprehensive mathematical and architectural resource.

4.1. Eigenvalue Spectrum Analysis (New)

Introduces an entirely new section visualizing the eigenvalue spectrum of the learned Koopman generator to mathematically prove its dissipative, stable nature. The results demonstrate that the learned eigenvalues lie predominantly in the left half of the complex plane, explaining the model's resistance to divergent error accumulation.

4.2. Spatial Error Distributions & Difference Maps (New)

Adds a section containing absolute spatial difference maps, visually proving that KAE errors are localized tightly around physical discontinuities (like shock fronts), while ACDM errors are diffusely spread across the spatial domain.




Difference maps show KAE errors are concentrated precisely at sharp shock fronts, while ACDM exhibits broader, compounding spatial noise.

4.3. Field-Wise Error Distributions (New)

Added rigorous violin and bar plots mapping MSE as a function of Reynolds number, showcasing KAE's controlled variance compared to the heavy-tailed error distributions of ACDM.

4.4. Ablation Studies (New)

Contains newly added ablation studies cleanly isolating the quantitative impacts of the LoRA parameterization and Cosine weighting.

Table: Ablation on Operator Parameterization & Weighting
| Configuration | $Inc_{low}$ (MSE) | $Tra_{ext}$ (MSE) | $Tra_{long}$ (MSE) |
| :--- | :--- | :--- | :--- |
| LoRA + Cosine (Proposed) | $\mathbf{1.3 \times 10^{-4}}$ | $\mathbf{2.2 \times 10^{-3}}$ | $\mathbf{14.9 \times 10^{-3}}$ |
| LoRA + Uniform | $1.3 \times 10^{-4}$ | $2.5 \times 10^{-3}$ | $17.0 \times 10^{-3}$ |
| MLP (Full-Rank) + Cosine | $10.4 \times 10^{-4}$ | $3.6 \times 10^{-3}$ | $15.1 \times 10^{-3}$ |
Reveals that Full-Rank MLP heavily degrades extrapolation performance compared to LoRA.

4.5. Integrator vs. Analytical Matrix Exponential (New)

Added comparisons proving that the numerical RK4 integrations perfectly align with the analytical matrix exponential solutions, validating the $O(1)$ continuous-time formulation.

4.6. Theoretical Equivalence Proof (New)

Adds a mathematical proof demonstrating that the proposed continuous-time latent consistency loss ($\|e^{K\Delta t} z_n - z_{n+1}\|_2^2 + \|e^{-K\Delta t} z_{n+1} - z_n\|_2^2$) is the exact continuous generalization of the discrete Consistent Koopman Autoencoder framework.
