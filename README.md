Version History & Changelog: Continuous-Time Koopman Autoencoders

This document outlines the major updates, structural changes, and new contributions introduced in the latest version of the paper "Koopman Autoencoders with Continuous-Time Latent Dynamics for Fluid Dynamics Forecasting". It serves as a comparison guide between the initial double-blind submission and the updated, fully revised manuscript, with a specific focus on the newly added results and methodological expansions.

1. Expanded Results & Experimental Rigor

The most significant updates to the manuscript are in the experimental section, which has been expanded to include extreme stress tests and deeper spectral analyses to validate the model's performance against diffusion baselines.

New Analysis: Extreme Long-Horizon Stability ($T=1000$): A major addition to the experiments is an extreme 1000-step stress test. The newly added results highlight that while diffusion models (ACDM) suffer from chaotic phase divergence and complete structural collapse over long horizons, the continuous-time KAE remains asymptotically bounded and degrades gracefully into a stable limit cycle.

New Analysis: Spectral Bias & Frequency Smoothing: The updated text formally acknowledges and investigates the "spectral bias" of the $L_2$-optimized KAE. Using new spatial and temporal frequency analyses, the results demonstrate how the KAE acts as a physical low-pass filter—smoothing high-frequency chaotic structures to maintain macroscopic phase stability and lock onto dominant shedding frequencies with near-zero variance.

Training Protocol Distinctions: The latest version introduces a clear distinction between "Stable-Only Training" and "Full-Dataset Training." It adds a fully retrained ACDM baseline evaluated from scratch to ensure a rigorous and fair comparison of generalization capabilities.

Sliding Window Strategy: The text now explicitly details the data-loading strategy, noting the use of an exhaustive sliding-window scheme to maximize data efficiency and temporal coverage, contrasted with the baseline's non-overlapping sequence approach.

2. Methodological Additions & Clarifications

The methodology section has been heavily revised to formally define the mathematical and structural additions that enable the reported stability and efficiency.

LoRA Parameterization Explicitly Defined: The new version formally introduces Low-Rank Adaptation (LoRA) to explain how the physics-conditioning network $\mathcal{N}_\psi(\phi)$ is parameterized. It details how this mechanism reduces the memory footprint from $O(N_z^2)$ to $O(2rN_z)$, preventing severe VRAM fragmentation during batched training while acting as a structural regularizer.

Gaussian Noise Regularization: The methodology now explicitly mentions the addition of injecting small Gaussian noise into the physical conditioning parameters ($\phi$) during training to improve robustness to sparse parameter sampling and discretization artifacts.

Expanded Objective Functions:

Cosine Weighting Schedule: The new version clearly defines a decaying cosine schedule for temporal weights during the rollout prediction loss, which replaced the simpler uniform approach to prevent autoregressive error accumulation early in training.

Physics-Inspired Regularization: The newer version significantly expands the explanation of structural regularizations ($\mathcal{L}_{phys}$), explicitly breaking them down into Temporal Sobolev Loss (velocity matching), Spatial Sobolev Loss (structure matching), and Spectral Consistency Loss (Fourier domain alignment).

3. Extensive Appendices & Ablation Studies

The latest version vastly expands the supplementary materials, transforming it into a much more comprehensive mathematical and architectural resource featuring entirely new studies:

Eigenvalue Spectrum Analysis (New Result): Introduces an entirely new section visualizing the eigenvalue spectrum of the learned Koopman generator to mathematically prove its dissipative, stable nature. The results demonstrate that the learned eigenvalues lie predominantly in the left half of the complex plane, explaining the model's resistance to divergent error accumulation.

Spatial Error Distributions (New Result): Adds a section containing absolute spatial difference maps, visually proving that KAE errors are localized tightly around physical discontinuities (like shock fronts), while ACDM errors are diffusely spread across the spatial domain.

Ablation Studies (New Result): Contains newly added ablation studies cleanly isolating the quantitative impacts of the LoRA parameterization (compared to a full-rank MLP) and the Cosine temporal weighting schedule (compared to uniform weighting).

Theoretical Equivalence Proof: Adds a mathematical proof demonstrating that the proposed continuous-time latent consistency loss is the exact continuous generalization of the discrete Consistent Koopman Autoencoder framework.

Architecture Details: Adds thorough descriptions of the Convolutional Block Attention Module (CBAM) used in the encoder, the Siamese structure of the temporal history encoder, and the Adaptive Group Normalization (AdaGN) utilized in the decoder.

4. Abstract & Introduction Framing

Shift in Focus to Trade-offs: The previous abstract focused heavily on the mechanics of replacing discrete-time rollouts with numerical integration schemes. The updated abstract and introduction pivot to emphasize the trade-offs between highly expressive generative models (specifically, autoregressive diffusion models like ACDM) and deterministic, structured representations.

Stronger Evaluated Claims: Backed by the new $T=1000$ tests and efficiency metrics, the latest version explicitly claims "massive computational efficiency and extreme long-horizon stability" when compared to autoregressive neural operators and diffusion models.

Streamlined Introduction: Mentions of older techniques (e.g., LE-PDE and standard DMD) have been condensed or removed to dedicate more space to the direct comparison between stochastic diffusion models and deterministic Koopman operators.
