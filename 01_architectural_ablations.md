# Ablation Studies: Architecture & Loss Components

This document details the ablation studies requested by reviewers to evaluate the structural components and parameterization choices of the Continuous-Time Koopman Autoencoder (KAE).

### 1. Operator Parameterization: LoRA vs. Full-Rank MLP 
**Addressed to:** Reviewer z2Gs (Motivation for LoRA vs. MLP)

**Experiment:** We compared our default Low-Rank Adaptation (LoRA) parameterization of the Koopman operator against a full-rank MLP ($\mathbf{K}_{\mathrm{cont}} = \mathrm{MLP}(\phi)$) and an unconditional base model. 

**Observations (Table 1):**
* **Expressivity vs. Overfitting:** The full-rank MLP performs adequately on interpolation tasks but degrades significantly on out-of-distribution extrapolation. For example, on the low-Reynolds task ($Inc_{low}$), MSE increases from $1.3 \times 10^{-4}$ (LoRA) to $10.4 \times 10^{-4}$ (MLP).
* **Conclusion:** The massive parameter space of a full-rank $O(N_z^2)$ update allows the model to overfit to specific training regimes. LoRA's low-rank bottleneck acts as a necessary structural regularizer, anchoring dynamics to a stable base operator and improving physical extrapolation.

### 2. Temporal Loss: Cosine vs. Uniform Weighting
**Addressed to:** Reviewer z2Gs (Ablation of weighting schedule)

**Experiment:** We compared the proposed decaying cosine weighting schedule for the prediction loss ($\mathcal{L}_{\text{pred}}$) against a standard uniform weighting baseline.

**Observations (Table 1):**
* **Chaotic Regimes:** On the smooth incompressible dataset, both schedules perform similarly. However, on the chaotic transonic dataset, the cosine schedule reduces the 240-step $Tra_{long}$ MSE from $17.0 \times 10^{-3}$ to $14.9 \times 10^{-3}$.
* **Conclusion:** Uniform weighting distributes the gradient penalty equally, allowing early-step phase shifts to compound. The cosine schedule heavily penalizes immediate step errors ($t+1, t+2$), forcing local phase alignment before optimizing for global stability.

#### Table 1: Parameterization and Weighting Ablations (MSE)

| Conditioning | Temporal Weighting | $Inc_{low}$ ($\times 10^{-4}$) | $Inc_{high}$ ($\times 10^{-5}$) | $Tra_{ext}$ ($\times 10^{-3}$) | $Tra_{int}$ ($\times 10^{-3}$) | $Tra_{long}$ ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LoRA (Proposed)** | **Cosine** | **1.3 ± 1.7** | **2.9 ± 1.1** | **2.2 ± 0.9** | **5.2 ± 2.4** | **14.9 ± 1.3** |
| LoRA | Uniform | 1.3 ± 1.7 | 2.9 ± 1.1 | 2.5 ± 0.8 | 6.5 ± 1.6 | 17.0 ± 2.3 |
| MLP (Full-Rank) | Cosine | 10.4 ± 17.5 | 21.4 ± 7.1 | 3.6 ± 1.0 | 5.7 ± 3.0 | 15.1 ± 1.9 |
| Base (Unconditional)| Cosine | 116.5 ± 31.0 | 2991.2 ± 12.5| 13.9 ± 0.8 | 21.0 ± 2.7 | 18.1 ± 1.7 |

**Table 1 Legend:** Quantitative comparison of operator conditioning methods and temporal loss weighting. Performance is reported as MSE across five datasets. **LoRA (Proposed)** uses a low-rank bottleneck for parameter-efficient conditioning, while **Full-Rank MLP** predicts the entire operator matrix directly. **Cosine** refers to our decaying temporal weighting schedule, and **Uniform** applies equal weighting across the entire rollout. **Base (Unconditional)** represents a single, regime-invariant Koopman operator without physical parameter conditioning.

---

### 3. Structural Constraints & Consistency
**Addressed to:** Reviewers z2Gs (Comparison to Azencot et al.), RCnK (History encoder, structural regularization), B4CM (Two encoders).

**Experiment:** We incrementally removed key architectural constraints to isolate their impact on long-horizon stability (Table 2).

* **Continuous-Time Invertibility (vs. Discrete Consistent Koopman):** Azencot et al. (2020) enforce discrete forward ($A$) and backward ($B$) consistency via $AB \approx I$. We enforce the continuous-time equivalent by integrating the dynamics at $\Delta t$ and $-\Delta t$, penalizing deviations from $e^{\mathbf{K}\Delta t} e^{-\mathbf{K}\Delta t} = I$. Removing this constraint increases 240-step MSE from $14.9 \times 10^{-3}$ to $18.5 \times 10^{-3}$.
* **History Encoder (Takens' Delay Embedding):** Fluid observations are non-Markovian (e.g., velocity alone does not capture pressure/density). Following Takens' delay embedding theorem, processing the immediate past alongside the present acts as a proxy for temporal derivatives. Removing the history encoder results in the loss of phase-space information, degrading long-term MSE to $18.6 \times 10^{-3}$.
* **Structural Regularization (Formerly "Physics-Informed"):** We use Sobolev (spatial gradients) and Fourier spectral norms to preserve sharp wavefronts and shedding frequencies. Removing these priors blurs predictions and degrades long-horizon MSE to $19.3 \times 10^{-3}$.

#### Table 2: Structural and Consistency Component Ablations

| Model Configuration | $Tra_{ext}$ MSE | $Tra_{ext}$ LSiM | $Tra_{int}$ MSE | $Tra_{int}$ LSiM | $Tra_{long}$ MSE | $Tra_{long}$ LSiM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Continuous KAE (Proposed)** | **2.2 ± 0.9** | **1.8 ± 0.3** | **5.2 ± 2.4** | **2.1 ± 0.6** | **14.9 ± 1.3** | **5.0 ± 0.4** |
| w/o Directional Cosine | 2.4 ± 0.6 | 3.5 ± 0.3 | 5.3 ± 2.4 | 3.6 ± 0.4 | 17.8 ± 1.5 | 6.6 ± 0.3 |
| w/o Latent Energy Norm | 2.5 ± 0.6 | 3.3 ± 0.2 | 5.9 ± 2.3 | 3.7 ± 0.4 | 18.3 ± 2.0 | 6.0 ± 0.2 |
| w/o Azencot Consistency | 2.6 ± 0.6 | 3.3 ± 0.3 | 5.8 ± 2.7 | 4.0 ± 0.3 | 18.5 ± 1.4 | 6.5 ± 0.3 |
| w/o History Encoder | 2.6 ± 0.8 | 3.6 ± 0.5 | 5.9 ± 3.5 | 3.7 ± 0.3 | 18.6 ± 0.5 | 7.4 ± 0.3 |
| w/o Structural Regularizers| 2.5 ± 0.5 | 3.5 ± 0.4 | 5.6 ± 2.3 | 3.7 ± 0.2 | 19.3 ± 1.2 | 6.6 ± 0.4 |

**Table 2 Legend:** Ablation of individual structural and consistency components evaluated on the Transonic regime. We evaluate the **Original Proposed Model** against variants removing specific constraints:
* **w/o Directional Cosine:** Removes the cosine similarity component from the latent loss mix used for trajectory alignment.
* **w/o Latent Energy Norm:** Replaces the multi-component latent loss mix with a simple L2 norm between true and predicted latent states.
* **w/o Azencot Consistency:** Removes the generalized continuous-time consistency operator (enforcing operator invertibility via $e^{\mathbf{K}\Delta t} e^{-\mathbf{K}\Delta t} = I$).
* **w/o History Encoder:** Removes the past-state snapshot, forcing a strictly Markovian ($t \to t+1$) initialization without Takens' delay embedding.
* **w/o Structural Regularizers:** Removes the Sobolev (gradient) and Fourier spectral norms (physics-based spatial/frequency priors). 

Metrics include Mean Squared Error (MSE) and Learned Perceptual Similarity (LSiM) for structural fidelity.
