# Architectural Ablations: Operator Parameterization & Structural Constraints

**Reviewer Comments Addressed:**
* **z2Gs**: Motivation for LoRA vs. MLP parameterization? Ablation of cosine weighting schedule? Comparison to Consistent Koopman (Azencot et al., 2020)?
* **RCnK**: Justification for the history encoder? Clarification on physics-informed regularization?
* **B4CM**: Why two encoders (history and present)?

**Response & Empirical Evidence:**

**1. LoRA vs. Full-Rank MLP (Addressing z2Gs)**
* **Experiment**: We replaced the low-rank adaptation with a full-rank MLP to predict the generator matrix $\mathbf{K}_{\mathrm{cont}}$.
* **Result**: The MLP overfits to the training regimes and degrades heavily on extrapolation tasks ($Inc_{low}$ MSE spikes from $1.3 \times 10^{-4}$ to $10.4 \times 10^{-4}$). The low-rank constraint is necessary to prevent the continuous generator from deviating from the globally stable base flow.

**2. Temporal Weighting & Consistency (Addressing z2Gs)**
* **Experiment**: We ablated the Cosine weighting schedule (replacing with uniform) and the continuous-time Azencot consistency loss.
* **Result**: Uniform weighting increases 240-step $Tra_{long}$ MSE from $14.9 \times 10^{-3}$ to $17.0 \times 10^{-3}$. Removing the continuous Azencot consistency (forward-backward integration loss $e^{\mathbf{K}\Delta t} e^{-\mathbf{K}\Delta t} = I$) increases long-term MSE to $18.5 \times 10^{-3}$. 

**3. History Encoder Justification (Addressing RCnK, B4CM)**
* **Experiment**: We evaluated the model strictly as a Markovian system by removing the history encoder.
* **Result**: Following Takens' delay embedding theorem, utilizing past states acts as a temporal derivative proxy for hidden variables. Removing it caused the highest structural deformation in long rollouts ($18.6 \times 10^{-3}$ MSE).

**4. Structural Regularization (Addressing RCnK)**
* **Experiment**: We removed the Sobolev (spatial gradient) and spectral norms to clarify our "physics-informed" terminology. 
* **Result**: These terms act as structural regularizers rather than embedding explicit Navier-Stokes equations. Removing them degrades long-horizon accuracy to $19.3 \times 10^{-3}$.

**Table: Structural and Consistency Ablations (MSE)**
| Configuration | $Inc_{low}$ ($10^{-4}$) | $Tra_{ext}$ ($10^{-3}$) | $Tra_{long}$ ($10^{-3}$) |
| :--- | :--- | :--- | :--- |
| **Proposed (LoRA + Cosine)** | **1.3 ± 1.7** | **2.2 ± 0.9** | **14.9 ± 1.3** |
| MLP (Full-Rank) | 10.4 ± 17.5 | 3.6 ± 1.0 | 15.1 ± 1.9 |
| Uniform Weighting | 1.3 ± 1.7 | 2.5 ± 0.8 | 17.0 ± 2.3 |
| w/o Azencot Consistency | - | 2.6 ± 0.6 | 18.5 ± 1.4 |
| w/o History Encoder | - | 2.6 ± 0.8 | 18.6 ± 0.5 |
| w/o Structural Reg. | - | 2.5 ± 0.5 | 19.3 ± 1.2 |
