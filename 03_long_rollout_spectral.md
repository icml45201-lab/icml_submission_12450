# The Closure Problem: Spectral Bias & Long-Rollout Stability

**Reviewer Comments Addressed:**
* **Ge7F**: The inherent closure problem of truncating infinite-dimensional chaotic features into a finite-dimensional linear operator. Convergence in chaos over long windows.

**Response & Empirical Evidence:**

**1. Spectral Bias (Addressing Ge7F)**
* **Experiment**: We analyzed the Fast Fourier Transform (FFT) on spatial and temporal domains.
* **Result**: [See Figure: fig7_premultiplied_grid_longer_250.png]. The Continuous KAE exhibits a steeper spatial energy drop-off compared to diffusion models. The truncation acts as a spatial low-pass filter. Temporally, it successfully isolates the dominant macro-scale vortex shedding frequencies (the oscillating flow pattern where vortices detach periodically from the body) while suppressing high-frequency noise.

**2. Asymptotic Stability via Eigenvalues (Addressing Ge7F)**
* **Experiment**: We computed the eigenvalue spectrum of $\mathbf{K}_{\mathrm{cont}}(\phi)$.
* **Result**: [See Figure: eigenvalues_spectrum.png]. The spectrum lies strictly in the left half of the complex plane ($Re(\lambda) < 0$). This mathematically guarantees dissipative dynamics, preventing errors from compounding exponentially during autoregressive rollout.

**3. Extreme 1000-Step Rollout Stability (Addressing Ge7F)**
* **Experiment**: We ran the models for 1000 steps and compared MSE and spatial correlation with ACDM.
* **Result**: [See Figure: spatial_correlation_longer.png / l2_error_rollout_longer.png]. In the highly chaotic transonic regime, ACDM completely decorrelates. The KAE model remains stable over long rollouts by collapsing to the base flow (the stable limit cycle). While this limits high-frequency textural accuracy, it strictly bounds the maximum error, providing safe bounds for long-horizon extrapolation.
