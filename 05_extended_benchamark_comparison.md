## Exhaustive Baseline Benchmarking & $O(1)$ Inference Efficiency
**Addressed to:** *Reviewers z2Gs, B4CM, RCnK (Requests for broader baseline comparisons beyond diffusion models).*

We fully agree with the reviewers that contextualizing our Continuous-Time KAE requires a broader lens than autoregressive diffusion models alone. To provide a definitive and irrefutable assessment of where our method sits within the current landscape of PDE forecasting, we conducted a massive benchmarking effort, evaluating **14 distinct spatial-temporal surrogate models**. 

To ensure a comprehensive analysis, these baselines span four dominant architectural paradigms:
1. **Generative / Probabilistic:** $ACDM$, $ACDM_{ncn}$ *(State-of-the-art for high-frequency flow synthesis)*
2. **Spectral / Neural Operators:** $FNO-16$, $FNO-32$ *(Standard baselines for resolution-invariant PDE solving)*
3. **Convolutional / Data-Space Autoregressive:** $U-Net$, $U-Net_{ut}$, $U-Net_{tn}$, $ResNet$, $ResNet-dil$, $Refiner$
4. **Attention / Graph-Based:** $TF-Enc$, $TF-MGN$, $TF-VAE$

### Empirical Conclusions: The Expressivity vs. Stability Trade-off
Our expanded results rigorously quantify the core trade-off of our proposed architecture. Highly non-linear models (such as $U-Net_{ut}$ and ACDM) capture slightly more high-frequency stochastic texture in the short term, yielding lower MSEs on the 60-step $Inc_{high}$ and $Tra_{ext}$ regimes. 

However, this short-term expressivity comes at a severe cost to long-horizon stability and inference efficiency. By strictly enforcing a global linear structure in the KAE's latent space, we completely bypass the iterative numerical solvers and autoregressive sampling procedures required by the 13 other baselines. 

Because we evaluate the latent state exactly via analytical matrix exponentiation $z(\tau)=\exp(\mathbf{K}_{\mathrm{cont}}\tau)z_0$, we achieve a substantial **inference speedup of >40,000×** over diffusion models (0.00104 ms vs 41.77 ms) and **>5,000×** over continuous U-Nets (0.00104 ms vs 6.16 ms). Furthermore, while highly expressive autoregressive models (FNO-32, Refiner) diverge over long horizons, our Continuous-Time KAE establishes state-of-the-art stability on the extreme 240-step Tra_long forecasting task.

---

### Table A: Inference Speed and Memory Efficiency Profiling
Profiling conducted over a 240-step rollout. The Continuous-Time KAE operates orders of magnitude faster than all evaluated baselines due to $O(1)$ latent state evaluation.

| Architecture | Avg. Step Inference (ms) | Mean VRAM (MB) |
| :--- | :--- | :--- |
| ResNet-m2 | $3.67 \pm 0.04$ | $188.0$ |
| Dil-ResNet-m2 | $3.46 \pm 0.02$ | $\mathbf{178.6}$ |
| FNO-16 | $1.17 \pm 0.01$ | $184.1$ |
| FNO-32 | $1.17 \pm 0.00$ | $183.9$ |
| UNet-m2 | $6.19 \pm 0.09$ | $183.7$ |
| UNet-m8 | $6.16 \pm 0.01$ | $184.1$ |
| TF-Enc | $0.60 \pm 0.25$ | $3448.6$ |
| TF-MGN | $0.69 \pm 0.01$ | $3498.0$ |
| TF-VAE | $0.30 \pm 0.01$ | $13749.9$ |
| Refiner-R4 | $10.31 \pm 0.02$ | $642.4$ |
| **Continuous KAE (Ours)** | **$\mathbf{0.00104 \pm 0.0001}$** | $2751.3$ |
| ACDM | $41.77 \pm 0.01$ | $659.2$ |
| $ACDM_{ncn}$ | $41.70 \pm 0.06$ | $649.2$ |

---


### Table B: Complete Quantitative Split Comparison (MSE)
Performance evaluated across both short-term extrapolation ( $Inc_{low}$, $Inc_{high}$, $Tra_{ext}$, $Tra_{int}$ ) and the critical long-horizon rollout ($Tra_{long}$, 240 steps). Note the catastrophic divergence of several standard baselines over extended horizons.

| Method | $Inc_{low}$ ($\times 10^{-4}$) | $Inc_{high}$ ($\times 10^{-5}$) | $Tra_{ext}$ ($\times 10^{-3}$) | $Tra_{int}$ ($\times 10^{-3}$) | $Tra_{long}$ ($\times 10^{-3}$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet | $10.0 \pm 9.1$ | $16.0 \pm 3.0$ | $2.3 \pm 0.9$ | $1.8 \pm 1.0$ | $24.2 \pm 4.6$ |
| ResNet-dil | $1.6 \pm 1.8$ | $2.6 \pm 0.7$ | $1.2 \pm 0.3$ | $\mathbf{1.0 \pm 0.5}$ | $22.0 \pm 2.4$ |
| $\text{FNO}_{16}$ | $2.8 \pm 3.1$ | $8.9 \pm 3.8$ | $4.8 \pm 1.2$ | $5.5 \pm 2.6$ | $20.8 \pm 2.0$ |
| $\text{FNO}_{32}$ | $160 \pm 50$ | $1000 \pm 140$ | $4.9 \pm 1.9$ | $6.8 \pm 3.4$ | *Diverged* |
| $\text{TF}_{MGN}$ | $5.7 \pm 4.3$ | $10.0 \pm 2.9$ | $3.9 \pm 1.0$ | $6.3 \pm 4.4$ | $18.9 \pm 4.5$ |
| $\text{TF}_{Enc}$ | $1.5 \pm 1.7$ | $8.7 \pm 3.8$ | $\mathbf{1.0 \pm 0.3}$ | $1.8 \pm 0.3$ | $22.2 \pm 3.9$ |
| $\text{TF}_{VAE}$ | $5.4 \pm 5.5$ | $4.1 \pm 1.4$ | $2.4 \pm 0.2$ | $2.7 \pm 0.6$ | $20.6 \pm 2.1$ |
| U-Net | $1.0 \pm 1.1$ | $2.7 \pm 0.6$ | $3.1 \pm 2.1$ | $2.3 \pm 2.0$ | $30.3 \pm 6.1$ |
| $\text{U-Net}_{ut}$ | $\mathbf{0.8 \pm 1.1}$ | $\mathbf{0.2 \pm 0.1}$ | $1.6 \pm 0.7$ | $1.5 \pm 1.5$ | $22.2 \pm 3.6$ |
| $\text{U-Net}_{tn}$ | $1.0 \pm 1.0$ | $0.9 \pm 0.6$ | $1.4 \pm 0.8$ | $1.8 \pm 1.1$ | $22.4 \pm 3.9$ |
| Refiner | $1.3 \pm 1.4$ | $3.5 \pm 2.2$ | $5.4 \pm 2.1$ | $7.1 \pm 2.1$ | *Diverged* |
| $\text{ACDM}_{ncn}$ | $0.9 \pm 0.8$ | $5.7 \pm 2.7$ | $4.1 \pm 1.9$ | $2.8 \pm 1.3$ | $22.8 \pm 3.8$ |
| ACDM | $1.7 \pm 2.2$ | $0.8 \pm 0.4$ | $2.3 \pm 1.4$ | $2.7 \pm 2.1$ | $22.6 \pm 4.0$ |
| **Continuous KAE (Ours)** | **$1.3 \pm 1.7$** | **$2.9 \pm 1.1$** | **$2.2 \pm 0.9$** | **$5.2 \pm 2.4$** | **$\mathbf{14.9 \pm 1.3}$** |

---
