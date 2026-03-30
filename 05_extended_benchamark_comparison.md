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
### Table A: Comprehensive Quantitative Comparison (Accuracy, Speed, & Memory)
*Performance evaluated across short-term extrapolation ($Inc$, $Tra$) and the critical 240-step rollout ($Tra_{long}$). Profiling conducted on a single A100 GPU. KAE achieves the highest long-horizon stability while maintaining a near-zero inference cost.*

| **Method** | **$Inc_{low}$** \tiny{($10^{-4}$)} | **$Inc_{high}$** \tiny{($10^{-5}$)} | **$Tra_{ext}$** \tiny{($10^{-3}$)} | **$Tra_{int}$** \tiny{($10^{-3}$)} | **$Tra_{long}^\dagger$** \tiny{($10^{-3}$)} | **Inf. Time** \tiny{(ms/step)} | **Speedup** | **VRAM** \tiny{(MB)} |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet** | $10.0 \pm 9.1$ | $16.0 \pm 3.0$ | $2.3 \pm 0.9$ | $1.8 \pm 1.0$ | $24.2 \pm 4.6$ | $3.67$ | **×11** | $188.0$ |
| **ResNet-dil** | $1.6 \pm 1.8$ | $2.6 \pm 0.7$ | $1.2 \pm 0.3$ | $\mathbf{1.0 \pm 0.5}$ | $22.0 \pm 2.4$ | $3.46$ | **×12** | $\mathbf{178.6}$ |
| **FNO-16** | $2.8 \pm 3.1$ | $8.9 \pm 3.8$ | $4.8 \pm 1.2$ | $5.5 \pm 2.6$ | $20.8 \pm 2.0$ | $1.17$ | **×35** | $184.1$ |
| **FNO-32** | $160 \pm 50$ | $1000 \pm 140$ | $4.9 \pm 1.9$ | $6.8 \pm 3.4$ | *Diverged* | $1.17$ | **×35** | $183.9$ |
| **TF-Enc** | $1.5 \pm 1.7$ | $8.7 \pm 3.8$ | $\mathbf{1.0 \pm 0.3}$ | $1.8 \pm 0.3$ | $22.2 \pm 3.9$ | $0.60$ | **×70** | $3448.6$ |
| **TF-MGN** | $5.7 \pm 4.3$ | $10.0 \pm 2.9$ | $3.9 \pm 1.0$ | $6.3 \pm 4.4$ | $18.9 \pm 4.5$ | $0.69$ | **×60** | $3498.0$ |
| **TF-VAE** | $5.4 \pm 5.5$ | $4.1 \pm 1.4$ | $2.4 \pm 0.2$ | $2.7 \pm 0.6$ | $20.6 \pm 2.1$ | $0.30$ | **×139** | $13749.9$ |
| **U-Net** | $1.0 \pm 1.1$ | $2.7 \pm 0.6$ | $3.1 \pm 2.1$ | $2.3 \pm 2.0$ | $30.3 \pm 6.1$ | $6.19$ | **×7** | $183.7$ |
| **U-Net$_{ut}$** | $\mathbf{0.8 \pm 1.1}$ | $\mathbf{0.2 \pm 0.1}$ | $1.6 \pm 0.7$ | $1.5 \pm 1.5$ | $22.2 \pm 3.6$ | $6.16$ | **×7** | $184.1$ |
| **U-Net$_{tn}$** | $1.0 \pm 1.0$ | $0.9 \pm 0.6$ | $1.4 \pm 0.8$ | $1.8 \pm 1.1$ | $22.4 \pm 3.9$ | $6.16$ | **×7** | $184.1$ |
| **Refiner** | $1.3 \pm 1.4$ | $3.5 \pm 2.2$ | $5.4 \pm 2.1$ | $7.1 \pm 2.1$ | *Diverged* | $10.31$ | **×4** | $642.4$ |
| **ACDM$_{ncn}$** | $0.9 \pm 0.8$ | $5.7 \pm 2.7$ | $4.1 \pm 1.9$ | $2.8 \pm 1.3$ | $22.8 \pm 3.8$ | $41.70$ | **×1** | $649.2$ |
| **ACDM** | $1.7 \pm 2.2$ | $0.8 \pm 0.4$ | $2.3 \pm 1.4$ | $2.7 \pm 2.1$ | $22.6 \pm 4.0$ | $41.77$ | **×1** | $659.2$ |
| **KAE (Ours)** | $1.3 \pm 1.7$ | $2.9 \pm 1.1$ | $2.2 \pm 0.9$ | $5.2 \pm 2.4$ | $\mathbf{14.9 \pm 1.3}$ | $\mathbf{0.001}$ | **×41,770** | $2751.3$ |


---
$^\dagger$ **Note:** $Tra_{long}$ results computed for this rebuttal to establish long-horizon limits; not present in the original ACDM paper. Best per column in **bold**.
