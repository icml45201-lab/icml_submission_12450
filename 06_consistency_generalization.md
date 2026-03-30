## Continuous-Time Generalization of Discrete Koopman Consistency

Consistent Koopman autoencoders introduced by [Azencot et al., 2020](https://proceedings.mlr.press/v119/azencot20a/azencot20a.pdf) learn separate forward and backward latent operators  
$A, B \in \mathbb{R}^{N_z \times N_z}$ satisfying

$$
z_{n+1} = A z_n, \qquad z_n = B z_{n+1}.
$$

Their supervised consistency objective is

$$
\mathcal{L}_{\mathrm{disc}} =
\|A z_n - z_{n+1}\|_2^2 +
\|B z_{n+1} - z_n\|_2^2.
$$

In our continuous-time formulation, latent dynamics are governed by the linear system

$$
\frac{dz}{dt} = K z,
$$

whose exact solution over a time interval $\Delta t$ is

$$
z(t+\Delta t) = e^{K\Delta t} z(t).
$$

Defining the discrete forward operator as

$$
A := e^{K\Delta t},
$$

immediately recovers the forward latent evolution:

$$
z_{n+1} = A z_n.
$$

Backward evolution follows from the same generator evaluated at negative time:

$$
z(t-\Delta t) = e^{-K\Delta t} z(t),
$$

which implies

$$
B := e^{-K\Delta t} = A^{-1}.
$$

Substituting these expressions into the discrete consistency objective yields

$$
\mathcal{L}_{\mathrm{disc}} =
\|e^{K\Delta t} z_n - z_{n+1}\|_2^2 +
\|e^{-K\Delta t} z_{n+1} - z_n\|_2^2,
$$

which exactly matches the latent forward-backward consistency loss used in our continuous-time model.

Unlike discrete formulations, where $A$ and $B$ are learned independently and invertibility must be promoted through explicit regularization, the continuous generator imposes

$$
B = A^{-1}
$$

by construction.

Therefore, the proposed latent consistency objective is the exact continuous-time counterpart of discrete consistent Koopman training under matrix exponential flow.
