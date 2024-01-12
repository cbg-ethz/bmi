# Fine distributions

In this tutorial we will take a closer look at the family of *fine distributions*, proposed in [The Mixtures and the Neural Critics](https://arxiv.org/abs/2310.10240) paper[@mixtures-neural-critics-2023].

We call a distribution $P_{XY}$ fine, if it is possible to evaluate the densities $\log p_{XY}(x, y)$, $\log p_X(x)$, and $\log p_Y(y)$ for arbitrary points and it is possible to efficiently sample from $P_{XY}$.

In particular, one can evaluate [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) $\mathrm{PMI}_{XY}(x, y) = \log \frac{p_{XY}(x, y)}{p_X(x)p_Y(y)}$.

\bibliography