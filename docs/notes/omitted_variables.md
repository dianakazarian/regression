_Overview: Results & derivations for OLS results._

## Section [?]: Much Ado About Omitted Variables

In the social sciences, we spend a lot of time thinking and talking about omitted variables. This is because there are often key variables that compromise our ability to identify $\hat\beta$. Let us make this more precise.

Suppose we are interested in calculating $\beta$ in the following model:
$$
Y = X'\beta + W'\gamma+V
$$
We have $\mathbb{E}[XV] =0 $ and $\mathbb{E}[WV] = 0$. For one another or another, we cannot observe $W$, so we are resigned to consider the regression of $Y$ on just $X$.
$$
Y = X'\beta + (W'\gamma + V) = X'\beta + U
$$
where $U = W'\gamma + V$. We have endogeneity because:
$$
\mathbb{E}[XU] = \mathbb{E}[XW']\gamma + \mathbb{E}[XV] = \mathbb{E}[XW']\gamma \neq \mathbf{0}
$$

What are the cases in which we wouldn't need to worry about endogeneity?

1. $\gamma = 0$, which would mean that $W$ has zero true effect on $Y$.
2. $E[XW'] = 0$, which would mean that $X$ and $W$ are uncorrelated.
3. $\gamma \neq 0$ and $\mathbb{E}[XW'] \neq 0$ but, because these are matrices, they can still multiply to yield zero. There is a very slim chance of this actually happening (each of the dot products involved in the matrix multiplication would need to equal zero), but it's technically not impossible.