"""Statistical metrics for comparing cosmological parameter constraints."""

from typing import NamedTuple

import numpy as np
from scipy.stats import norm


class TensionResult(NamedTuple):
    """Result of a Gaussian tension computation.

    Attributes:
        tension: Tension in units of $\sigma$:
            $T = |\mu_1 - \mu_2| / \sqrt{\sigma_1^2 + \sigma_2^2}$.
        p_value: Two-tailed p-value $p = 2(1 - \Phi(T))$.
    """

    tension: float
    p_value: float


def get_Rm1(samples: dict) -> dict:
    """Compute the Gelman-Rubin $R-1$ convergence diagnostic for MCMC chains.

    Args:
        samples: Mapping of label to GetDist `MCSamples` chain object.

    Returns:
        Mapping of label to $R-1$ value. Values near zero indicate convergence;
        the conventional threshold is $R-1 < 0.01$.
    """
    return {lbl: chain.getGelmanRubin() for lbl, chain in samples.items()}


def gaussian_tension(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> TensionResult:
    """Compute the Gaussian tension between two independent 1D constraints.

    Args:
        mu1: Mean of the first constraint $\mu_1$.
        sigma1: Standard deviation of the first constraint $\sigma_1$.
        mu2: Mean of the second constraint $\mu_2$.
        sigma2: Standard deviation of the second constraint $\sigma_2$.
            May be zero when comparing against a precise reference value.

    Returns:
        Named tuple with fields:

        - `tension` — tension $T$ in units of $\sigma$.
        - `p_value` — two-tailed p-value under the null hypothesis of consistency.

    Note:
        The Gaussian tension statistic is:

        $$T = \\frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}, \quad p = 2(1 - \Phi(T))$$

        where $\Phi$ is the standard normal CDF. This is appropriate only when
        both constraints are approximately Gaussian.

    Example:
        >>> gaussian_tension(73.04, 1.04, 67.4, 0.5)
        TensionResult(tension=4.73..., p_value=...)
    """
    T = abs(mu1 - mu2) / np.sqrt(sigma1**2 + sigma2**2)
    p = 2.0 * (1.0 - norm.cdf(T))
    return TensionResult(tension=T, p_value=p)


def gaussian_tension_from_chain(
    chain, param: str, mu_ref: float, sigma_ref: float
) -> TensionResult:
    """Compute Gaussian tension for a GetDist chain parameter vs a fixed reference.

    Args:
        chain: A loaded `getdist.MCSamples` chain object.
        param: Parameter name as it appears in the chain (e.g. `"H0"`).
        mu_ref: Mean of the reference constraint.
        sigma_ref: Standard deviation of the reference constraint.

    Returns:
        See `gaussian_tension`.
    """
    mu = chain.mean([param])[0]
    sigma = chain.std([param])[0]
    return gaussian_tension(mu, sigma, mu_ref, sigma_ref)
