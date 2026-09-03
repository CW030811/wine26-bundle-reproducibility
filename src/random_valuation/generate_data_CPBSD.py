import argparse
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch


# -------------------------------
# CPBSD Formulation (Paper Table 2)
# -------------------------------
# Marginal valuation family: exponential / logit(gumbel) / lognormal / normal / uniform
# Correlation structure: Gaussian copula with proxy rho in {-0.5, 0, 0.5}
# Product count: N in {5, 10, 30} (script supports any N >= 2)
# Heterogeneity: none / partial / full
# Cost scenarios: zero / HVHM / HVLM

DIST_FAMILIES = {"exponential", "logit", "lognormal", "normal", "uniform"}
HETEROGENEITY = {"none", "partial", "full"}
COST_SCENARIOS = {"zero", "hvhm", "hvlm", "random_ind", "random_corr"}


@dataclass
class CPBSDSetup:
    n_products: int
    k_samples: int
    dist_family: str
    rho: float
    heterogeneity: str
    cost_scenario: str
    seed: int


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    # Inverse standard normal CDF via erfinv: Phi^{-1}(u) = sqrt(2) * erfinv(2u-1)
    t = torch.as_tensor(u, dtype=torch.float64)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * t - 1.0)
    return z.numpy()


def _toeplitz_corr(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def valuation_means(n: int, heterogeneity: str) -> np.ndarray:
    if heterogeneity == "none":
        return np.ones(n)

    if heterogeneity == "partial":
        n1 = int(0.4 * n)
        means = np.ones(n)
        means[n1:] = 5.0
        return means

    # full
    if n == 1:
        return np.array([1.0])
    idx = np.arange(n)
    return 1.0 + 9.0 * idx / (n - 1)


def production_costs(n: int, heterogeneity: str, scenario: str, rng=None) -> np.ndarray:
    idx = np.arange(n)

    if scenario == "zero":
        return np.zeros(n)

    if scenario == "random_ind":
        # Independent random costs, same scale as valuation means [1, 10]
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(0.0, 10.0, size=n)

    if scenario == "random_corr":
        # Correlated with valuation means, but with random ratio + noise
        if rng is None:
            rng = np.random.default_rng()
        means = valuation_means(n, heterogeneity)
        ratio = rng.uniform(0.3, 0.9, size=n)
        noise = rng.normal(0, 0.5, size=n)
        return np.maximum(means * ratio + noise, 0.0)

    if heterogeneity == "none":
        if scenario == "hvhm":
            return np.full(n, 0.8)
        # hvlm
        return 0.1 + 1.5 * idx / max(1, (n - 1))

    if heterogeneity == "partial":
        n1 = int(0.4 * n)
        c = np.zeros(n)
        if scenario == "hvhm":
            c[:n1] = 0.8
            c[n1:] = 4.4
            return c

        # hvlm (piecewise in Table 2)
        c[:n1] = 0.1 + 1.5 * np.arange(n1) / max(1, (n - 1))
        n2 = n - n1
        c[n1:] = 4.1 + 1.5 * np.arange(n2) / max(1, (n2 - 1))
        return c

    # full heterogeneity
    if scenario == "hvhm":
        return 0.8 + 8.6 * idx / max(1, (n - 1))
    return 0.1 + 10.5 * idx / max(1, (n - 1))


def sample_valuations(k: int, means: np.ndarray, family: str, rho: float, rng: np.random.Generator) -> np.ndarray:
    n = len(means)
    corr = _toeplitz_corr(n, rho)
    z = rng.multivariate_normal(mean=np.zeros(n), cov=corr, size=k)
    u = np.clip(_norm_cdf(z), 1e-10, 1 - 1e-10)

    if family == "exponential":
        v = -means * np.log(1.0 - u)
    elif family == "logit":
        # Gumbel(loc, scale) with scale=0.25 and loc=E[V]-0.577*scale
        scale = 0.25
        loc = means - 0.577 * scale
        v = loc + (-scale * np.log(-np.log(u)))
    elif family == "lognormal":
        sigma = 0.5  # sqrt(0.25)
        mu = np.log(means) - 0.5 * (sigma**2)
        v = np.exp(mu + sigma * _norm_ppf(u))
    elif family == "normal":
        sigma = 0.5
        v = means + sigma * _norm_ppf(u)
    elif family == "uniform":
        v = (means - 1.0) + 2.0 * u
    else:
        raise ValueError(f"Unsupported family: {family}")

    # Keep numerical stability and economic meaning
    return np.maximum(v, 0.0)


def generate_cpbsd_instance(setup: CPBSDSetup) -> Dict:
    rng = np.random.default_rng(setup.seed)

    means = valuation_means(setup.n_products, setup.heterogeneity)
    costs = production_costs(setup.n_products, setup.heterogeneity, setup.cost_scenario, rng=rng)
    valuations = sample_valuations(
        k=setup.k_samples,
        means=means,
        family=setup.dist_family,
        rho=setup.rho,
        rng=rng,
    )

    potential_surplus = valuations - costs[None, :]

    # Minimal formulation sanity check for CPBSD payment rule
    # Pay(S) = sum_{n in S} p_n - |S| d_|S|
    p_demo = rng.uniform(0.5, 2.0, size=setup.n_products)
    d_demo = np.zeros(setup.n_products + 1)
    if setup.n_products >= 1:
        d_demo[1:] = rng.uniform(0.0, 0.3, size=setup.n_products)

    S = rng.integers(0, 2, size=setup.n_products)
    s = int(S.sum())
    pay_by_definition = float((p_demo * S).sum() - s * d_demo[s])

    return {
        "setup": asdict(setup),
        "means_E_V": means,
        "production_cost_c": costs,
        "valuation_samples_V": valuations,
        "potential_surplus_Z": potential_surplus,
        "cpbsd_definition_check": {
            "S_binary": S,
            "bundle_size_s": s,
            "component_price_p": p_demo,
            "size_discount_d": d_demo,
            "payment_sum_p_minus_s_d": pay_by_definition,
        },
    }


def generate_batch(
    out_dir: str,
    n_products: int,
    k_samples: int,
    dist_family: str,
    rho: float,
    heterogeneity: str,
    cost_scenario: str,
    n_instances: int,
    seed: int,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(n_instances):
        setup = CPBSDSetup(
            n_products=n_products,
            k_samples=k_samples,
            dist_family=dist_family,
            rho=rho,
            heterogeneity=heterogeneity,
            cost_scenario=cost_scenario,
            seed=seed + i,
        )
        data = generate_cpbsd_instance(setup)

        fname = (
            f"cpbsd_instance_{i+1:03d}_N{n_products}_K{k_samples}_"
            f"{dist_family}_rho{rho}_{heterogeneity}_{cost_scenario}.msgpack"
        )
        fpath = os.path.join(out_dir, fname)
        with open(fpath, "wb") as f:
            msgpack.dump(data, f, default=mnp.encode)
        paths.append(fpath)

    return paths


def parse_args():
    p = argparse.ArgumentParser(description="CPBSD instance generator (formulation-aligned)")
    p.add_argument("--out-dir", default="./dataset/cpbsd")
    p.add_argument("--N", type=int, default=10)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--dist", choices=sorted(DIST_FAMILIES), default="normal")
    p.add_argument("--rho", type=float, default=0.0, help="Correlation proxy, e.g. -0.5, 0, 0.5")
    p.add_argument("--hetero", choices=sorted(HETEROGENEITY), default="full")
    p.add_argument("--cost", choices=sorted(COST_SCENARIOS), default="hvhm")
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260304)
    return p.parse_args()


def main():
    args = parse_args()
    paths = generate_batch(
        out_dir=args.out_dir,
        n_products=args.N,
        k_samples=args.K,
        dist_family=args.dist,
        rho=args.rho,
        heterogeneity=args.hetero,
        cost_scenario=args.cost,
        n_instances=args.instances,
        seed=args.seed,
    )
    print(f"Generated {len(paths)} CPBSD instance files:")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
