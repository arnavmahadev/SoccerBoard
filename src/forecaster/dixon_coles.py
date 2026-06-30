"""Dixon-Coles bivariate-Poisson scoreline model (competition-agnostic).

Each team has an **attack** and a **defense** strength; there's a global
**home-advantage** term and the Dixon-Coles low-score correction `rho` (the
dependence adjustment for 0-0, 1-0, 0-1, 1-1, which independent Poissons get
wrong). For a fixture the model emits a **goal matrix** P(home=i, away=j), from
which W/D/L and the expected scoreline follow.

Goal expectations
-----------------
    lambda (home) = exp(attack_home - defense_away + home_adv * [not neutral])
    mu     (away) = exp(attack_away - defense_home)

Home advantage is applied only when the match is NOT at a neutral venue — which
is exactly what makes this usable for a World Cup, where every game is neutral
bar the hosts'.

Fitting
-------
Maximum likelihood via scipy `minimize` (L-BFGS-B). Two honest-modelling knobs,
both configurable:
  - time decay `xi` (per year): older matches are down-weighted exp(-xi * age),
    so the fit tracks current form. Combined with per-match importance weights
    (friendlies < qualifiers < tournament games) from `data.importance_weight`.
  - ridge `reg`: L2 shrinkage of attack/defense toward average, which stabilises
    teams with few recent matches (minnows) and improves out-of-sample log-loss.

Identifiability: attack strengths are mean-centred (a shared constant added to
all attacks and defenses leaves every lambda/mu unchanged), so the fit is gauge-free.

Interface:  fit(matches) -> Params ;  predict(params, home, away) -> goal matrix
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# scipy is imported lazily inside fit(); the serving/prediction path below is
# pure numpy, so the deployed image needs neither scipy nor scikit-learn — the
# committed params are just loaded and evaluated.
from forecaster.data import Match, importance_weight


@dataclass
class Params:
    teams: list[str]
    attack: list[float]
    defense: list[float]
    home_adv: float
    rho: float
    xi: float
    reg: float
    n_matches: int
    ref_date: str

    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: Path) -> "Params":
        return cls(**json.loads(Path(path).read_text()))


def _days_between(a: str, b: str) -> float:
    from datetime import date

    ya, ma, da = (int(x) for x in a.split("-"))
    yb, mb, db = (int(x) for x in b.split("-"))
    return (date(yb, mb, db) - date(ya, ma, da)).days


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score dependence correction (vectorised)."""
    t = np.ones_like(lam, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lam[m00] * mu[m00] * rho
    t[m01] = 1.0 + lam[m01] * rho
    t[m10] = 1.0 + mu[m10] * rho
    t[m11] = 1.0 - rho
    return t


def fit(
    matches: list[Match],
    xi: float = 0.35,
    reg: float = 0.02,
    ref_date: str | None = None,
    max_iter: int = 200,
) -> Params:
    """Fit attack/defense/home-advantage/rho by weighted maximum likelihood.

    Uses an analytic gradient (scatter-summed per team), so the ~600-parameter
    fit converges in a second or two rather than minutes of finite differences.
    """
    from scipy.optimize import minimize
    from scipy.special import gammaln

    matches = [m for m in matches if m.played]
    if not matches:
        raise ValueError("no played matches to fit")
    ref = ref_date or max(m.date for m in matches)

    teams = sorted({m.home for m in matches} | {m.away for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = np.array([idx[m.home] for m in matches])
    a = np.array([idx[m.away] for m in matches])
    x = np.array([m.home_goals for m in matches], dtype=float)
    y = np.array([m.away_goals for m in matches], dtype=float)
    home_flag = np.array([0.0 if m.neutral else 1.0 for m in matches])

    age_years = np.array([_days_between(m.date, ref) for m in matches]) / 365.25
    w = np.array([importance_weight(m.tournament) for m in matches])
    w = w * np.exp(-xi * np.maximum(age_years, 0.0))

    lgx = gammaln(x + 1.0)
    lgy = gammaln(y + 1.0)
    # Masks for the four Dixon-Coles low-score cells (fixed across iterations).
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    def objective(p):
        att = p[:n]
        att = att - att.mean()  # gauge fix (mean attack = 0)
        deff = p[n : 2 * n]
        gamma, rho = p[2 * n], p[2 * n + 1]
        lam = np.exp(att[h] - deff[a] + gamma * home_flag)
        mu = np.exp(att[a] - deff[h])

        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.maximum(tau, 1e-10)

        ll = w * (
            x * np.log(lam) - lam - lgx
            + y * np.log(mu) - mu - lgy
            + np.log(tau)
        )
        penalty = reg * (np.sum(att**2) + np.sum(deff**2))
        nll = -np.sum(ll) + penalty

        # dtau/dlam, dtau/dmu, dtau/drho (nonzero only on the low-score cells).
        dt_dl = np.zeros_like(lam)
        dt_dl[m00] = -mu[m00] * rho
        dt_dl[m01] = rho
        dt_dm = np.zeros_like(mu)
        dt_dm[m00] = -lam[m00] * rho
        dt_dm[m10] = rho
        dt_dr = np.zeros_like(lam)
        dt_dr[m00] = -lam[m00] * mu[m00]
        dt_dr[m01] = lam[m01]
        dt_dr[m10] = mu[m10]
        dt_dr[m11] = -1.0

        # u = dLL/d(log lam), v = dLL/d(log mu)  (per match, weighted below)
        u = w * (x - lam + lam / tau * dt_dl)
        v = w * (y - mu + mu / tau * dt_dm)

        g_att = np.zeros(n)
        g_def = np.zeros(n)
        # attack of home team enters lam; attack of away team enters mu
        np.add.at(g_att, h, u)
        np.add.at(g_att, a, v)
        # mean-centering couples every team by -1/n
        g_att = -(g_att - (u.sum() + v.sum()) / n) + 2 * reg * att
        # defense of away enters lam (negatively); defense of home enters mu
        np.add.at(g_def, a, u)
        np.add.at(g_def, h, v)
        g_def = g_def + 2 * reg * deff
        g_gamma = -np.sum(u * home_flag)
        g_rho = -np.sum(w / tau * dt_dr)

        grad = np.concatenate([g_att, g_def, [g_gamma, g_rho]])
        return nll, grad

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = 0.25   # home advantage start
    p0[2 * n + 1] = -0.05  # rho start
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2)]
    res = minimize(
        objective, p0, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": max_iter, "maxfun": 100000},
    )

    att = res.x[:n]
    att = att - att.mean()
    deff = res.x[n : 2 * n]
    return Params(
        teams=teams,
        attack=att.tolist(),
        defense=deff.tolist(),
        home_adv=float(res.x[2 * n]),
        rho=float(res.x[2 * n + 1]),
        xi=xi,
        reg=reg,
        n_matches=len(matches),
        ref_date=ref,
    )


# --- Prediction --------------------------------------------------------------
def _poisson_pmf(lam: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    return np.exp(-lam) * lam**k / fact


def goal_expectations(
    params: Params, home: str, away: str, neutral: bool = True
) -> tuple[float, float]:
    idx = params.index()
    att = np.array(params.attack)
    deff = np.array(params.defense)
    # Unknown team -> league-average (0 attack, 0 defense): graceful, never crashes.
    ah = att[idx[home]] if home in idx else 0.0
    aa = att[idx[away]] if away in idx else 0.0
    dh = deff[idx[home]] if home in idx else 0.0
    da = deff[idx[away]] if away in idx else 0.0
    gamma = 0.0 if neutral else params.home_adv
    lam = float(np.exp(ah - da + gamma))
    mu = float(np.exp(aa - dh))
    return lam, mu


def predict(
    params: Params, home: str, away: str, neutral: bool = True, max_goals: int = 10
) -> np.ndarray:
    """Goal matrix P[i, j] = P(home scores i, away scores j)."""
    lam, mu = goal_expectations(params, home, away, neutral)
    ph = _poisson_pmf(lam, max_goals)
    pa = _poisson_pmf(mu, max_goals)
    mat = np.outer(ph, pa)
    # Apply DC correction to the four low-score cells, then renormalize.
    rho = params.rho
    mat[0, 0] *= 1.0 - lam * mu * rho
    mat[0, 1] *= 1.0 + lam * rho
    mat[1, 0] *= 1.0 + mu * rho
    mat[1, 1] *= 1.0 - rho
    mat = np.clip(mat, 0.0, None)
    s = mat.sum()
    return mat / s if s > 0 else mat


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) from a goal matrix."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return home, draw, away


def most_likely_score(matrix: np.ndarray) -> tuple[int, int]:
    i, j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(i), int(j)


def expected_goals(matrix: np.ndarray) -> tuple[float, float]:
    n = matrix.shape[0]
    g = np.arange(n)
    return float((matrix.sum(axis=1) * g).sum()), float((matrix.sum(axis=0) * g).sum())
