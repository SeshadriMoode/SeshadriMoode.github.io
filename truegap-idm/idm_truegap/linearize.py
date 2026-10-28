"""Equilibrium gaps, linear gains, and the string-stability transfer function.

Sign convention matches models.py: partial derivatives of the follower
acceleration. At a car-following equilibrium one has f_s > 0, f_v < 0
and f_dv < 0.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .models import beta_blend, beta_derivatives, free_ratio, idm_accel, tg_accel
from .params import Params, DEFAULT


def idm_equilibrium_gap(v, p: Params = DEFAULT):
    """Closed-form IDM bumper gap. Infinite as v approaches v0 from below."""
    v = np.asarray(v, dtype=float)
    phi = free_ratio(v, p.v0, p)
    out = np.full(np.shape(v), np.inf, dtype=float)
    ok = phi < 1.0 - 1e-12
    scf = p.s0 + v * p.T
    out = np.where(ok, scf / np.sqrt(1.0 - np.minimum(phi, 1.0 - 1e-15)), np.inf)
    return out


def tg_equilibrium_gap(v, p: Params = DEFAULT):
    """Implicit true-gap equilibrium. Solves the fixed point in the bumper gap.

    Because beta is not exactly zero at s = s0 + v T, a residual inflation
    remains. It is exponentially small when the operating point sits
    several blend-lengths below the transition.
    """
    v = float(v)
    phi = float(free_ratio(v, p.v0, p))
    scf = p.s0 + v * p.T
    if phi >= 1.0:
        return np.inf

    def residual(s):
        return 1.0 - float(beta_blend(s, v, p)) * phi - (scf / s) ** 2

    # residual(scf) = -beta(scf)*phi <= 0, and residual -> 1 - phi > 0
    # as s -> infinity, so a root exists in (scf, infinity) unless beta is 0,
    # in which case the root is scf itself.
    if residual(scf) >= -1e-14:
        return scf
    hi = max(scf * 1.5, scf + 5.0)
    while residual(hi) < 0.0 and hi < scf * 50.0:
        hi *= 1.5
    if residual(hi) < 0.0:
        return np.inf
    return float(brentq(residual, scf, hi, xtol=1e-10))


def flow(v, gap, p: Params = DEFAULT):
    """Lane flow [veh/h] of a homogeneous string. ``gap`` is the bumper gap."""
    v = np.asarray(v, dtype=float)
    gap = np.asarray(gap, dtype=float)
    return 3600.0 * v / (gap + p.length)


def gains_at(s, v, p: Params = DEFAULT, kind="truegap", drop_beta_derivatives=False):
    """Analytic partial derivatives (f_s, f_v, f_dv) at dv = 0.

    ``kind='idm'`` forces beta = 1 with zero derivatives (classical IDM).
    ``kind='truegap'`` uses beta(s, v). Setting ``drop_beta_derivatives``
    reproduces the incomplete linearization that neglects d(beta)/ds
    and d(beta)/dv.
    """
    s = float(s)
    v = float(v)
    dv = 0.0
    if kind == "idm":
        bta = 1.0
        db_ds = 0.0
        db_dv = 0.0
    elif kind == "truegap":
        bta = float(beta_blend(s, v, p))
        db_ds, db_dv = beta_derivatives(s, v, p)
        db_ds = float(db_ds)
        db_dv = float(db_dv)
        if drop_beta_derivatives:
            db_ds = 0.0
            db_dv = 0.0
    else:
        raise ValueError(kind)

    phi = float(free_ratio(v, p.v0, p))
    dphi = p.delta * phi / v if v > 1e-6 else 0.0
    # s* at dv = 0 equals s0 + v T; the positive-part kink is inactive
    # whenever that quantity is positive, which it is for v >= 0.
    ss = p.s0 + v * p.T
    dss_dv = p.T
    dss_ddv = v / (2.0 * np.sqrt(p.a * p.b))

    f_s = p.a * (-phi * db_ds + 2.0 * ss ** 2 / s ** 3)
    f_v = p.a * (-db_dv * phi - bta * dphi - 2.0 * (ss / s ** 2) * dss_dv)
    f_dv = p.a * (-2.0 * (ss / s ** 2) * dss_ddv)
    return f_s, f_v, f_dv


def finite_difference_gains(s, v, p: Params = DEFAULT, kind="truegap", eps=1e-5):
    """Central differences of the acceleration, for verification."""
    accel = tg_accel if kind == "truegap" else idm_accel

    def a(ss, vv, dv):
        return float(accel(ss, vv, dv, p))

    f_s = (a(s + eps, v, 0.0) - a(s - eps, v, 0.0)) / (2 * eps)
    f_v = (a(s, v + eps, 0.0) - a(s, v - eps, 0.0)) / (2 * eps)
    f_dv = (a(s, v, eps) - a(s, v, -eps)) / (2 * eps)
    return f_s, f_v, f_dv


def transfer(omega, f_s, f_v, f_dv, ka, tau, tau_act):
    """Velocity (and command) transfer function Gamma(j omega) = U_i / U_{i-1}.

    Plant: tau_act * a_dot + a = u, with u the acceleration command.
    Feedforward enters as ka * exp(-tau s) * u_{i-1}, i.e. the broadcast
    command, not the lagged realisation. The same expression is the
    transfer function from velocity perturbations, because every vehicle
    has the same lag.

    Numerator:  f_s - s f_dv + ka (tau_act s + 1) s^2 exp(-tau s)
    Denominator: tau_act s^3 + s^2 - s (f_v + f_dv) + f_s
    with s = j omega.
    """
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * w
    delay = np.exp(-tau * s)
    num = f_s - s * f_dv + ka * (tau_act * s + 1.0) * s ** 2 * delay
    den = tau_act * s ** 3 + s ** 2 - s * (f_v + f_dv) + f_s
    g = num / den
    return g[0] if np.ndim(omega) == 0 else g


def hinf_norm(f_s, f_v, f_dv, ka, tau, tau_act, w_max=30.0, n=4000):
    """Peak |Gamma(j omega)| on (0, w_max]. Returns +inf if a pole is unstable."""
    if tau_act > 0.0:
        # s^3 + (1/tau) s^2 + (-(fv+fdv)/tau) s + fs/tau
        coeffs = [
            1.0,
            1.0 / tau_act,
            -(f_v + f_dv) / tau_act,
            f_s / tau_act,
        ]
    else:
        coeffs = [1.0, -(f_v + f_dv), f_s]
    roots = np.roots(coeffs)
    if np.any(np.real(roots) >= 1e-7):
        return np.inf
    w = np.logspace(-3, np.log10(w_max), n)
    mag = np.abs(transfer(w, f_s, f_v, f_dv, ka, tau, tau_act))
    return float(np.max(mag))


def ka_min_delay_free(f_s, f_v, f_dv):
    """Minimum feedforward gain for ||Gamma||_inf <= 1 at tau = tau_act = 0.

    Derived by requiring |N(j omega)|^2 <= |D(j omega)|^2 for every omega.
    The resulting condition is necessary and sufficient on the lag-free,
    delay-free plant, together with the upper bound ka <= 1 (otherwise
    |Gamma| -> ka > 1 at high frequency).

        ka >= 1 - (f_v^2 + 2 f_v f_dv) / (2 f_s)

    The right-hand side is dimensionless. If it exceeds 1, no admissible
    feedforward gain string-stabilises the linearisation.
    """
    if f_s <= 0.0:
        return np.inf
    return 1.0 - (f_v ** 2 + 2.0 * f_v * f_dv) / (2.0 * f_s)


def tau_max(f_s, f_v, f_dv, ka, tau_act, t_hi=1.5, tol=1e-3, level=1.0):
    """Largest delay with ||Gamma||_inf <= ``level``.

    ``level=1`` is strict L2 string stability. A driveline lag can push the
    norm slightly above 1 even at zero communication delay; a ``level``
    above 1 is then a stated tolerance, not a strict certificate.
    """
    base = hinf_norm(f_s, f_v, f_dv, ka, 0.0, tau_act, n=1500)
    if base > level + 1e-3:
        return 0.0
    if hinf_norm(f_s, f_v, f_dv, ka, t_hi, tau_act, n=1200) <= level + 1e-3:
        return t_hi
    lo, hi = 0.0, t_hi
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if hinf_norm(f_s, f_v, f_dv, ka, mid, tau_act, n=1200) <= level + 1e-3:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(lo)


def linear_cacc_transfer(omega, kp, kd, ka, T, tau, tau_act):
    """Transfer function of the PATH-style PD + feedforward law.

    Denominator tau_act s^3 + s^2 + (kd + kp T) s + kp
    Numerator    kp + kd s + ka (tau_act s + 1) s^2 exp(-tau s)
    """
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    s = 1j * w
    num = kp + kd * s + ka * (tau_act * s + 1.0) * s ** 2 * np.exp(-tau * s)
    den = tau_act * s ** 3 + s ** 2 + (kd + kp * T) * s + kp
    g = num / den
    return g[0] if np.ndim(omega) == 0 else g


def linear_cacc_hinf(kp, kd, ka, T, tau, tau_act, w_max=30.0, n=3000):
    if tau_act <= 0.0:
        coeffs = [1.0, kd + kp * T, kp]
    else:
        coeffs = [tau_act, 1.0, kd + kp * T, kp]
    roots = np.roots(coeffs)
    if np.any(np.real(roots) >= 1e-7):
        return np.inf
    w = np.logspace(-3, np.log10(w_max), n)
    mag = np.abs(linear_cacc_transfer(w, kp, kd, ka, T, tau, tau_act))
    return float(np.max(mag))


def tune_linear_cacc(T, tau, tau_act, ka=1.0):
    """Smallest string-stability norm in a PD family with a gain floor.

    The floor (kp >= 0.2, kd >= 0.4) excludes the trivial law that copies
    the leader command and does not regulate the gap. Inside this family
    the norm typically decreases as the gains approach the floor: feedforward
    already carries the string, and extra feedback adds phase. This is not a
    mixed-sensitivity H-infinity synthesis.
    """
    best = None
    kps = np.linspace(0.2, 1.2, 12)
    kds = np.linspace(0.4, 1.6, 12)
    for kp in kps:
        for kd in kds:
            val = linear_cacc_hinf(float(kp), float(kd), ka, T, tau, tau_act, n=900)
            if best is None or val < best[0]:
                best = (val, float(kp), float(kd))
    _, kp0, kd0 = best
    for kp in np.linspace(max(0.2, kp0 - 0.08), min(1.2, kp0 + 0.08), 7):
        for kd in np.linspace(max(0.4, kd0 - 0.1), min(1.6, kd0 + 0.1), 7):
            val = linear_cacc_hinf(float(kp), float(kd), ka, T, tau, tau_act, n=1800)
            if val < best[0]:
                best = (val, float(kp), float(kd))
    return {"hinf": best[0], "kp": best[1], "kd": best[2], "ka": ka, "T": T, "tau": tau}
