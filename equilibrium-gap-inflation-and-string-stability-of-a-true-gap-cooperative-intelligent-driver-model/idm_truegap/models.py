"""Car-following accelerations.

Spacing ``s`` is the bumper gap. Relative velocity is
``dv = v - v_lead`` (positive when the follower is closing the gap
from behind by being faster).
"""

from __future__ import annotations

import numpy as np

from .params import Params, DEFAULT


def s_cf(v, p: Params = DEFAULT):
    """Uninflated constant-time-gap bumper gap s0 + v T."""
    return p.s0 + np.asarray(v, dtype=float) * p.T


def s_star(v, dv, p: Params = DEFAULT):
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    return p.s0 + np.maximum(0.0, v * p.T + v * dv / (2.0 * np.sqrt(p.a * p.b)))


def beta_blend(s, v, p: Params = DEFAULT):
    """C-infinity blend in (0, 1). Near 0 in car-following, near 1 in free flow.

    The transition is centred at ``lam * s_cf``, with ``lam > 1``, so a
    vehicle sitting on the uninflated gap is on the flat shoulder of the
    hyperbolic tangent rather than on its steepest slope.
    """
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    xi = p.gamma * (s - p.lam * s_cf(v, p))
    return 0.5 * (1.0 + np.tanh(xi))


def beta_derivatives(s, v, p: Params = DEFAULT):
    """Partial derivatives of beta. sech^2(xi) = 1 - tanh^2(xi)."""
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    xi = p.gamma * (s - p.lam * s_cf(v, p))
    sech2 = 1.0 - np.tanh(xi) ** 2
    db_ds = 0.5 * p.gamma * sech2
    db_dv = -p.lam * p.T * db_ds
    return db_ds, db_dv


def free_ratio(v, v_des, p: Params = DEFAULT):
    v = np.asarray(v, dtype=float)
    return np.clip(v / v_des, 0.0, None) ** p.delta


def idm_accel(s, v, dv, p: Params = DEFAULT, v_des=None):
    """Classical IDM acceleration, before actuator lag and saturation."""
    if v_des is None:
        v_des = p.v0
    s = np.maximum(np.asarray(s, dtype=float), 1e-3)
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    ss = s_star(v, dv, p)
    return p.a * (1.0 - free_ratio(v, v_des, p) - (ss / s) ** 2)


def tg_accel(s, v, dv, p: Params = DEFAULT):
    """True-gap IDM: free-speed term multiplied by beta(s, v)."""
    s = np.maximum(np.asarray(s, dtype=float), 1e-3)
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    ss = s_star(v, dv, p)
    bta = beta_blend(s, v, p)
    return p.a * (1.0 - bta * free_ratio(v, p.v0, p) - (ss / s) ** 2)


def virtual_speed(s, v, p: Params = DEFAULT, capped=True):
    """Virtual desired speed. Uncapped, this reproduces the beta model exactly.

    (v / v_vir)^delta = beta * (v / v0)^delta when
    v_vir = v0 * beta^(-1/delta). A finite supervisory cap breaks the
    identity and leaves a residual equilibrium gap.
    """
    bta = np.maximum(beta_blend(s, v, p), 1e-12)
    v_vir = p.v0 * bta ** (-1.0 / p.delta)
    if capped:
        v_vir = np.minimum(v_vir, p.v_cap)
    return v_vir


def virtual_accel(s, v, dv, p: Params = DEFAULT, capped=True):
    v_vir = virtual_speed(s, v, p, capped=capped)
    return idm_accel(s, v, dv, p, v_des=v_vir)


def eidm_accel(s, v, dv, a_lead, p: Params = DEFAULT):
    """Enhanced IDM (Kesting, Treiber, Helbing, 2010).

    The coolness modification is inactive at equilibrium (both the IDM
    and the constant-acceleration heuristic are zero), so the equilibrium
    gap is identical to the classical IDM.
    """
    s_use = np.maximum(np.asarray(s, dtype=float), 1e-3)
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    a_lead = np.asarray(a_lead, dtype=float)
    a_idm = idm_accel(s_use, v, dv, p)
    v_lead = v - dv
    a_l = np.minimum(a_lead, p.a)
    closing = v_lead * (v_lead - v) <= -2.0 * s_use * a_l
    denom = v_lead ** 2 - 2.0 * s_use * a_l
    a_braking = np.where(
        np.abs(denom) < 1e-9,
        a_l,
        v ** 2 * a_l / np.where(np.abs(denom) < 1e-9, 1.0, denom),
    )
    a_else = a_l - (np.maximum(v - v_lead, 0.0) ** 2) / (2.0 * s_use)
    a_cah = np.where(closing, a_braking, a_else)
    cool = (1.0 - p.c_eidm) * a_idm + p.c_eidm * (
        a_cah + p.b * np.tanh((a_idm - a_cah) / p.b)
    )
    return np.where(a_idm >= a_cah, a_idm, cool)


def linear_cacc_accel(s, v, v_lead, u_lead, p: Params = DEFAULT, kp=None, kd=None, ka=None):
    """PATH-style linear CACC: PD on the time-gap error plus feedforward.

    u = kp (s - s0 - T v) + kd (v_lead - v) + ka * u_lead_delayed.
    Equilibrium is exactly the uninflated gap, for any kp, kd, ka.
    """
    if kp is None:
        kp = p.kp
    if kd is None:
        kd = p.kd
    if ka is None:
        ka = p.ka
    s = np.asarray(s, dtype=float)
    v = np.asarray(v, dtype=float)
    v_lead = np.asarray(v_lead, dtype=float)
    u_lead = np.asarray(u_lead, dtype=float)
    gap_error = s - p.s0 - p.T * v
    return kp * gap_error + kd * (v_lead - v) + ka * u_lead


def saturate(u, p: Params = DEFAULT, b_max=None):
    if b_max is None:
        b_max = p.b_max
    return np.clip(u, -b_max, p.a_max)
