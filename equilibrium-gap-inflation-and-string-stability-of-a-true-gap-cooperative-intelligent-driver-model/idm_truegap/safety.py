"""Kinematic safety under bounded braking, and a one-dimensional CBF filter.

The filter is the closed-form solution of

    min_u  0.5 (u - u_nom)^2
    s.t.   u in [-b_F, a_max],   h_dot_worst + kappa h >= 0

with the barrier

    h = s - s0 - v^2 / (2 b_F) + v_lead^2 / (2 b_L).

Against a leader who never brakes harder than b_L, the input u = -b_F
renders {h >= 0} forward invariant when the follower can realise that
input immediately. A reaction lag is handled by lengthening the barrier
with v * tau_r; once both actuators are saturated, raising the
feedforward gain does not enlarge the invariant set.
"""

from __future__ import annotations

import numpy as np

from .params import Params, DEFAULT


def safe_gap(v, v_lead, b_f, b_l, s0, tau_r=0.0):
    """Minimum bumper gap that keeps h >= 0."""
    v = np.asarray(v, dtype=float)
    v_lead = np.asarray(v_lead, dtype=float)
    quad = v ** 2 / (2.0 * b_f) - v_lead ** 2 / (2.0 * b_l)
    return s0 + v * tau_r + quad


def alpha_min(v, T, b_lead, tau_r, s0=0.0):
    """Smallest braking ratio alpha = b_follower / b_lead for which the
    constant-time-gap equilibrium lies inside the safe set.

    Returns a value in (0, 1]. If even alpha = 1 is not enough (the time
    gap does not cover the reaction distance), returns a number greater
    than 1, which is not physically attainable by scaling the follower.
    """
    # s0 + v T >= s0 + v tau_r + v^2/2 (1/(alpha b_lead) - 1/b_lead)
    # v (T - tau_r) >= v^2 /(2 b_lead) (1/alpha - 1)
    slack = T - tau_r
    if v <= 1e-6:
        return 0.0
    if slack <= 0.0:
        return np.inf
    # 1/alpha - 1 <= 2 b_lead slack / v
    rhs = 2.0 * b_lead * slack / v
    return 1.0 / (1.0 + rhs)


def min_time_gap(v, alpha, b_lead, tau_r):
    """Smallest T such that the equilibrium is inside the safe set."""
    alpha = np.asarray(alpha, dtype=float)
    extra = np.where(alpha >= 1.0 - 1e-12, 0.0, (1.0 / alpha - 1.0))
    return tau_r + v / (2.0 * b_lead) * extra


def barrier(s, v, v_lead, b_f, b_l, s0, tau_r=0.0):
    return s - safe_gap(v, v_lead, b_f, b_l, s0, tau_r)


def cbf_filter(u_nom, s, v, v_lead, p: Params = DEFAULT, b_f=None, b_l=None, tau_r=0.0):
    """Project the nominal command onto the safe input set.

    Worst-case leader acceleration is -b_l. With the barrier above and
    tau_r = 0, full braking gives h_dot_worst = 0 on h = 0. The filter
    therefore forces

        u <= -b_f + kappa * b_f * h / v

    and clips to [-b_f, a_max]. If h is already negative the upper bound
    falls below -b_f and the constraint is infeasible; the filter then
    applies full braking and reports infeasible=True.
    """
    if b_f is None:
        b_f = p.b_max
    if b_l is None:
        b_l = p.b_max
    h = barrier(s, v, v_lead, b_f, b_l, p.s0, tau_r)
    v_safe = max(float(v), 0.5)
    u_hi = -b_f + p.kappa_cbf * b_f * h / v_safe
    # The lag term -tau_r * u in h_dot makes the inequality affine in u
    # with a more negative coefficient, so the same full-braking vertex
    # remains the most helpful input. The v*tau_r already inside h is the
    # conservative reaction distance; we do not add a second tau_r*u term
    # on top of a plant lag (that would double-count). See the paper.
    infeasible = u_hi < -b_f - 1e-9
    u = min(float(u_nom), float(u_hi))
    u = float(np.clip(u, -b_f, p.a_max))
    return u, float(h), bool(infeasible)
