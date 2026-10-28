"""Platoon integration with actuator lag, delay, and packet loss.

The leader acceleration is exogenous. Followers are integrated with a
semi-implicit Euler step of size ``dt``. Communication is a zero-order
hold at ``comm_dt``: a delivered packet carries the sender's command
from ``tau`` seconds earlier; a lost packet holds the last delivered
command.

Packet processes
----------------
bernoulli: each packet is delivered independently with probability ``p``.
gilbert:   two-state Markov chain. In the good state a packet is
           delivered with probability ``p_good``; in the bad state, with
           ``p_bad``. Per-packet transition probabilities are ``p_gb``
           (good to bad) and ``p_bg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import (
    eidm_accel,
    idm_accel,
    linear_cacc_accel,
    saturate,
    tg_accel,
    virtual_accel,
)
from .params import Params, DEFAULT
from .safety import cbf_filter


@dataclass
class Comm:
    tau: float = 0.0
    p: float = 1.0
    model: str = "bernoulli"   # or "gilbert" or "none"
    comm_dt: float = 0.1
    p_good: float = 0.98
    p_bad: float = 0.20
    p_gb: float = 0.05
    p_bg: float = 0.30
    seed: int = 0


@dataclass
class SimResult:
    t: np.ndarray
    x: np.ndarray          # (n_steps, n_veh) front-bumper position
    v: np.ndarray
    a: np.ndarray          # realised acceleration
    u: np.ndarray          # commanded acceleration after saturation / CBF
    s: np.ndarray          # bumper gap of followers; column 0 is nan
    delivered: np.ndarray  # fraction of packets delivered on each link
    collision: bool
    min_gap: float
    name: str


def _nominal(kind, s, v, dv, a_lead, u_ff, p, kp, kd, ka):
    if kind == "idm":
        return idm_accel(s, v, dv, p)
    if kind == "cidm":
        return idm_accel(s, v, dv, p) + ka * u_ff
    if kind == "eidm":
        return eidm_accel(s, v, dv, a_lead, p)
    if kind == "truegap":
        return tg_accel(s, v, dv, p) + ka * u_ff
    if kind == "virtual":
        return virtual_accel(s, v, dv, p, capped=True) + ka * u_ff
    if kind == "linear":
        v_lead = v - dv
        return linear_cacc_accel(s, v, v_lead, u_ff, p, kp=kp, kd=kd, ka=ka)
    raise ValueError(kind)


def simulate(
    leader_accel,
    t_end,
    gap0,
    v0,
    n_veh=10,
    p: Params = DEFAULT,
    kind="truegap",
    comm: Comm | None = None,
    dt=0.01,
    b_max=None,
    use_cbf=False,
    tau_r=0.0,
    kp=None,
    kd=None,
    ka=None,
    name="",
    gap_scale=None,
):
    """Simulate one platoon.

    ``leader_accel(t)`` returns the leader's realised acceleration.
    ``gap0`` is the initial bumper gap of every follower (scalar or
    length n_veh-1). ``v0`` is the common initial speed.
    ``b_max`` may be a scalar or an array of length n_veh (leader entry
    is ignored). ``gap_scale(t, s, v)`` optionally overwrites follower 0's
    gap at a cut-in instant by returning a new gap, or None to leave it.
    """
    if comm is None:
        comm = Comm()
    if ka is None:
        ka = p.ka
    n = int(n_veh)
    steps = int(np.round(t_end / dt))
    t = np.arange(steps) * dt
    rng = np.random.default_rng(comm.seed)

    x = np.zeros((steps, n))
    v = np.zeros((steps, n))
    a = np.zeros((steps, n))
    u = np.zeros((steps, n))
    s_hist = np.full((steps, n), np.nan)

    gaps = np.full(n - 1, gap0, dtype=float) if np.isscalar(gap0) else np.asarray(gap0, dtype=float)
    # Place the leader at 0 and string followers behind it.
    x[0, 0] = 0.0
    v[0, 0] = v0
    for i in range(1, n):
        x[0, i] = x[0, i - 1] - p.length - gaps[i - 1]
        v[0, i] = v0

    if b_max is None:
        b_max = np.full(n, p.b_max)
    else:
        b_max = np.broadcast_to(np.asarray(b_max, dtype=float), (n,)).copy()

    delay_steps = int(np.round(comm.tau / dt))
    comm_steps = max(1, int(np.round(comm.comm_dt / dt)))
    held = np.zeros(n)
    link_state = np.zeros(n, dtype=int)  # 0 good, 1 bad; index is the sender
    n_attempt = np.zeros(n)
    n_deliver = np.zeros(n)
    cmd_hist = np.zeros((steps, n))
    cut_done = False

    collision = False
    for k in range(steps - 1):
        tk = t[k]
        a_lead_cmd = float(leader_accel(tk))
        # Do not command a speed reversal through zero. Realised leader
        # acceleration a[:, 0] is a state and is not overwritten here.
        if v[k, 0] <= 1e-8 and a_lead_cmd < 0.0:
            a_lead_cmd = 0.0
        u[k, 0] = a_lead_cmd
        cmd_hist[k, 0] = u[k, 0]

        # Follower commands from the current state.
        for i in range(1, n):
            s_i = x[k, i - 1] - x[k, i] - p.length
            s_hist[k, i] = s_i
            if s_i < 0.0:
                collision = True
            dv = v[k, i] - v[k, i - 1]
            u_ff = held[i - 1]
            u_nom = _nominal(
                kind, max(s_i, 0.05), v[k, i], dv, a[k, i - 1], u_ff, p, kp, kd, ka
            )
            if use_cbf and kind in ("truegap", "idm", "cidm", "virtual", "eidm"):
                u_cmd, _, _ = cbf_filter(
                    u_nom, s_i, v[k, i], v[k, i - 1],
                    p, b_f=b_max[i], b_l=b_max[i - 1] if i > 1 else p.b_max,
                    tau_r=tau_r,
                )
            else:
                u_cmd = float(saturate(u_nom, p, b_max=b_max[i]))
            u[k, i] = u_cmd
            cmd_hist[k, i] = u_cmd

        # Ideal channel: every integration step sees the command from tau ago.
        # Packet channels: a zero-order hold updated at comm_dt.
        src = k - delay_steps
        if comm.model == "ideal":
            if src >= 0:
                held[:] = cmd_hist[src, :]
                n_deliver += 1
                n_attempt += 1
        elif k % comm_steps == 0:
            for i in range(n):
                n_attempt[i] += 1
                deliver = True
                if comm.model == "bernoulli":
                    deliver = rng.random() < comm.p
                elif comm.model == "gilbert":
                    if link_state[i] == 0:
                        deliver = bool(rng.random() < comm.p_good)
                        if rng.random() < comm.p_gb:
                            link_state[i] = 1
                    else:
                        deliver = bool(rng.random() < comm.p_bad)
                        if rng.random() < comm.p_bg:
                            link_state[i] = 0
                if deliver:
                    held[i] = cmd_hist[src, i] if src >= 0 else 0.0
                    n_deliver[i] += 1

        # One-shot cut-in: gap_scale returns a new bumper gap once, else None.
        if gap_scale is not None and not cut_done:
            new_gap = gap_scale(tk, s_hist[k, 1], v[k, 1])
            if new_gap is not None:
                x[k, 1] = x[k, 0] - p.length - float(new_gap)
                cut_done = True

        # Same first-order lag on the leader, so the broadcast command and
        # the realised acceleration are related exactly as in the linear
        # transfer function.
        a[k + 1, 0] = a[k, 0] + dt * (u[k, 0] - a[k, 0]) / p.tau_act
        v[k + 1, 0] = max(0.0, v[k, 0] + dt * a[k + 1, 0])
        x[k + 1, 0] = x[k, 0] + dt * v[k + 1, 0]
        for i in range(1, n):
            a[k + 1, i] = a[k, i] + dt * (u[k, i] - a[k, i]) / p.tau_act
            v[k + 1, i] = max(0.0, v[k, i] + dt * a[k + 1, i])
            x[k + 1, i] = x[k, i] + dt * v[k + 1, i]

    # Last sample of gaps.
    for i in range(1, n):
        s_hist[-1, i] = x[-1, i - 1] - x[-1, i] - p.length
    min_gap = float(np.nanmin(s_hist[:, 1:]))
    if min_gap < 0.0:
        collision = True
    delivered = np.divide(n_deliver, np.maximum(n_attempt, 1))
    return SimResult(
        t=t, x=x, v=v, a=a, u=u, s=s_hist,
        delivered=delivered, collision=collision, min_gap=min_gap,
        name=name or kind,
    )


def rms(z, mask):
    z = np.asarray(z)[mask]
    return float(np.sqrt(np.mean(z ** 2)))


def disturbance_metrics(res: SimResult, v_eq, t0, t1):
    """L2 velocity-amplification ratios over [t0, t1], and comfort metrics."""
    m = (res.t >= t0) & (res.t <= t1)
    e0 = rms(res.v[:, 0] - v_eq, m)
    ratios = []
    for i in range(res.v.shape[1]):
        ei = rms(res.v[:, i] - v_eq, m)
        ratios.append(ei / e0 if e0 > 1e-6 else np.nan)
    finite = [r for r in ratios[1:] if np.isfinite(r)]
    # Jerk from realised acceleration.
    dt = res.t[1] - res.t[0]
    jerk = np.diff(res.a, axis=0) / dt
    m_j = (res.t[:-1] >= t0) & (res.t[:-1] <= t1)
    return {
        "l2_ratio": np.asarray(ratios),
        "l2_tail": float(ratios[-1]) if np.isfinite(ratios[-1]) else float("nan"),
        "l2_max": float(max(finite)) if finite else float("nan"),
        "min_gap": res.min_gap,
        "collision": res.collision,
        "max_brake": float(np.min(res.a[m])),
        "max_jerk": float(np.max(np.abs(jerk[m_j]))),
        "rms_jerk": float(np.sqrt(np.mean(jerk[m_j] ** 2))),
    }
