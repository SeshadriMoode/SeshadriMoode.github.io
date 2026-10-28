"""Narrow checks asked for after review. Does not rerun the full study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import _own_gap, _run
from .params import DEFAULT, with_updates
from .safety import barrier
from .simulate import Comm, disturbance_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "truegap"
FIG = ROOT / "figures" / "truegap"


def _brake(decel, t0=8.0, dur=4.0):
    def leader(t):
        if t0 <= t < t0 + dur:
            return -decel
        return 0.0
    return leader


def feasible_barrier():
    """Find a brake on which h starts non-negative and the filter is active."""
    v = 25.0
    T = 0.6
    pt = with_updates(DEFAULT, T=T, ka=1.0)
    gap = _own_gap("truegap", v, pt)
    rows = []
    chosen = None
    for alpha in (0.85, 0.90, 1.0):
        for decel, dur in ((5.0, 4.0), (5.0, 6.0), (3.5, 8.0)):
            for use_cbf in (False, True):
                b = np.full(4, pt.b_max)
                b[1] = alpha * pt.b_max
                res = _run(
                    "truegap", _brake(decel, dur=dur), 28.0, gap, v, pt,
                    Comm(model="ideal", tau=0.1), None, None, 1.0,
                    n_veh=4, dt=0.01, b_max=b, use_cbf=use_cbf, tau_r=0.1,
                )
                h = barrier(
                    res.s[:, 1], res.v[:, 1], res.v[:, 0],
                    b_f=float(b[1]), b_l=pt.b_max, s0=pt.s0, tau_r=0.1,
                )
                # Commands after the lag has something to track.
                u = res.u[:, 1]
                row = {
                    "alpha": alpha,
                    "decel": decel,
                    "dur": dur,
                    "cbf": use_cbf,
                    "h0": float(h[0]),
                    "min_h": float(np.min(h)),
                    "min_gap": float(np.min(res.s[:, 1])),
                    "max_abs_u": float(np.max(np.abs(u))),
                    "frac_at_bound": float(np.mean(np.abs(u + b[1]) < 1e-3)),
                }
                rows.append(row)
                print(row)
    # Prefer nominal min_h < 0 and filtered min_h greater, with h0 > 0.
    for alpha, decel, dur in (
        (a, d, du)
        for a in (0.85, 0.90, 1.0)
        for d, du in ((5.0, 4.0), (5.0, 6.0), (3.5, 8.0))
    ):
        nom = next(r for r in rows if r["alpha"] == alpha and r["decel"] == decel and r["dur"] == dur and not r["cbf"])
        cbf = next(r for r in rows if r["alpha"] == alpha and r["decel"] == decel and r["dur"] == dur and r["cbf"])
        if nom["h0"] > 0.0 and nom["min_h"] < 0.0 and cbf["min_h"] > nom["min_h"] + 0.2:
            chosen = (alpha, decel, dur, nom, cbf)
            break
    return rows, chosen, gap, v, pt


def plot_barrier(alpha, decel, dur, gap, v, pt):
    b = np.full(4, pt.b_max)
    b[1] = alpha * pt.b_max
    series = {}
    for use_cbf, name in ((False, "nominal"), (True, "cbf")):
        res = _run(
            "truegap", _brake(decel, dur=dur), 28.0, gap, v, pt,
            Comm(model="ideal", tau=0.1), None, None, 1.0,
            n_veh=4, dt=0.01, b_max=b, use_cbf=use_cbf, tau_r=0.1,
        )
        h = barrier(
            res.s[:, 1], res.v[:, 1], res.v[:, 0],
            b_f=float(b[1]), b_l=pt.b_max, s0=pt.s0, tau_r=0.1,
        )
        series[name] = (res.t, h, res.s[:, 1], res.u[:, 1])
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.spines.top": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    t, h, _, _ = series["nominal"]
    ax.plot(t, h, color="#0072B2", ls="-", lw=1.5, label="nominal")
    t, h, _, _ = series["cbf"]
    ax.plot(t, h, color="#D55E00", ls="--", lw=1.5, label="barrier filter")
    ax.axhline(0.0, color="#000000", lw=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"barrier $h$ (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig11_barrier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    out = {}
    for name, (t, h, s, u) in series.items():
        out[name] = {
            "h0": float(h[0]),
            "min_h": float(np.min(h)),
            "min_gap": float(np.min(s)),
            "max_abs_u": float(np.max(np.abs(u))),
        }
    return out


def blend_crossing():
    """Relaxation from above the blend spike. Leader speed is constant."""
    v = 25.0
    T = 0.6
    s_ctg = 2.0 + v * T
    rows = []
    for gamma in (0.10, 0.35, 1.00):
        pt = with_updates(DEFAULT, T=T, ka=1.0, gamma=gamma)
        res = _run(
            "truegap", lambda t: 0.0, 40.0, 1.9 * s_ctg, v, pt,
            Comm(model="ideal", tau=0.0), None, None, 1.0,
            n_veh=2, dt=0.01, name="relax",
        )
        s = res.s[:, 1]
        u = res.u[:, 1]
        spike = pt.lam * (pt.s0 + res.v[:, 1] * pt.T)
        crossed = bool(np.any(s > spike) and np.any(s < spike))
        rows.append({
            "gamma": gamma,
            "s0_gap": float(s[0]),
            "s_final": float(np.mean(s[-200:])),
            "spike_at_v": float(pt.lam * s_ctg),
            "crossed": crossed,
            "max_abs_u": float(np.max(np.abs(u))),
            "max_abs_a": float(np.max(np.abs(res.a[:, 1]))),
            "time_to_within_1m": float(res.t[np.argmax(np.abs(s - s_ctg) < 1.0)]) if np.any(np.abs(s - s_ctg) < 1.0) else None,
        })
        print(rows[-1])
    return rows


def small_wave():
    v = 25.0
    T = 0.6
    pt = with_updates(DEFAULT, T=T, ka=1.0)
    amp = 0.05

    def sinusoid(t, t0=15.0, t1=35.0, freq=0.70):
        if t0 <= t <= t1:
            w = 2.0 * np.pi * freq
            return amp * w * np.cos(w * (t - t0))
        return 0.0

    specs = (
        ("idm", _own_gap("idm", v, pt), None, None, 0.0),
        ("truegap", _own_gap("truegap", v, pt), None, None, 1.0),
        ("ploeg", pt.s0 + v * pt.T, 0.2, 0.7, 1.0),
        ("soft", pt.s0 + v * pt.T, 0.2, 0.4, 1.0),
    )
    rows = {}
    for kind, gap, kp, kd, ka in specs:
        sim_kind = "linear" if kind in ("ploeg", "soft") else kind
        res = _run(
            sim_kind, sinusoid, 45.0, gap, v, pt,
            Comm(model="ideal", tau=0.1), kp, kd, ka,
            n_veh=10, dt=0.01, name=kind,
        )
        met = disturbance_metrics(res, v, 15.0, 40.0)
        m = (res.t >= 15.0) & (res.t <= 35.0)
        rows[kind] = {
            "l2_tail": met["l2_tail"],
            "peak_abs_a": float(np.max(np.abs(res.a[m, 1:]))),
            "peak_abs_u": float(np.max(np.abs(res.u[m, 1:]))),
        }
        print(kind, rows[kind])
    return {"amp_mps": amp, "controllers": rows}


def plot_blend():
    v = 25.0
    T = 0.6
    s_ctg = 2.0 + v * T
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    styles = {
        0.10: dict(color="#0072B2", ls="-", lw=1.5),
        0.35: dict(color="#D55E00", ls="--", lw=1.5),
        1.00: dict(color="#009E73", ls=":", lw=1.8),
    }
    for gamma, style in styles.items():
        pt = with_updates(DEFAULT, T=T, ka=1.0, gamma=gamma)
        res = _run(
            "truegap", lambda t: 0.0, 40.0, 1.9 * s_ctg, v, pt,
            Comm(model="ideal", tau=0.0), None, None, 1.0,
            n_veh=2, dt=0.01, name="relax",
        )
        ax.plot(res.t, res.s[:, 1], label=rf"$\gamma={gamma:.2f}$", **style)
    ax.axhline(s_ctg, color="#000000", ls=(0, (1, 1)), lw=1.0, label="policy gap")
    ax.axhline(1.6 * s_ctg, color="0.45", ls=(0, (3, 1, 1, 1)), lw=1.0, label="blend centre")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("bumper gap (m)")
    ax.set_xlim(0, 40)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig12_blend_relax.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def root_stability():
    """Local stability of every true-gap root, and which gap a follower reaches."""
    from scipy.optimize import brentq

    from .linearize import free_ratio, gains_at, hinf_norm
    from .models import beta_blend, s_cf
    from .simulate import simulate

    p = DEFAULT

    def roots_of(v):
        scf = float(s_cf(v, p))
        phi = float(free_ratio(v, p.v0, p))

        def residual(s):
            return 1.0 - float(beta_blend(s, v, p)) * phi - (scf / s) ** 2

        grid = np.linspace(scf, 400.0, 80000)
        values = np.array([residual(s) for s in grid])
        flips = np.where(np.diff(np.sign(values)))[0]
        found = []
        for i in flips:
            a, b = float(grid[i]), float(grid[i + 1])
            if residual(a) * residual(b) < 0.0:
                found.append(float(brentq(residual, a, b)))
        return found

    def hurwitz(fs, fv, fdv, tau):
        real = np.real(np.roots([tau, 1.0, -(fv + fdv), fs]))
        return bool(np.all(real < -1e-8)), float(np.max(real))

    speeds = np.round(np.arange(30.8, 33.21, 0.2), 2)
    rows = []
    for v in speeds:
        for s in roots_of(float(v)):
            fs, fv, fdv = gains_at(s, float(v), p, "truegap")
            h0, r0 = hurwitz(fs, fv, fdv, 0.0)
            h1, r1 = hurwitz(fs, fv, fdv, 0.1)
            rows.append({
                "v_mps": float(v),
                "gap_m": s,
                "beta": float(beta_blend(s, float(v), p)),
                "f_s": fs,
                "hurwitz_tau0": h0,
                "max_real_tau0": r0,
                "hurwitz_tau0p1": h1,
                "max_real_tau0p1": r1,
            })

    def finish(v, gap, t_end):
        out = simulate(
            lambda t: 0.0, t_end, gap, v,
            n_veh=2, p=p, kind="truegap", ka=0.0, dt=0.02,
        )
        return float(out.s[-1, 1])

    v = 32.0
    gaps = roots_of(v)
    fs, fv, fdv = gains_at(gaps[2], v, p, "truegap")
    return {
        "grid_mps": [float(s) for s in speeds],
        "rows": rows,
        "example_v_mps": v,
        "roots_m": gaps,
        "settles_from_below_m": finish(v, gaps[1] - 1.0, 250.0),
        "settles_from_above_m": finish(v, gaps[1] + 1.0, 250.0),
        "large_root_hinf_ka0_tau0": hinf_norm(fs, fv, fdv, 0.0, 0.0, 0.0),
        "large_root_hinf_ka0_tau0p1": hinf_norm(fs, fv, fdv, 0.0, 0.0, 0.1),
    }


def main():
    rows, _, gap, v, pt = feasible_barrier()
    # Equal braking bounds: the cruise command is left alone, and a leader
    # brake at the mechanical bound drives the nominal barrier negative.
    plotted = plot_barrier(1.0, 5.0, 6.0, gap, v, pt)
    plot_blend()
    summary = {
        "barrier_grid": rows,
        "barrier_case": {"alpha": 1.0, "decel": 5.0, "dur": 6.0, "plotted": plotted},
        "blend": blend_crossing(),
        "small_wave": small_wave(),
        "roots": root_stability(),
    }
    wave = summary["small_wave"]["controllers"]["truegap"]
    print("per-link", wave["l2_tail"] ** (1.0 / 9.0))
    print("BARRIER", plotted)
    (OUT / "author_checks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
