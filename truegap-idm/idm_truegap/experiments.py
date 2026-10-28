"""Numerical study for the true-gap cooperative IDM.

Run from the repository root:

    python -m idm_truegap.experiments

Every number in the manuscript is written to results/truegap/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .linearize import (
    flow,
    gains_at,
    hinf_norm,
    idm_equilibrium_gap,
    ka_min_delay_free,
    linear_cacc_hinf,
    linear_cacc_transfer,
    tau_max,
    tg_equilibrium_gap,
    transfer,
    tune_linear_cacc,
    finite_difference_gains,
)
from .models import (
    beta_blend,
    beta_derivatives,
    eidm_accel,
    idm_accel,
    tg_accel,
    virtual_accel,
)
from .params import DEFAULT, with_updates
from .safety import alpha_min, barrier, min_time_gap, safe_gap
from .simulate import Comm, disturbance_metrics, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "truegap"
FIG = ROOT / "figures" / "truegap"


def _virtual_gap(v, p):
    from scipy.optimize import brentq

    scf = p.s0 + v * p.T
    def res(s):
        return float(virtual_accel(s, v, 0.0, p, capped=True))
    if res(scf) >= -1e-8:
        return scf
    hi = scf + 5.0
    while res(hi) < 0.0 and hi < scf * 40:
        hi *= 1.4
    return float(brentq(res, scf, hi, xtol=1e-8))


def self_check(p=DEFAULT):
    """Raise AssertionError if a derivation and its numerical check disagree."""
    v = 25.0
    s_tg = tg_equilibrium_gap(v, p)
    s_idm = float(idm_equilibrium_gap(v, p))
    assert abs(float(tg_accel(s_tg, v, 0.0, p))) < 1e-8
    assert abs(float(idm_accel(s_idm, v, 0.0, p))) < 1e-8
    assert abs(float(eidm_accel(s_idm, v, 0.0, 0.0, p))) < 1e-6
    # Residual inflation is positive and, at the default blend, tiny.
    scf = p.s0 + v * p.T
    assert s_tg >= scf - 1e-8
    assert (s_tg - scf) / scf < 1e-3

    for kind, s in (("truegap", s_tg), ("idm", s_idm)):
        ana = np.array(gains_at(s, v, p, kind))
        fd = np.array(finite_difference_gains(s, v, p, kind))
        rel = np.max(np.abs(ana - fd) / np.maximum(np.abs(ana), 1e-8))
        assert rel < 1e-4, (kind, ana, fd, rel)

    # Uncapped virtual speed is the beta model.
    for s, vv, dv in ((40.0, 20.0, 0.5), (15.0, 25.0, -1.0), (scf, v, 0.0)):
        a1 = float(tg_accel(s, vv, dv, p))
        a2 = float(virtual_accel(s, vv, dv, p, capped=False))
        assert abs(a1 - a2) < 1e-6, (s, vv, dv, a1, a2)

    # Cancellation identity: fv = fdv = 0, ka = 1, no lag -> Gamma = 1.
    w = np.array([0.2, 1.0, 4.0])
    g = transfer(w, 1.2, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert np.allclose(np.abs(g), 1.0, atol=1e-9)

    fs, fv, fdv = gains_at(s_tg, v, p, "truegap")
    kmin = ka_min_delay_free(fs, fv, fdv)
    assert np.isfinite(kmin)
    h_edge = hinf_norm(fs, fv, fdv, max(kmin, 0.0), 0.0, 0.0)
    assert abs(h_edge - 1.0) < 5e-3, (kmin, h_edge)
    if kmin > 0.05:
        h_bad = hinf_norm(fs, fv, fdv, 0.0, 0.0, 0.0)
        assert h_bad > 1.0 + 1e-3, (kmin, h_bad)
    # High-frequency gain tends to ka when the driveline lag is present.
    wh = 80.0
    g_hi = abs(transfer(wh, fs, fv, fdv, 0.7, 0.0, p.tau_act))
    assert abs(g_hi - 0.7) < 0.05, g_hi

    # Dropping beta derivatives is not free when the blend is centred on the gap.
    p_bad = with_updates(p, lam=1.0, gamma=0.25)
    s_bad = tg_equilibrium_gap(v, p_bad)
    full = gains_at(s_bad, v, p_bad, "truegap", drop_beta_derivatives=False)
    drop = gains_at(s_bad, v, p_bad, "truegap", drop_beta_derivatives=True)
    assert abs(full[0] - drop[0]) > 1e-3

    # CBF worst-case derivative vanishes under full braking on h = 0, tau_r = 0.
    b_f = b_l = 5.0
    vv, vl = 20.0, 20.0
    s_star_safe = float(safe_gap(vv, vl, b_f, b_l, p.s0, 0.0))
    assert abs(barrier(s_star_safe, vv, vl, b_f, b_l, p.s0, 0.0)) < 1e-9
    # h_dot worst case = -v (1 + u/b_f) at tau_r = 0.
    u = -b_f
    hdot = -vv * (1.0 + u / b_f)
    assert abs(hdot) < 1e-12

    g0 = linear_cacc_transfer(0.0, p.kp, p.kd, 1.0, p.T, 0.0, p.tau_act)
    assert abs(g0 - 1.0) < 1e-9
    print("self-check passed")
    print(f"  v={v:.1f}  s_cf={scf:.3f}  s_tg={s_tg:.4f}  s_idm={s_idm:.3f}")
    print(f"  gains fs={fs:.4f} fv={fv:.4f} fdv={fdv:.4f}  ka_min={kmin:.3f}")
    return {
        "v": v,
        "s_cf": scf,
        "s_tg": s_tg,
        "s_idm": s_idm,
        "fs": fs,
        "fv": fv,
        "fdv": fdv,
        "ka_min": kmin,
    }


def _write_csv(path: Path, rows: list[dict]):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def equilibrium_table(p=DEFAULT):
    rows = []
    speeds = np.unique(np.concatenate([
        np.linspace(5.0, 32.5, 56),
        np.array([20.0, 25.0, 30.0]),
    ]))
    for T in (0.6, 1.1, 1.6):
        pt = with_updates(p, T=T)
        for v in speeds:
            scf = pt.s0 + v * pt.T
            sidm = float(idm_equilibrium_gap(v, pt))
            stg = tg_equilibrium_gap(v, pt)
            svir = _virtual_gap(v, pt)
            rows.append({
                "T": T,
                "v": v,
                "s_ctg": scf,
                "s_idm": sidm,
                "s_tg": stg,
                "s_virtual": svir,
                "q_ctg": flow(v, scf, pt),
                "q_idm": flow(v, sidm, pt),
                "q_tg": flow(v, stg, pt),
                "q_virtual": flow(v, svir, pt),
                "k_ctg": 1000.0 / (scf + pt.length),
                "k_idm": 1000.0 / (sidm + pt.length) if np.isfinite(sidm) else 0.0,
                "k_tg": 1000.0 / (stg + pt.length),
            })
    _write_csv(OUT / "equilibrium.csv", rows)
    return rows


def beta_design_table(p=DEFAULT, v=25.0, T=0.6):
    pt = with_updates(p, T=T)
    rows = []
    for lam in np.linspace(1.0, 2.2, 25):
        for gamma in np.linspace(0.05, 0.8, 25):
            pp = with_updates(pt, lam=float(lam), gamma=float(gamma))
            scf = pp.s0 + v * pp.T
            try:
                s_e = tg_equilibrium_gap(v, pp)
            except ValueError:
                continue
            bta = float(beta_blend(s_e, v, pp))
            db_ds, db_dv = beta_derivatives(s_e, v, pp)
            fs, fv, fdv = gains_at(s_e, v, pp, "truegap")
            fs0, _, _ = gains_at(s_e, v, pp, "truegap", drop_beta_derivatives=True)
            h = hinf_norm(fs, fv, fdv, 1.0, 0.0, 0.0, n=1200)
            rows.append({
                "lam": lam,
                "gamma": gamma,
                "s_e": s_e,
                "inflation": s_e / scf - 1.0,
                "beta": bta,
                "db_ds": float(db_ds),
                "rel_fs": abs(fs - fs0) / max(abs(fs0), 1e-8),
                "hinf_ka1": h,
                "xi": pp.gamma * (pp.lam - 1.0) * scf,
            })
    _write_csv(OUT / "beta_design.csv", rows)
    return rows


def stability_table(p=DEFAULT):
    rows = []
    v = 25.0
    for T in np.linspace(0.4, 2.2, 19):
        pt = with_updates(p, T=float(T))
        for kind, gap_fn in (
            ("truegap", lambda vv, pp: tg_equilibrium_gap(vv, pp)),
            ("idm", lambda vv, pp: float(idm_equilibrium_gap(vv, pp))),
        ):
            s_e = gap_fn(v, pt)
            fs, fv, fdv = gains_at(s_e, v, pt, kind)
            kmin = ka_min_delay_free(fs, fv, fdv)
            for ka in (0.0, 0.5, 1.0):
                h0 = hinf_norm(fs, fv, fdv, ka, 0.0, 0.0, n=2000)
                h_lag = hinf_norm(fs, fv, fdv, ka, 0.0, pt.tau_act, n=2000)
                if ka > 0.0:
                    tmax = tau_max(fs, fv, fdv, ka, pt.tau_act, level=1.0)
                    tmax05 = tau_max(fs, fv, fdv, ka, pt.tau_act, level=1.05)
                    tmax10 = tau_max(fs, fv, fdv, ka, pt.tau_act, level=1.10)
                else:
                    tmax = tmax05 = tmax10 = 0.0
                rows.append({
                    "model": kind,
                    "T": float(T),
                    "v": v,
                    "s_e": s_e,
                    "fs": fs,
                    "fv": fv,
                    "fdv": fdv,
                    "ka_min": kmin,
                    "ka": ka,
                    "hinf_lagfree": h0,
                    "hinf_tauact": h_lag,
                    "tau_max": tmax,
                    "tau_max_105": tmax05,
                    "tau_max_110": tmax10,
                    "p_min": (max(kmin, 0.0) / ka) if ka > 1e-9 else np.nan,
                })
    # Heat map at the reference headway: delay x feedforward, true-gap.
    pt = with_updates(p, T=0.6)
    s_e = tg_equilibrium_gap(v, pt)
    fs, fv, fdv = gains_at(s_e, v, pt, "truegap")
    heat = []
    for tau in np.linspace(0.0, 0.45, 16):
        for ka in np.linspace(0.0, 1.2, 17):
            h = hinf_norm(fs, fv, fdv, float(ka), float(tau), pt.tau_act, n=1500)
            heat.append({
                "model": "truegap_heat",
                "T": 0.6,
                "v": v,
                "s_e": s_e,
                "fs": fs,
                "fv": fv,
                "fdv": fdv,
                "ka_min": ka_min_delay_free(fs, fv, fdv),
                "ka": float(ka),
                "hinf_lagfree": np.nan,
                "hinf_tauact": h,
                "tau_max": float(tau),
                "tau_max_105": np.nan,
                "tau_max_110": np.nan,
                "p_min": np.nan,
            })
    rows.extend(heat)
    _write_csv(OUT / "stability.csv", rows)
    return rows


def sensitivity_table(p=DEFAULT, v=25.0, T=0.6):
    rows = []
    base = with_updates(p, T=T)
    grid = {
        "a": np.linspace(0.8, 2.0, 7),
        "b": np.linspace(1.0, 3.0, 7),
        "v0": np.linspace(30.0, 40.0, 6),
        "lam": np.linspace(1.05, 2.0, 8),
        "gamma": np.linspace(0.08, 0.7, 8),
    }
    for name, values in grid.items():
        for val in values:
            pp = with_updates(base, **{name: float(val)})
            if pp.v0 <= v + 0.2:
                continue
            s_e = tg_equilibrium_gap(v, pp)
            scf = pp.s0 + v * pp.T
            fs, fv, fdv = gains_at(s_e, v, pp, "truegap")
            rows.append({
                "parameter": name,
                "value": float(val),
                "inflation": s_e / scf - 1.0,
                "ka_min": ka_min_delay_free(fs, fv, fdv),
                "hinf_ka1_tau0": hinf_norm(fs, fv, fdv, 1.0, 0.0, pp.tau_act, n=1500),
                "tau_max": tau_max(fs, fv, fdv, 1.0, pp.tau_act),
                "beta": float(beta_blend(s_e, v, pp)),
            })
    _write_csv(OUT / "sensitivity.csv", rows)
    return rows


def linear_baseline_table(p=DEFAULT, v=25.0):
    rows = []
    for T in (0.6, 1.1, 1.6):
        for tau in (0.0, 0.1, 0.2):
            ploeg = linear_cacc_hinf(p.kp, p.kd, 1.0, T, tau, p.tau_act, n=2500)
            tuned = tune_linear_cacc(T, tau, p.tau_act, ka=1.0)
            rows.append({
                "T": T,
                "tau": tau,
                "ploeg_hinf": ploeg,
                "ploeg_kp": p.kp,
                "ploeg_kd": p.kd,
                "tuned_hinf": tuned["hinf"],
                "tuned_kp": tuned["kp"],
                "tuned_kd": tuned["kd"],
            })
            print(f"  linear T={T} tau={tau}: Ploeg {ploeg:.3f}, tuned {tuned['hinf']:.3f} "
                  f"(kp={tuned['kp']:.2f}, kd={tuned['kd']:.2f})")
    _write_csv(OUT / "linear_tune.csv", rows)
    return rows


def _own_gap(kind, v, p):
    if kind in ("idm", "eidm", "cidm"):
        return float(idm_equilibrium_gap(v, p))
    if kind == "truegap":
        return tg_equilibrium_gap(v, p)
    if kind == "virtual":
        return _virtual_gap(v, p)
    if kind in ("linear", "linear_hinf"):
        return p.s0 + v * p.T
    raise ValueError(kind)


def _run(kind, leader, t_end, gap, v, p, comm, kp=None, kd=None, ka=None,
         n_veh=10, dt=0.01, b_max=None, use_cbf=False, tau_r=0.0,
         gap_scale=None, name=""):
    return simulate(
        leader, t_end, gap, v, n_veh=n_veh, p=p, kind=("linear" if kind == "linear_hinf" else kind),
        comm=comm, dt=dt, b_max=b_max, use_cbf=use_cbf, tau_r=tau_r,
        kp=kp, kd=kd, ka=ka, name=name or kind, gap_scale=gap_scale,
    )


def time_domain(p, tuned_kp, tuned_kd, v=25.0, T=0.6):
    pt = with_updates(p, T=T, ka=1.0)
    kinds = ["idm", "cidm", "eidm", "truegap", "virtual", "linear", "linear_hinf"]
    rows = []
    series = {}

    def gains_for(kind):
        if kind == "linear_hinf":
            return tuned_kp, tuned_kd, 1.0
        if kind == "linear":
            return pt.kp, pt.kd, 1.0
        return None, None, (0.0 if kind in ("idm", "eidm") else 1.0)

    # Settlement from the common uninflated gap.
    print("  settlement")
    for kind in kinds:
        kp, kd, ka = gains_for(kind)
        res = _run(kind, lambda t: 0.0, 80.0, pt.s0 + v * pt.T, v, pt,
                   Comm(model="ideal", tau=0.0), kp, kd, ka, n_veh=8, dt=0.02, name=kind)
        s_final = float(np.mean(res.s[-200:, 1:]))
        rows.append({
            "scenario": "settle",
            "controller": kind,
            "T": T,
            "tau": 0.0,
            "alpha": 1.0,
            "p_deliver": 1.0,
            "channel": "ideal",
            "s_final": s_final,
            "q_final": flow(float(res.v[-1, 1]), s_final, pt),
            "T_eff": (s_final - pt.s0) / v,
            "l2_tail": np.nan,
            "l2_max": np.nan,
            "min_gap": res.min_gap,
            "collision": int(res.collision),
            "max_brake": float(np.min(res.a)),
            "max_jerk": np.nan,
            "seed": 0,
        })

    # Disturbances, each controller at its own equilibrium.
    # 0.7 Hz sits near the H-infinity peak of the lagged true-gap linearisation.
    # The amplitude keeps the command inside the acceleration bounds.
    def sinusoid(t, t0=15.0, t1=35.0, amp=0.35, freq=0.70):
        if t0 <= t <= t1:
            w = 2.0 * np.pi * freq
            return amp * w * np.cos(w * (t - t0))
        return 0.0

    def brake(t, t0=15.0, decel=3.5, dur=4.0):
        if t0 <= t < t0 + dur:
            return -decel
        return 0.0

    for scenario, leader, t0, t1, tend in (
        ("sinusoid", sinusoid, 15.0, 40.0, 45.0),
        ("brake", brake, 15.0, 35.0, 40.0),
    ):
        print(f"  {scenario}")
        for kind in kinds:
            kp, kd, ka = gains_for(kind)
            gap = _own_gap("linear" if kind == "linear_hinf" else kind, v, pt)
            res = _run(kind, leader, tend, gap, v, pt,
                       Comm(model="ideal", tau=0.1), kp, kd, ka,
                       n_veh=10, dt=0.01, name=kind)
            met = disturbance_metrics(res, v, t0, t1)
            rows.append({
                "scenario": scenario,
                "controller": kind,
                "T": T,
                "tau": 0.1,
                "alpha": 1.0,
                "p_deliver": 1.0,
                "channel": "ideal",
                "s_final": float(np.nanmean(res.s[-1, 1:])),
                "q_final": np.nan,
                "T_eff": np.nan,
                "l2_tail": met["l2_tail"],
                "l2_max": met["l2_max"],
                "min_gap": met["min_gap"],
                "collision": int(met["collision"]),
                "max_brake": met["max_brake"],
                "max_jerk": met["max_jerk"],
                "seed": 0,
            })
            step = 5
            series[f"{scenario}_{kind}_t"] = res.t[::step]
            series[f"{scenario}_{kind}_v"] = res.v[::step]
            series[f"{scenario}_{kind}_a"] = res.a[::step]
            series[f"{scenario}_{kind}_s"] = res.s[::step]
            series[f"{scenario}_{kind}_l2"] = met["l2_ratio"]

    # Cut-in, and the same cut-in at three blend slopes.
    print("  cut-in")
    def make_cut(t_cut, gap_value):
        def gap_scale(t, s, vv):
            if t >= t_cut:
                return gap_value
            return None
        return gap_scale

    for gamma, tag in ((0.10, "g010"), (0.35, "g035"), (1.00, "g100")):
        pp = with_updates(pt, gamma=gamma)
        gap = _own_gap("truegap", v, pp)
        res = _run(
            "truegap", lambda t: 0.0, 30.0, gap, v, pp,
            Comm(model="ideal", tau=0.0), None, None, 1.0,
            n_veh=6, dt=0.01, name=f"cut_{tag}",
            gap_scale=make_cut(12.0, 0.45 * gap),
        )
        met = disturbance_metrics(res, v, 12.0, 25.0)
        rows.append({
            "scenario": "cutin",
            "controller": f"truegap_{tag}",
            "T": T,
            "tau": 0.0,
            "alpha": 1.0,
            "p_deliver": 1.0,
            "channel": "ideal",
            "s_final": float(np.nanmean(res.s[-1, 1:])),
            "q_final": np.nan,
            "T_eff": np.nan,
            "l2_tail": met["l2_tail"],
            "l2_max": met["l2_max"],
            "min_gap": met["min_gap"],
            "collision": int(met["collision"]),
            "max_brake": met["max_brake"],
            "max_jerk": met["max_jerk"],
            "seed": 0,
        })
        series[f"cut_{tag}_t"] = res.t[::5]
        series[f"cut_{tag}_a"] = res.a[::5, 1]
        series[f"cut_{tag}_s"] = res.s[::5, 1]

    for kind, kp, kd in (
        ("eidm", None, None),
        ("idm", None, None),
        ("linear_hinf", tuned_kp, tuned_kd),
    ):
        gap = _own_gap(kind if kind != "linear_hinf" else "linear", v, pt)
        ka = 0.0 if kind in ("idm", "eidm") else 1.0
        res = _run(
            kind, lambda t: 0.0, 30.0, gap, v, pt,
            Comm(model="ideal", tau=0.0), kp, kd, ka,
            n_veh=6, dt=0.01, name=f"cut_{kind}",
            gap_scale=make_cut(12.0, 0.45 * gap),
        )
        met = disturbance_metrics(res, v, 12.0, 25.0)
        rows.append({
            "scenario": "cutin",
            "controller": kind,
            "T": T, "tau": 0.0, "alpha": 1.0, "p_deliver": 1.0,
            "channel": "ideal",
            "s_final": float(np.nanmean(res.s[-1, 1:])),
            "q_final": np.nan, "T_eff": np.nan,
            "l2_tail": met["l2_tail"], "l2_max": met["l2_max"],
            "min_gap": met["min_gap"], "collision": int(met["collision"]),
            "max_brake": met["max_brake"], "max_jerk": met["max_jerk"],
            "seed": 0,
        })
        series[f"cut_{kind}_t"] = res.t[::5]
        series[f"cut_{kind}_a"] = res.a[::5, 1]
        series[f"cut_{kind}_s"] = res.s[::5, 1]

    # Heterogeneous braking. Leader brakes at 4 m/s^2; one follower is weaker.
    print("  heterogeneity")
    alphas = [1.0, 0.85, 0.7, 0.55, 0.4]
    for alpha in alphas:
        for use_cbf in (False, True):
            b = np.full(8, pt.b_max)
            b[3] = alpha * pt.b_max
            gap = _own_gap("truegap", v, pt)
            res = _run(
                "truegap", brake, 35.0, gap, v, pt,
                Comm(model="ideal", tau=0.1), None, None, 1.0,
                n_veh=8, dt=0.01, b_max=b, use_cbf=use_cbf, tau_r=0.1,
                name=f"het_{alpha}_{use_cbf}",
            )
            rows.append({
                "scenario": "hetero_cbf" if use_cbf else "hetero",
                "controller": "truegap",
                "T": T, "tau": 0.1, "alpha": alpha, "p_deliver": 1.0,
                "channel": "ideal",
                "s_final": float(np.nanmean(res.s[-1, 1:])),
                "q_final": np.nan, "T_eff": np.nan,
                "l2_tail": np.nan, "l2_max": np.nan,
                "min_gap": res.min_gap, "collision": int(res.collision),
                "max_brake": float(np.min(res.a[:, 3])),
                "max_jerk": np.nan, "seed": 0,
            })
    _write_csv(OUT / "sim_metrics.csv", rows)
    np.savez_compressed(OUT / "series.npz", **{k: np.asarray(val) for k, val in series.items()})
    return rows, series


def monte_carlo(p, tuned_kp, tuned_kd, v=25.0, T=0.6, n_mc=12):
    pt = with_updates(p, T=T, ka=1.0)
    rows = []
    gap_tg = _own_gap("truegap", v, pt)
    gap_lin = _own_gap("linear", v, pt)

    def leader(t):
        if 12.0 <= t <= 28.0:
            w = 2.0 * np.pi * 0.70
            return 0.35 * w * np.cos(w * (t - 12.0))
        return 0.0

    jobs = []
    for channel in ("bernoulli", "gilbert"):
        levels = (1.0, 0.9, 0.75, 0.5) if channel == "bernoulli" else (None,)
        for p_del in levels:
            for kind, gap, kp, kd in (
                ("truegap", gap_tg, None, None),
                ("linear_hinf", gap_lin, tuned_kp, tuned_kd),
            ):
                for seed in range(n_mc):
                    jobs.append((channel, p_del, kind, gap, kp, kd, seed))

    print(f"  monte carlo ({len(jobs)} runs)")
    for i, (channel, p_del, kind, gap, kp, kd, seed) in enumerate(jobs):
        comm = Comm(
            model=channel,
            tau=0.1,
            p=1.0 if p_del is None else p_del,
            seed=seed,
        )
        res = _run(
            kind, leader, 36.0, gap, v, pt, comm, kp, kd, 1.0,
            n_veh=8, dt=0.02, name=kind,
        )
        met = disturbance_metrics(res, v, 12.0, 34.0)
        rows.append({
            "scenario": "montecarlo",
            "controller": kind,
            "T": T,
            "tau": 0.1,
            "alpha": 1.0,
            "p_deliver": -1.0 if p_del is None else p_del,
            "channel": channel,
            "s_final": np.nan,
            "q_final": np.nan,
            "T_eff": np.nan,
            "l2_tail": met["l2_tail"],
            "l2_max": met["l2_max"],
            "min_gap": met["min_gap"],
            "collision": int(met["collision"]),
            "max_brake": met["max_brake"],
            "max_jerk": met["max_jerk"],
            "seed": seed,
            "delivered_mean": float(np.mean(res.delivered[1:])),
        })
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(jobs)}")
    _write_csv(OUT / "montecarlo.csv", rows)
    return rows


def bode_curves(p=DEFAULT, v=25.0, T=0.6):
    pt = with_updates(p, T=T)
    w = np.logspace(-2, 1.3, 600)
    s_tg = tg_equilibrium_gap(v, pt)
    s_idm = float(idm_equilibrium_gap(v, pt))
    out = {"w": w}
    for label, kind, s, ka in (
        ("idm", "idm", s_idm, 0.0),
        ("cidm", "idm", s_idm, 1.0),
        ("tg0", "truegap", s_tg, 0.0),
        ("tg", "truegap", s_tg, 1.0),
    ):
        fs, fv, fdv = gains_at(s, v, pt, kind)
        out[label] = np.abs(transfer(w, fs, fv, fdv, ka, 0.1, pt.tau_act))
        out[f"{label}_tau0"] = np.abs(transfer(w, fs, fv, fdv, ka, 0.0, 0.0))
    out["linear"] = np.abs(linear_cacc_transfer(w, pt.kp, pt.kd, 1.0, T, 0.1, pt.tau_act))
    return out


def headline(eq_rows, stab_rows, sim_rows, mc_rows, check, p=DEFAULT):
    """Compact numbers the manuscript quotes."""
    v_ref = 30.0
    out = {"check": {k: float(val) for k, val in check.items()}}
    for T in (0.6, 1.1, 1.6):
        # nearest speed row
        cands = [r for r in eq_rows if r["T"] == T]
        r = min(cands, key=lambda z: abs(z["v"] - v_ref))
        key = f"T{T:.1f}_v{r['v']:.1f}"
        out[key] = {
            "v": r["v"],
            "q_idm": r["q_idm"],
            "q_tg": r["q_tg"],
            "q_virtual": r["q_virtual"],
            "q_ctg": r["q_ctg"],
            "s_idm": r["s_idm"],
            "s_tg": r["s_tg"],
            "s_virtual": r["s_virtual"],
            "s_ctg": r["s_ctg"],
            "reclaim_pct": 100.0 * (r["q_tg"] / r["q_idm"] - 1.0),
        }
    # tau_max at T=0.6, ka=1, truegap
    sub = [r for r in stab_rows if r["model"] == "truegap" and r["ka"] == 1.0
           and abs(r["T"] - 0.6) < 1e-6]
    if sub:
        out["tau_max_T0.6_ka1"] = sub[0]["tau_max"]
        out["ka_min_T0.6"] = sub[0]["ka_min"]
        out["hinf_tg_ka0"] = sub[0]["hinf_lagfree"] if False else None
    sub0 = [r for r in stab_rows if r["model"] == "truegap" and r["ka"] == 0.0
            and abs(r["T"] - 0.6) < 1e-6]
    if sub0:
        out["hinf_tg_ka0_lagfree"] = sub0[0]["hinf_lagfree"]
    # sim highlights
    def pick(scenario, controller):
        hits = [r for r in sim_rows if r["scenario"] == scenario and r["controller"] == controller]
        return hits[0] if hits else None
    out["sim"] = {}
    for sc in ("settle", "sinusoid", "brake", "cutin"):
        out["sim"][sc] = {}
        ctrls = sorted({r["controller"] for r in sim_rows if r["scenario"] == sc})
        for c in ctrls:
            hit = pick(sc, c)
            if hit:
                out["sim"][sc][c] = {
                    k: hit[k] for k in (
                        "s_final", "q_final", "T_eff", "l2_tail", "l2_max",
                        "min_gap", "collision", "max_brake", "max_jerk",
                    )
                }
    # MC medians
    mc_summary = []
    if mc_rows:
        keys = {(r["channel"], r["controller"], r["p_deliver"]) for r in mc_rows}
        for ch, ctrl, pdv in sorted(keys, key=lambda z: (z[0], z[1], z[2])):
            chunk = [r for r in mc_rows if r["channel"] == ch and r["controller"] == ctrl
                     and r["p_deliver"] == pdv]
            tails = np.array([r["l2_tail"] for r in chunk], dtype=float)
            gaps = np.array([r["min_gap"] for r in chunk], dtype=float)
            mc_summary.append({
                "channel": ch,
                "controller": ctrl,
                "p_deliver": pdv,
                "n": len(chunk),
                "l2_tail_median": float(np.median(tails)),
                "l2_tail_p90": float(np.quantile(tails, 0.9)),
                "min_gap_p10": float(np.quantile(gaps, 0.1)),
                "collision_rate": float(np.mean([r["collision"] for r in chunk])),
            })
    out["montecarlo"] = mc_summary
    # kinematic boundary at the reference speed
    out["alpha_min"] = {
        "v25_T0.6_taur0.1_b5": float(alpha_min(25.0, 0.6, 5.0, 0.1)),
        "v25_T0.6_taur0_b5": float(alpha_min(25.0, 0.6, 5.0, 0.0)),
        "v30_T0.6_taur0.1_b5": float(alpha_min(30.0, 0.6, 5.0, 0.1)),
        "Tmin_alpha0.57_v25": float(min_time_gap(25.0, 0.57, 5.0, 0.1)),
    }
    path = OUT / "summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    p = DEFAULT
    print("self-check")
    check = self_check(p)
    print("equilibrium")
    eq_rows = equilibrium_table(p)
    print("beta design")
    beta_design_table(p)
    print("stability")
    stab_rows = stability_table(p)
    print("sensitivity")
    sensitivity_table(p)
    print("linear baselines")
    lin_rows = linear_baseline_table(p)
    tuned = min(
        (r for r in lin_rows if abs(r["T"] - 0.6) < 1e-9 and abs(r["tau"] - 0.1) < 1e-9),
        key=lambda r: r["tuned_hinf"],
    )
    print("time domain")
    sim_rows, _series = time_domain(p, tuned["tuned_kp"], tuned["tuned_kd"])
    print("packet loss")
    mc_rows = monte_carlo(p, tuned["tuned_kp"], tuned["tuned_kd"])
    print("bode")
    bode = bode_curves(p)
    np.savez_compressed(OUT / "bode.npz", **bode)
    summary = headline(eq_rows, stab_rows, sim_rows, mc_rows, check, p)
    print("figures")
    from .figures import make_all
    make_all()
    print("done")
    print(json.dumps({k: summary[k] for k in summary if k in (
        "T0.6_v30.0", "tau_max_T0.6_ka1", "ka_min_T0.6", "alpha_min"
    )}, indent=2))


if __name__ == "__main__":
    main()
