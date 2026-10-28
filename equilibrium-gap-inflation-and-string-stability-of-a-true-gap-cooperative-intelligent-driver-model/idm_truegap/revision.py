"""Numbers and one extra figure asked for after the numerical study.

Nothing here refits a controller or reruns the Monte Carlo. The delay
map is read from stability.csv. Comfort on the wave is read from
series.npz. Two short simulations fill gaps that those files do not
store: the barrier value on the weak-braking vehicle, and the tail
ratio at a second string length.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiments import _own_gap, _run
from .linearize import flow, tg_equilibrium_gap
from .params import DEFAULT, with_updates
from .safety import barrier
from .simulate import Comm, disturbance_metrics, rms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "truegap"
FIG = ROOT / "figures" / "truegap"


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.8",
    })


def delay_map():
    df = pd.read_csv(OUT / "stability.csv")
    tg = df[(df["model"] == "truegap") & (np.isclose(df["ka"], 1.0))].sort_values("T")
    idm = df[(df["model"] == "idm") & (np.isclose(df["ka"], 0.0))].sort_values("T")
    rows = []
    for _, r in tg.iterrows():
        rows.append({
            "T": float(r["T"]),
            "ka_min": float(r["ka_min"]),
            "hinf_tau_0": float(r["hinf_tauact"]),
            "tau_to_105_s": float(r["tau_max_105"]),
            "tau_to_110_s": float(r["tau_max_110"]),
        })
    idm_rows = [
        {"T": float(r["T"]), "ka_min": float(r["ka_min"])}
        for _, r in idm.iterrows()
    ]
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 3.25))
    ax.plot(tg["T"], tg["ka_min"], color="#0072B2", ls="-", marker="o", ms=4.0, label=r"$k_{a,\min}$")
    ax.set_xlabel(r"time gap $T$ (s)")
    ax.set_ylabel(r"$k_{a,\min}$")
    ax.set_ylim(0.0, 0.40)
    ax2 = ax.twinx()
    ax2.plot(
        tg["T"], 1000.0 * tg["tau_max_105"],
        color="#D55E00", ls="--", marker="s", ms=4.5, label=r"delay to $1.05$",
    )
    ax2.plot(
        tg["T"], 1000.0 * tg["tau_max_110"],
        color="#009E73", ls=":", marker="^", ms=4.5, label=r"delay to $1.10$",
    )
    ax2.set_ylabel("communication delay (ms)")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_delay_margin.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig10_delay_margin.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows, idm_rows


def wave_comfort():
    series = dict(np.load(OUT / "series.npz"))
    out = {}
    for kind in ("idm", "truegap"):
        t = series[f"sinusoid_{kind}_t"]
        v = series[f"sinusoid_{kind}_v"]
        s = series[f"sinusoid_{kind}_s"]
        a = series[f"sinusoid_{kind}_a"]
        m = (t >= 15.0) & (t <= 35.0)
        dt = float(t[1] - t[0])
        # Gap rate of every follower, from the saved samples.
        ds = np.diff(s, axis=0) / dt
        m_ds = (t[:-1] >= 15.0) & (t[:-1] <= 35.0)
        out[kind] = {
            "rms_accel_followers": float(np.sqrt(np.mean(a[m, 1:] ** 2))),
            "rms_gap_rate_followers": float(np.sqrt(np.mean(ds[m_ds, 1:] ** 2))),
            "rms_accel_tail": rms(a[:, -1], m),
            "peak_abs_accel_followers": float(np.max(np.abs(a[m, 1:]))),
        }
    return out


def openacc_flow():
    cal = json.loads((OUT / "openacc_calibration.json").read_text(encoding="utf-8"))
    v = float(cal["test_median_speed_mps"])
    T = float(cal["T"])
    a = float(cal["a"])
    p = with_updates(DEFAULT, a=a, b=2.0, T=T, s0=2.0, v0=50.0, delta=4.0)
    s_ctg = float(cal["s_ctg_at_median_mps"])
    s_idm = float(cal["s_idm_at_median_mps"])
    s_meas = float(cal["test_median_gap_m"])
    s_tg = float(tg_equilibrium_gap(v, p))
    return {
        "v_mps": v,
        "T": T,
        "a": a,
        "s_ctg": s_ctg,
        "s_idm": s_idm,
        "s_truegap": s_tg,
        "s_measured_median": s_meas,
        "q_ctg": float(flow(v, s_ctg, p)),
        "q_idm": float(flow(v, s_idm, p)),
        "q_truegap": float(flow(v, s_tg, p)),
        "q_measured": float(flow(v, s_meas, p)),
    }


def barrier_on_weak_brake():
    """Minimum barrier on vehicle 3, the follower limited to alpha = 0.4."""
    v = 25.0
    T = 0.6
    pt = with_updates(DEFAULT, T=T, ka=1.0)
    gap = _own_gap("truegap", v, pt)

    def leader(t, t0=15.0, decel=3.5, dur=4.0):
        if t0 <= t < t0 + dur:
            return -decel
        return 0.0

    out = {}
    for use_cbf in (False, True):
        b = np.full(8, pt.b_max)
        b[3] = 0.4 * pt.b_max
        res = _run(
            "truegap", leader, 35.0, gap, v, pt,
            Comm(model="ideal", tau=0.1), None, None, 1.0,
            n_veh=8, dt=0.01, b_max=b, use_cbf=use_cbf, tau_r=0.1,
        )
        h = barrier(
            res.s[:, 3], res.v[:, 3], res.v[:, 2],
            b_f=float(b[3]), b_l=float(b[2]), s0=pt.s0, tau_r=0.1,
        )
        out["cbf" if use_cbf else "nominal"] = {
            "min_h_vehicle3": float(np.min(h)),
            "min_gap_vehicle3": float(np.min(res.s[:, 3])),
            "min_gap_platoon": float(res.min_gap),
        }
    return out


def string_lengths():
    """Tail L2 ratio at two lengths. Same wave as the published N=10 run."""
    v = 25.0
    T = 0.6
    pt = with_updates(DEFAULT, T=T, ka=1.0)

    def sinusoid(t, t0=15.0, t1=35.0, amp=0.35, freq=0.70):
        if t0 <= t <= t1:
            w = 2.0 * np.pi * freq
            return amp * w * np.cos(w * (t - t0))
        return 0.0

    rows = {}
    for n_veh in (6, 10):
        for kind in ("idm", "truegap"):
            gap = _own_gap(kind, v, pt)
            ka = 0.0 if kind == "idm" else 1.0
            res = _run(
                kind, sinusoid, 45.0, gap, v, pt,
                Comm(model="ideal", tau=0.1), None, None, ka,
                n_veh=n_veh, dt=0.01, name=kind,
            )
            met = disturbance_metrics(res, v, 15.0, 40.0)
            rows[f"{kind}_N{n_veh}"] = {
                "l2_tail": met["l2_tail"],
                "l2_max": met["l2_max"],
            }
    return rows


def main():
    delay, idm_ka = delay_map()
    comfort = wave_comfort()
    flows = openacc_flow()
    print("barrier")
    hstat = barrier_on_weak_brake()
    print("string length")
    lengths = string_lengths()
    summary = {
        "delay_truegap_ka1": delay,
        "idm_ka_min": idm_ka,
        "wave_comfort": comfort,
        "openacc_flow": flows,
        "barrier_alpha_0p4": hstat,
        "string_length": lengths,
    }
    (OUT / "revision_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
