"""IDM baseline fit on one public OpenACC recording, with held-out runs.

The file is the Ispra-Casale platoon part7.csv from the OpenACC release
(JRC, DOI 10.2905/JRC.KMH3D00; Makridis, Mattas, Anesiadou and Ciuffo, 2021).
The copy used here is the one redistributed with the public notebook
https://github.com/theocharisap/acc_identification_pinns
because the JRC box host did not resolve. The header identifies the
recording: 27 October 2020, leader Rexton, follower Hyundai, ACC on,
minimum distance setting.

The loss is the mean squared bumper-gap error of a forward simulation,
plus a speed penalty. Parameters are chosen on the even-numbered
car-following runs only. Odd-numbered runs are scored afterwards and
never enter the loss. The true-gap law is not fitted: this recording
is a commercial ACC, not that controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .models import tg_accel
from .params import DEFAULT, with_updates

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "openacc" / "part7.csv"
OUT = ROOT / "results" / "truegap"
FIG = ROOT / "figures" / "truegap"

# Even runs train, odd runs test, in chronological order. Fixed before fitting.
TRAIN_MOD = 0
MIN_RUN_S = 20.0
GAP_MAX = 80.0
DELTA = 4.0
# Speed errors are in m/s. The weight keeps a 1 m/s error comparable to a 1 m gap error.
SPEED_WEIGHT = 1.0


def load_runs(path: Path = DATA):
    raw = path.read_text(encoding="utf-8").splitlines()
    header = {
        "date": raw[0].split(",")[1:4],
        "vehicles": [v for v in raw[1].split(",")[1:] if v],
        "n_vehicles": int(raw[2].split(",")[1]),
        "acc": raw[3].split(",")[1],
        "distance_setting": raw[4].split(",")[1],
    }
    df = pd.read_csv(path, skiprows=5)
    t = df["Time"].to_numpy(float)
    vl = df["Speed1"].to_numpy(float)
    v = df["Speed2"].to_numpy(float)
    s = df["IVS1"].to_numpy(float)
    driver = df["Driver2"].astype(str).to_numpy()
    if not np.all(driver == "ACC"):
        raise RuntimeError("expected an ACC-only follower column")
    mask = (s > 5.0) & (s < GAP_MAX) & (v > 5.0) & (vl > 5.0)
    runs = []
    n = len(t)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1] and (t[j + 1] - t[j]) < 0.25:
            j += 1
        if t[j] - t[i] >= MIN_RUN_S:
            sl = slice(i, j + 1)
            runs.append({
                "t": t[sl].copy(),
                "v_lead": vl[sl].copy(),
                "v": v[sl].copy(),
                "s": s[sl].copy(),
            })
        i = j + 1
    return header, runs


def _params(a, b, T, s0, v0):
    return with_updates(
        DEFAULT, a=float(a), b=float(b), T=float(T), s0=float(s0),
        v0=float(v0), delta=DELTA,
    )


def _rollout(run, a, b, T, s0, v0, kind="idm"):
    """Integrate a car-following law with the measured leader speed as input.

    No acceleration command is available in the recording, so the true-gap
    law is run with zero feedforward. That is a counterfactual, not a fit.
    """
    t = run["t"]
    vl = run["v_lead"]
    n = len(t)
    s = np.empty(n)
    v = np.empty(n)
    s[0] = run["s"][0]
    v[0] = run["v"][0]
    p = _params(a, b, T, s0, v0)
    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        dv = v[k] - vl[k]
        gap = s[k] if s[k] > 0.3 else 0.3
        if kind == "idm":
            u = float(idm_from_params(gap, v[k], dv, p))
        elif kind == "truegap":
            u = float(tg_accel(gap, v[k], dv, p))
        else:
            raise ValueError(kind)
        v[k + 1] = v[k] + dt * u
        if v[k + 1] < 0.0:
            v[k + 1] = 0.0
        s[k + 1] = s[k] + dt * (vl[k] - v[k])
    return s, v


def idm_from_params(gap, v, dv, p):
    sqrt_ab = np.sqrt(p.a * p.b)
    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * sqrt_ab))
    free = 0.0 if v <= 0.0 else (v / p.v0) ** p.delta
    return p.a * (1.0 - free - (s_star / gap) ** 2)


def _gap_sse(runs, a, b, T, s0, v0, kind="idm"):
    err_s = 0.0
    err_v = 0.0
    n = 0
    for run in runs:
        s, v = _rollout(run, a, b, T, s0, v0, kind=kind)
        err_s += float(np.sum((s - run["s"]) ** 2))
        err_v += float(np.sum((v - run["v"]) ** 2))
        n += len(run["s"])
    return err_s, err_v, n


def fit(train, v0, b, s0):
    """Fit acceleration and time gap. Jam gap, comfortable deceleration and desired speed are fixed.

    Freeing the jam gap and the comfortable deceleration drives both to the
    edge of any wide box, with the time gap falling so that the equilibrium
    gap stays put. They are not separately identified on this recording.
    """
    def objective(x):
        err_s, err_v, n = _gap_sse(train, x[0], b, x[1], s0, v0)
        return (err_s + SPEED_WEIGHT * err_v) / n

    bounds = [(0.2, 3.0), (0.3, 2.5)]
    result = differential_evolution(
        objective,
        bounds,
        seed=0,
        popsize=10,
        mutation=0.5,
        recombination=0.7,
        atol=1e-5,
        workers=1,
        updating="immediate",
        polish=True,
    )
    return result


def rmse(runs, a, b, T, s0, v0, kind="idm"):
    err_s, err_v, n = _gap_sse(runs, a, b, T, s0, v0, kind=kind)
    return float(np.sqrt(err_s / n)), float(np.sqrt(err_v / n)), n


def algebraic_ctg_rmse(runs, T, s0):
    err = 0.0
    n = 0
    for run in runs:
        pred = s0 + T * run["v"]
        err += float(np.sum((pred - run["s"]) ** 2))
        n += len(run["s"])
    return float(np.sqrt(err / n))


def mean_gap_rmse(runs, mean_gap):
    err = 0.0
    n = 0
    for run in runs:
        err += float(np.sum((mean_gap - run["s"]) ** 2))
        n += len(run["s"])
    return float(np.sqrt(err / n))


def equilibrium_gap(v, a, b, T, s0, v0):
    """Classical IDM equilibrium. Infinite when v >= v0."""
    phi = (v / v0) ** DELTA
    if phi >= 1.0:
        return float("inf")
    return (s0 + v * T) / np.sqrt(1.0 - phi)


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def _figure(test, a, b, T, s0, v0):
    _style()
    run = max(test, key=lambda r: r["t"][-1] - r["t"][0])
    s_idm, _ = _rollout(run, a, b, T, s0, v0, kind="idm")
    s_tg, _ = _rollout(run, a, b, T, s0, v0, kind="truegap")
    t0 = run["t"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.45))
    ax = axes[0]
    ax.plot(run["t"] - t0, run["s"], color="#000000", ls="--", lw=1.15, label="measured ACC")
    ax.plot(run["t"] - t0, s_idm, color="#0072B2", ls="-", lw=1.55, label="IDM, fitted")
    ax.plot(run["t"] - t0, s_tg, color="#D55E00", ls="-.", lw=1.45, label="true-gap, not fitted")
    ax.set_xlabel("time within the held-out run (s)")
    ax.set_ylabel("bumper gap (m)")
    ax.set_title("longest held-out run")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.2)

    ax = axes[1]
    vv = np.linspace(25.0, 42.0, 80)
    s_eq = np.array([equilibrium_gap(v, a, b, T, s0, v0) for v in vv])
    s_ctg = s0 + vv * T
    rng = np.random.default_rng(0)
    for item in test:
        take = rng.choice(len(item["v"]), size=min(350, len(item["v"])), replace=False)
        ax.scatter(
            item["v"][take] * 3.6, item["s"][take],
            s=6, c="#B0B0B0", alpha=0.4, linewidths=0, zorder=1,
        )
    ax.plot(vv * 3.6, s_ctg, color="#000000", ls=(0, (1, 1)), lw=1.25, label="constant time gap", zorder=2)
    ax.plot(vv * 3.6, s_eq, color="#0072B2", ls="-", lw=1.6, label="IDM equilibrium", zorder=3)
    ax.set_xlabel("follower speed (km/h)")
    ax.set_ylabel("bumper gap (m)")
    ax.set_ylim(0, 90)
    ax.set_title("held-out runs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=7.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.26, wspace=0.32)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig9_openacc.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig9_openacc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return float(np.median(np.concatenate([r["v"] for r in test])))


def main():
    header, runs = load_runs()
    train = [r for i, r in enumerate(runs) if i % 2 == TRAIN_MOD]
    test = [r for i, r in enumerate(runs) if i % 2 != TRAIN_MOD]
    print(f"{len(runs)} car-following runs, {len(train)} train, {len(test)} test")
    # Desired speed is above every sample in the file. Jam gap and comfortable
    # deceleration are the values used in the rest of the paper; freeing them
    # does not pin them down on this one recording.
    v0 = 50.0
    b = 2.0
    s0 = 2.0
    result = fit(train, v0, b, s0)
    a, T = [float(x) for x in result.x]
    train_s, train_v, n_train = rmse(train, a, b, T, s0, v0, kind="idm")
    test_s, test_v, n_test = rmse(test, a, b, T, s0, v0, kind="idm")
    tg_s, tg_v, _ = rmse(test, a, b, T, s0, v0, kind="truegap")
    ctg_test = algebraic_ctg_rmse(test, T, s0)
    mean_train_gap = float(np.mean(np.concatenate([r["s"] for r in train])))
    null_test = mean_gap_rmse(test, mean_train_gap)
    v_med = _figure(test, a, b, T, s0, v0)
    s_test = np.concatenate([r["s"] for r in test])
    summary = {
        "source": "OpenACC Ispra-Casale part7.csv",
        "doi": "10.2905/JRC.KMH3D00",
        "file_copy": "https://github.com/theocharisap/acc_identification_pinns/blob/main/part7.csv",
        "feedforward_on_recording": "none; true-gap counterfactual uses ka = 0",
        "date": "-".join(header["date"]),
        "leader": header["vehicles"][0],
        "follower": header["vehicles"][1],
        "distance_setting": header["distance_setting"],
        "driver": "ACC",
        "gap_filter_m": GAP_MAX,
        "split": "even-numbered runs train, odd-numbered runs test, chronological",
        "n_runs_train": len(train),
        "n_runs_test": len(test),
        "n_samples_train": n_train,
        "n_samples_test": n_test,
        "loss": "mean((s_sim-s)^2 + (v_sim-v)^2), delta fixed at 4",
        "a": a,
        "b": b,
        "T": T,
        "s0": s0,
        "v0_fixed_mps": v0,
        "b_fixed": b,
        "s0_fixed": s0,
        "free_parameters": ["a", "T"],
        "delta": DELTA,
        "train_gap_rmse_m": train_s,
        "train_speed_rmse_mps": train_v,
        "test_gap_rmse_m": test_s,
        "test_speed_rmse_mps": test_v,
        "test_truegap_gap_rmse_m": tg_s,
        "test_truegap_speed_rmse_mps": tg_v,
        "test_algebraic_ctg_rmse_m": ctg_test,
        "test_constant_gap_rmse_m": null_test,
        "test_median_speed_mps": v_med,
        "test_median_gap_m": float(np.median(s_test)),
        "s_ctg_at_median_mps": float(s0 + v_med * T),
        "s_idm_at_median_mps": float(equilibrium_gap(v_med, a, b, T, s0, v0)),
        "optimizer_success": bool(result.success),
        "loss_value": float(result.fun),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "openacc_calibration.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
