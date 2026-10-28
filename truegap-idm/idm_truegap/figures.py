"""Publication figures. Reads results/truegap/ written by experiments.py."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .models import beta_blend, beta_derivatives, s_cf
from .params import DEFAULT, with_updates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "truegap"
FIG = ROOT / "figures" / "truegap"

# Colour is never the only cue: each series also has its own line style.
# The palette is Okabe–Ito. Yellow-on-white is avoided.
STYLE = {
    "ctg": dict(color="#000000", ls=(0, (1, 1)), lw=1.15),
    "idm": dict(color="#0072B2", ls="-", lw=1.7),
    "cidm": dict(color="#56B4E9", ls="--", lw=1.45),
    "eidm": dict(color="#E69F00", ls=":", lw=1.9),
    "truegap": dict(color="#D55E00", ls="-", lw=1.7),
    "virtual": dict(color="#009E73", ls="-.", lw=1.55),
    "linear": dict(color="#CC79A7", ls="--", lw=1.45),
    "linear_hinf": dict(color="#882255", ls="-.", lw=1.55),
    "leader": dict(color="#666666", ls=":", lw=1.05),
    "gamma_lo": dict(color="#000000", ls="--", lw=1.35),
    "gamma_mid": dict(color="#D55E00", ls="-", lw=1.7),
    "gamma_hi": dict(color="#0072B2", ls="-.", lw=1.45),
}


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.5,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
    })


def _legend_below(ax, ncol=2, fontsize=7.2):
    """Legend under the axes, so it cannot cover the data."""
    return ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=ncol,
        fontsize=fontsize,
        borderaxespad=0.0,
    )


def _read(name):
    path = OUT / name
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if v is None or v == "":
                d[k] = np.nan
                continue
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _sub(rows, **kw):
    out = []
    for r in rows:
        if all(abs(r[k] - v) < 1e-8 if isinstance(v, float) else r[k] == v for k, v in kw.items()):
            out.append(r)
    return out


def fig_equilibrium(eq):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.55))
    ref = sorted([r for r in eq if abs(r["T"] - 0.6) < 1e-9], key=lambda r: r["v"])
    v = np.array([r["v"] for r in ref]) * 3.6
    series = (
        ("truegap", [r["s_tg"] for r in ref], "true-gap"),
        ("idm", [r["s_idm"] for r in ref], "IDM / eIDM"),
        ("virtual", [r["s_virtual"] for r in ref], "virtual speed, capped"),
        ("ctg", [r["s_ctg"] for r in ref], "constant time gap"),
    )
    ax = axes[0]
    for key, y, lab in series:
        ax.plot(v, y, label=lab, **STYLE[key])
    ax.set_xlabel("speed (km/h)")
    ax.set_ylabel("equilibrium bumper gap (m)")
    ax.set_ylim(0, 80)
    ax.set_title(r"gap, $T = 0.6$ s")

    ax = axes[1]
    flow_series = (
        ("truegap", [r["q_tg"] for r in ref], "true-gap"),
        ("idm", [r["q_idm"] for r in ref], "IDM / eIDM"),
        ("virtual", [r["q_virtual"] for r in ref], "virtual speed, capped"),
        ("ctg", [r["q_ctg"] for r in ref], "constant time gap"),
    )
    for key, y, lab in flow_series:
        ax.plot(v, y, label=lab, **STYLE[key])
    ax.set_xlabel("speed (km/h)")
    ax.set_ylabel("pipeline flow (veh/h)")
    ax.set_title(r"flow, $T = 0.6$ s")

    ax = axes[2]
    # Colour identifies the law; line style identifies the time gap.
    for T, ls in ((0.6, "-"), (1.1, "--"), (1.6, ":")):
        rows = sorted([r for r in eq if abs(r["T"] - T) < 1e-9], key=lambda r: r["v"])
        ax.plot(
            [r["k_idm"] for r in rows], [r["q_idm"] for r in rows],
            color=STYLE["idm"]["color"], ls=ls, lw=1.55,
        )
        ax.plot(
            [r["k_tg"] for r in rows], [r["q_tg"] for r in rows],
            color=STYLE["truegap"]["color"], ls=ls, lw=1.55,
        )
    ax.set_xlabel("density (veh/km)")
    ax.set_ylabel("flow (veh/h)")
    ax.set_title("fundamental diagram")
    fd_handles = [
        Line2D([0], [0], color=STYLE["idm"]["color"], lw=1.6, label="IDM"),
        Line2D([0], [0], color=STYLE["truegap"]["color"], lw=1.6, label="true-gap"),
        Line2D([0], [0], color="0.15", ls="-", lw=1.4, label=r"$T = 0.6$ s"),
        Line2D([0], [0], color="0.15", ls="--", lw=1.4, label=r"$T = 1.1$ s"),
        Line2D([0], [0], color="0.15", ls=":", lw=1.6, label=r"$T = 1.6$ s"),
    ]
    shared, shared_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        shared, shared_labels, loc="upper center", ncol=2,
        bbox_to_anchor=(0.34, 0.02), fontsize=7.0,
    )
    fig.legend(
        fd_handles, [h.get_label() for h in fd_handles],
        loc="upper center", ncol=2, bbox_to_anchor=(0.84, 0.02), fontsize=7.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.42)
    _save(fig, "fig1_equilibrium")


def fig_beta(beta_rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.15), constrained_layout=True)
    p = with_updates(DEFAULT, T=0.6)
    v = 25.0
    scf = float(s_cf(v, p))
    s = np.linspace(0.4 * scf, 3.2 * scf, 400)
    b = beta_blend(s, v, p)
    db, _ = beta_derivatives(s, v, p)
    ax = axes[0]
    ax.plot(s, b, label=r"blend $\beta$", **STYLE["truegap"])
    ax.plot(
        s, db / max(float(np.max(db)), 1e-9),
        label=r"$\partial\beta/\partial s$, scaled", **STYLE["idm"],
    )
    ax.axvline(scf, color="0.35", lw=0.7, ls="--")
    ax.axvline(p.lam * scf, color="0.35", lw=0.7, ls=":")
    # Left of the spike the blend is flat, so these labels sit in empty space.
    ax.text(scf - 1.5, 0.50, "policy gap", ha="right", va="center", fontsize=6.5, color="0.2")
    ax.text(min(s[-1] - 1.0, p.lam * scf + 8.0), 0.38, "transition", ha="left", va="center", fontsize=6.5, color="0.2")
    ax.set_xlabel("bumper gap (m)")
    ax.set_ylabel("blend value")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(r"default blend, 25 m/s")
    ax.legend(loc="upper left", fontsize=6.8, frameon=True, framealpha=1.0, edgecolor="0.85")

    lams = np.array(sorted({round(r["lam"], 6) for r in beta_rows}))
    gams = np.array(sorted({round(r["gamma"], 6) for r in beta_rows}))
    infl = np.full((len(gams), len(lams)), np.nan)
    rel = np.full_like(infl, np.nan)
    lam_i = {round(x, 6): i for i, x in enumerate(lams)}
    gam_i = {round(x, 6): i for i, x in enumerate(gams)}
    for r in beta_rows:
        i = gam_i[round(r["gamma"], 6)]
        j = lam_i[round(r["lam"], 6)]
        infl[i, j] = 100.0 * r["inflation"]
        rel[i, j] = 100.0 * r["rel_fs"]
    extent = [lams[0], lams[-1], gams[0], gams[-1]]
    for ax, Z, title, label in (
        (axes[1], infl, "residual gap inflation", "inflation (%)"),
        (axes[2], rel, r"effect of dropping $\partial\beta$", r"% of $f_s$"),
    ):
        im = ax.imshow(
            Z, origin="lower", aspect="auto", extent=extent,
            cmap="cividis", vmin=0, vmax=np.nanpercentile(Z, 98),
        )
        ax.set_xlabel(r"transition location $\lambda$")
        ax.set_ylabel(r"slope $\gamma$ (1/m)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    _save(fig, "fig2_beta_design")


def fig_stability(stab, bode):
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.35))
    ax = axes[0]
    w = bode["w"]
    curves = (
        ("idm", bode["idm"], "IDM"),
        ("cidm", bode["cidm"], "IDM + feedforward"),
        ("eidm", bode["tg0"], r"true-gap, $k_a=0$"),
        ("truegap", bode["tg"], r"true-gap, $k_a=1$"),
        ("linear", bode["linear"], "linear CACC"),
    )
    for key, mag, lab in curves:
        ax.semilogx(w, 20 * np.log10(np.maximum(mag, 1e-6)), label=lab, **STYLE[key])
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set_xlabel(r"frequency $\omega$ (rad/s)")
    ax.set_ylabel(r"$|\Gamma(j\omega)|$ (dB)")
    ax.set_title(r"$\tau = 0.1$ s, with lag")
    ax.set_ylim(-28, 6)
    # The responses sit on 0 dB, so the legend goes in the empty lower-left.
    ax.legend(
        loc="lower left", fontsize=6.2, frameon=True, framealpha=1.0, edgecolor="0.85",
    )

    ax = axes[1]
    for model, key, label in (
        ("truegap", "truegap", "true-gap"),
        ("idm", "idm", "IDM"),
    ):
        rows = sorted(
            [r for r in stab if r["model"] == model and abs(r["ka"]) < 1e-9],
            key=lambda r: r["T"],
        )
        ax.plot([r["T"] for r in rows], [r["ka_min"] for r in rows], label=label, **STYLE[key])
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.axhline(1.0, color="0.45", lw=0.6, ls=(0, (1, 1)))
    ax.set_xlabel("time gap $T$ (s)")
    ax.set_ylabel(r"delay-free threshold $k_{a,\min}$")
    ax.set_title(r"$v = 25$ m/s, no lag")
    ax.legend(loc="lower left", fontsize=7.2)
    ax.set_ylim(-0.9, 1.15)

    heat = [r for r in stab if r["model"] == "truegap_heat"]
    taus = np.array(sorted({round(r["tau_max"], 5) for r in heat}))
    kas = np.array(sorted({round(r["ka"], 5) for r in heat}))
    Z = np.full((len(kas), len(taus)), np.nan)
    ti = {round(x, 5): i for i, x in enumerate(taus)}
    ki = {round(x, 5): i for i, x in enumerate(kas)}
    for r in heat:
        val = r["hinf_tauact"]
        if not np.isfinite(val):
            val = 2.5
        Z[ki[round(r["ka"], 5)], ti[round(r["tau_max"], 5)]] = min(val, 2.5)
    ax = axes[2]
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[taus[0], taus[-1], kas[0], kas[-1]],
        cmap="cividis", vmin=0.9, vmax=1.8,
    )
    ax.set_xlabel(r"communication delay $\tau$ (s)")
    ax.set_ylabel(r"feedforward gain $k_a$")
    ax.set_title(r"true-gap $\|\Gamma\|_\infty$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.55)
    _save(fig, "fig3_string_stability")


def fig_timeseries(series):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.05), sharex="col")
    wave = (
        ("idm", "IDM"),
        ("truegap", "true-gap"),
        ("linear_hinf", "soft PD"),
    )
    brake = (
        ("idm", "IDM"),
        ("cidm", "IDM + feedforward"),
        ("truegap", "true-gap"),
        ("linear_hinf", "soft PD"),
    )
    ax = axes[0, 0]
    t0 = series["sinusoid_idm_t"]
    ax.plot(t0, series["sinusoid_idm_v"][:, 0] * 3.6, label="leader", **STYLE["leader"])
    for kind, label in wave:
        t = series[f"sinusoid_{kind}_t"]
        ax.plot(t, series[f"sinusoid_{kind}_v"][:, -1] * 3.6, label=label, **STYLE[kind])
    ax.set_ylabel("tail speed (km/h)")
    ax.set_title(r"0.7 Hz speed wave")
    ax.set_xlim(10, 45)

    ax = axes[1, 0]
    for kind, _label in wave:
        t = series[f"sinusoid_{kind}_t"]
        ax.plot(t, series[f"sinusoid_{kind}_s"][:, -1], **STYLE[kind])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tail bumper gap (m)")
    ax.set_xlim(10, 45)

    ax = axes[0, 1]
    ax.plot(series["brake_idm_t"], series["brake_idm_a"][:, 0], label="leader", **STYLE["leader"])
    for kind, label in brake:
        t = series[f"brake_{kind}_t"]
        ax.plot(t, series[f"brake_{kind}_a"][:, -1], label=label, **STYLE[kind])
    ax.set_ylabel(r"tail acceleration (m/s$^2$)")
    ax.set_title(r"leader brake, $3.5$ m/s$^2$")
    ax.set_xlim(12, 32)

    ax = axes[1, 1]
    for kind, _label in brake:
        t = series[f"brake_{kind}_t"]
        ax.plot(t, np.min(series[f"brake_{kind}_s"][:, 1:], axis=1), **STYLE[kind])
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("minimum bumper gap (m)")
    ax.set_xlim(12, 32)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.02), fontsize=7.2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, hspace=0.32, wspace=0.32)
    _save(fig, "fig4_platoon")


def fig_amplification(series):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    markers = {
        "idm": "o", "eidm": "v", "cidm": "s", "truegap": "D",
        "virtual": "^", "linear": "P", "linear_hinf": "X",
    }
    order = (
        ("idm", "IDM"),
        ("eidm", "eIDM"),
        ("cidm", "IDM + feedforward"),
        ("virtual", "virtual speed"),
        ("truegap", "true-gap"),
        ("linear", "linear CACC"),
        ("linear_hinf", "soft PD"),
    )
    for ax, scenario, title in (
        (axes[0], "sinusoid", "0.7 Hz wave"),
        (axes[1], "brake", "braking transient"),
    ):
        for kind, label in order:
            key = f"{scenario}_{kind}_l2"
            if key not in series:
                continue
            ratio = series[key]
            ax.plot(
                np.arange(len(ratio)), ratio,
                marker=markers[kind], ms=4.5, markerfacecolor="white", markeredgewidth=1.0,
                label=label, **STYLE[kind],
            )
        ax.axhline(1.0, color="0.45", lw=0.6, ls=(0, (1, 1)))
        ax.set_xlabel("vehicle index (0 = leader)")
        ax.set_title(title)
    axes[0].set_ylabel(r"empirical $L_2$ speed ratio")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.04), fontsize=7.2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    _save(fig, "fig5_amplification")


def fig_montecarlo(mc):
    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    specs = (
        ("truegap", "true-gap", "o", "-"),
        ("linear_hinf", "soft PD", "s", "-."),
    )
    for ctrl, label, marker, ls in specs:
        rows = [r for r in mc if r["channel"] == "bernoulli" and r["controller"] == ctrl]
        ps = sorted({r["p_deliver"] for r in rows})
        med, lo, hi = [], [], []
        for p in ps:
            chunk = [r["l2_tail"] for r in rows if r["p_deliver"] == p]
            med.append(np.median(chunk))
            lo.append(np.quantile(chunk, 0.1))
            hi.append(np.quantile(chunk, 0.9))
        med, lo, hi = map(np.array, (med, lo, hi))
        color = STYLE[ctrl]["color"]
        ax.fill_between(ps, lo, hi, color=color, alpha=0.16, linewidth=0)
        ax.plot(
            ps, med, color=color, marker=marker, ls=ls, lw=1.6,
            markerfacecolor="white", markeredgewidth=1.1, label=label,
        )
        g = [r["l2_tail"] for r in mc if r["channel"] == "gilbert" and r["controller"] == ctrl]
        if g:
            ax.scatter(
                [0.86], [np.median(g)], marker="D", s=32, color=color,
                zorder=3, edgecolor="black", linewidth=0.4,
            )
    ax.axhline(1.0, color="0.45", lw=0.6, ls=(0, (1, 1)))
    ax.set_xlabel("Bernoulli delivery probability")
    ax.set_ylabel(r"tail $L_2$ speed ratio")
    ax.set_title("packet loss, delay 0.1 s")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D(
        [0], [0], marker="D", color="0.2", ls="none",
        markerfacecolor="white", markeredgewidth=0.8, label="Gilbert–Elliott",
    ))
    ax.legend(
        handles, labels + ["Gilbert–Elliott"],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    _save(fig, "fig6_packet_loss")


def fig_safety(sim):
    fig, ax = plt.subplots(figsize=(4.8, 3.45))
    specs = (
        ("hetero", "truegap", "nominal true-gap", "o"),
        ("hetero_cbf", "idm", "true-gap with CBF filter", "s"),
    )
    for scenario, key, label, marker in specs:
        rows = sorted([r for r in sim if r["scenario"] == scenario], key=lambda r: r["alpha"])
        ax.plot(
            [r["alpha"] for r in rows], [r["min_gap"] for r in rows],
            marker=marker, ms=5.5, markerfacecolor="white", markeredgewidth=1.1,
            label=label, **STYLE[key],
        )
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.axvline(0.833, color="0.25", lw=0.8, ls=(0, (3, 1.5)))
    ax.text(
        0.80, 1.1, r"$\alpha_{\min}$", ha="right", va="bottom", fontsize=7.5, color="0.2",
    )
    ax.set_xlabel(r"braking ratio $\alpha = b_F / b_L$ of vehicle 3")
    ax.set_ylabel("minimum bumper gap (m)")
    ax.set_title(r"leader brakes at $3.5$ m/s$^2$")
    _legend_below(ax, ncol=1, fontsize=7.2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    _save(fig, "fig7_safety")


def fig_cutin(series, sim):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.45))
    ax = axes[0]
    traces = (
        ("g010", "gamma_lo", r"true-gap, $\gamma=0.10$"),
        ("g035", "gamma_mid", r"true-gap, $\gamma=0.35$"),
        ("g100", "gamma_hi", r"true-gap, $\gamma=1.00$"),
        ("eidm", "eidm", "eIDM"),
        ("linear_hinf", "linear_hinf", "soft PD"),
    )
    for tag, key, label in traces:
        key_t = f"cut_{tag}_t"
        if key_t not in series:
            continue
        ax.plot(series[key_t], series[f"cut_{tag}_s"], label=label, **STYLE[key])
    ax.axhline(0.0, color="0.45", lw=0.6)
    ax.set_xlim(10, 25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("bumper gap after the cut-in (m)")
    ax.set_title("gap recovery")
    _legend_below(ax, ncol=1, fontsize=6.6)

    ax = axes[1]
    order = [
        ("truegap_g010", r"$\gamma=0.10$"),
        ("truegap_g035", r"$\gamma=0.35$"),
        ("truegap_g100", r"$\gamma=1.00$"),
        ("eidm", "eIDM"),
        ("idm", "IDM"),
        ("linear_hinf", "soft PD"),
    ]
    labels, jerks, brakes = [], [], []
    for key, lab in order:
        hits = [r for r in sim if r["scenario"] == "cutin" and r["controller"] == key]
        if not hits:
            continue
        labels.append(lab)
        jerks.append(hits[0]["max_jerk"])
        brakes.append(-hits[0]["max_brake"])
    x = np.arange(len(labels))
    ax.bar(
        x - 0.16, jerks, width=0.32, color=STYLE["truegap"]["color"],
        label=r"peak |jerk| (m/s$^3$)", zorder=2,
    )
    ax.set_ylabel(r"peak |jerk| (m/s$^3$)")
    ax2 = ax.twinx()
    ax2.bar(
        x + 0.16, brakes, width=0.32, color=STYLE["idm"]["color"],
        hatch="///", edgecolor="white", linewidth=0.4,
        label=r"peak deceleration (m/s$^2$)", zorder=2,
    )
    ax2.set_ylabel(r"peak deceleration (m/s$^2$)")
    ax2.set_ylim(0, max(brakes) * 1.25)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title("cut-in response")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.42),
        ncol=1, fontsize=6.8,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.40, wspace=0.55)
    _save(fig, "fig8_cutin")


def make_all():
    _style()
    eq = _read("equilibrium.csv")
    beta = _read("beta_design.csv")
    stab = _read("stability.csv")
    sim = _read("sim_metrics.csv")
    mc = _read("montecarlo.csv")
    series = dict(np.load(OUT / "series.npz"))
    bode = dict(np.load(OUT / "bode.npz"))
    fig_equilibrium(eq)
    fig_beta(beta)
    fig_stability(stab, bode)
    fig_timeseries(series)
    fig_amplification(series)
    fig_montecarlo(mc)
    fig_safety(sim)
    fig_cutin(series, sim)
    print(f"figures written to {FIG}")
