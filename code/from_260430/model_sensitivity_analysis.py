# file name: model_sensitivity_analysis.py

# Source: ./display_sensitive.ipynb

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# robust Pearson
def pearsonr_np(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    # drop NaN pairs
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if x.size < 2:
        return np.nan, x.size
    x = x - x.mean(); y = y - y.mean()
    sx = np.sqrt(np.dot(x, x)); sy = np.sqrt(np.dot(y, y))
    if sx == 0.0 or sy == 0.0:
        return np.nan, x.size
    r = float(np.dot(x, y) / (sx * sy))
    return max(-1.0, min(1.0, r)), x.size


def plot_2x5_A2_panels_with_reps_v2(
    csv_path,
    out_dir,
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time",
    rep_col="run_seed",   # or "rep"
    dpi=180,
    alpha=0.25,
    lw=0.8,
    ms=3.0,
    marker="o",
    # y-range
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.05,
    # plotted definition
    min_points_per_rep=4,
    legend_loc="upper left",
):
    os.makedirs(out_dir, exist_ok=True)

    # ========= 1) Read raw =========
    df0 = pd.read_csv(csv_path)

    # ========= 2) TOTAL seeds per A2: only need A2 + rep =========
    df_total = df0.copy()
    for c in [A2_col, rep_col]:
        df_total[c] = pd.to_numeric(df_total[c], errors="coerce")
    df_total = df_total.dropna(subset=[A2_col, rep_col]).copy()
    df_total[A2_col] = df_total[A2_col].round(10)

    # ========= 3) PLOT data: need A1 + A2 + B1 + rep (no NaNs) =========
    df_plot = df0.copy()
    for c in [A1_col, A2_col, B1_col, rep_col]:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(subset=[A1_col, A2_col, B1_col, rep_col]).copy()
    df_plot[A2_col] = df_plot[A2_col].round(10)
    df_plot[A1_col] = df_plot[A1_col].round(10)

    # Panels
    A2_vals = np.sort(df_total[A2_col].unique())
    n_panels = min(len(A2_vals), 10)

    # ========= 4) y-limits computed from PLOTTED values only =========
    y_all = df_plot[B1_col].to_numpy(float)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        raise ValueError(f"No finite values in {B1_col} after cleaning (df_plot).")

    if y_mode == "minmax":
        y0, y1 = float(y_all.min()), float(y_all.max())
    elif y_mode == "robust":
        q0, q1 = y_quantiles
        y0, y1 = np.quantile(y_all, [q0, q1]).astype(float)
    else:
        raise ValueError("y_mode must be 'robust' or 'minmax'")

    span = y1 - y0
    if not np.isfinite(span) or span <= 0:
        span = max(abs(y0), 1.0)
    y0 -= y_pad * span
    y1 += y_pad * span

    # ========= 5) Plot =========
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    for k, a2 in enumerate(A2_vals[:n_panels]):
        ax = axes[k]

        # ----- TOTAL seeds (not affected by NaNs in B1) -----
        total_reps = int(df_total.loc[df_total[A2_col] == a2, rep_col].nunique())

        # ----- PLOTTED seeds (must have enough valid points) -----
        g = df_plot[df_plot[A2_col] == a2]
        plotted_reps = 0

        for rep, gg in g.groupby(rep_col, sort=True):
            # average duplicates at same A1 within a seed
            gg = gg.groupby(A1_col, as_index=False)[B1_col].mean().sort_values(A1_col)

            x = gg[A1_col].to_numpy()
            y = gg[B1_col].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)

            if m.sum() < min_points_per_rep:
                continue

            plotted_reps += 1
            ax.plot(
                x[m], y[m],
                linestyle="-",
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markersize=ms,
                markerfacecolor="none",
            )

        ax.set_title(f"{A2_col}={a2:g}", fontsize=11)
        ax.set_ylim(y0, y1)

        # legend text (total vs plotted)
        txt = f"samples: {total_reps}\nvalid: {plotted_reps}"
        ax.plot([], [], " ", label=txt)  # invisible handle
        ax.legend(loc=legend_loc, frameon=True, fontsize=9, handlelength=0, handletextpad=0)

    # hide unused axes
    for k in range(n_panels, 10):
        axes[k].axis("off")

    fig.suptitle(f"{A1_col} vs {B1_col} (line+dot per {rep_col})", fontsize=14)
    fig.supxlabel(A1_col)
    fig.supylabel(B1_col)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"{A1_col}_vs_{B1_col}_panels_by_{A2_col}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print("Saved:", out_path)
    print(f"[Y range] mode={y_mode} -> ({y0:.3g}, {y1:.3g})")


def plot_2x5_A2_panels_band_only(
    csv_path,
    out_dir,
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time",
    rep_col="run_seed",
    dpi=180,
    center="mean",               # "mean" or "median"
    band=(0.05, 0.95),
    min_n=30,                    # keep A1 columns with >= min_n valid seeds contributing
    min_points_per_rep=4,        # seed validity: must have >= this many A1 points
    y_mode="robust",
    y_quantiles=(0.05, 0.99),
    top_pad=0.25,
    bottom_pad=0.05,
    legend_loc="upper left",
):
    os.makedirs(out_dir, exist_ok=True)
    df0 = pd.read_csv(csv_path)

    # ---- samples (count seeds BEFORE dropping B1 NaNs) ----
    df_samples = df0.copy()
    for c in [A2_col, rep_col]:
        df_samples[c] = pd.to_numeric(df_samples[c], errors="coerce")
    df_samples = df_samples.dropna(subset=[A2_col, rep_col]).copy()
    df_samples[A2_col] = df_samples[A2_col].round(10)

    # ---- plot data (need A1/A2/B1/seed) ----
    df_plot = df0.copy()
    for c in [A1_col, A2_col, B1_col, rep_col]:
        df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")
    df_plot = df_plot.dropna(subset=[A1_col, A2_col, B1_col, rep_col]).copy()
    df_plot[A2_col] = df_plot[A2_col].round(10)
    df_plot[A1_col] = df_plot[A1_col].round(10)

    A2_vals = np.sort(df_samples[A2_col].unique())
    n_panels = min(len(A2_vals), 10)
    qlo, qhi = band

    # ---- y-limits from plotted values ----
    y_all = df_plot[B1_col].to_numpy(float)
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size == 0:
        raise ValueError(f"No finite values found in {B1_col} (after cleaning df_plot).")

    if y_mode == "minmax":
        y0, y1 = float(y_all.min()), float(y_all.max())
    elif y_mode == "robust":
        q0, q1 = y_quantiles
        y0, y1 = np.quantile(y_all, [q0, q1]).astype(float)
    else:
        raise ValueError("y_mode must be 'robust' or 'minmax'")

    span = y1 - y0
    if not np.isfinite(span) or span <= 0:
        span = max(abs(y0), 1.0)
    y0 -= bottom_pad * span
    y1 += top_pad * span

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
    axes = axes.ravel()

    # figure-level legend handles (band + center)
    band_handle = None
    center_handle = None
    center_label = "Median" if center == "median" else "Mean"
    band_label = f"{int(qlo*100)}–{int(qhi*100)}% band"

    for k, a2 in enumerate(A2_vals[:n_panels]):
        ax = axes[k]

        samples = int(df_samples.loc[df_samples[A2_col] == a2, rep_col].nunique())

        g = df_plot[df_plot[A2_col] == a2]
        if g.empty:
            ax.set_title(f"{A2_col}={a2:g}", fontsize=11)
            ax.set_ylim(y0, y1)
            txt = f"samples: {samples}\nvalid: 0"
            ax.plot([], [], " ", label=txt)
            ax.legend(loc=legend_loc, frameon=True, fontsize=9, handlelength=0, handletextpad=0)
            continue

        # reps × A1 pivot
        P = g.pivot_table(index=rep_col, columns=A1_col, values=B1_col, aggfunc="mean").sort_index(axis=1)
        x = P.columns.to_numpy(float)
        Y = P.to_numpy(float)  # (n_seeds_with_any_data, n_A1)

        # ---- seed validity like with_reps_v2: >= min_points_per_rep points anywhere ----
        counts_per_seed = np.sum(np.isfinite(Y), axis=1)
        valid_seed_mask = counts_per_seed >= min_points_per_rep
        valid = int(valid_seed_mask.sum())

        if valid < 2:
            # too few valid seeds to compute quantiles meaningfully
            ax.set_title(f"{A2_col}={a2:g}", fontsize=11)
            ax.set_ylim(y0, y1)
            txt = f"samples: {samples}\nvalid: {valid}"
            ax.plot([], [], " ", label=txt)
            ax.legend(loc=legend_loc, frameon=True, fontsize=9, handlelength=0, handletextpad=0)
            continue

        # Filter to valid seeds only for band/center
        Yv = Y[valid_seed_mask, :]

        # ---- keep columns based on valid seeds only (optional but consistent) ----
        n_valid_col = np.sum(np.isfinite(Yv), axis=0)
        keep = n_valid_col >= min_n

        # center + band computed from valid seeds
        if center == "median":
            c = np.nanmedian(Yv, axis=0)
        else:
            c = np.nanmean(Yv, axis=0)

        lo = np.nanquantile(Yv, qlo, axis=0)
        hi = np.nanquantile(Yv, qhi, axis=0)

        m = keep & np.isfinite(c) & np.isfinite(lo) & np.isfinite(hi)
        if m.sum() >= 2:
            fb = ax.fill_between(x[m], lo[m], hi[m], alpha=0.25)
            ln, = ax.plot(x[m], c[m], linewidth=2.2)
            if band_handle is None:
                band_handle = fb
            if center_handle is None:
                center_handle = ln

        ax.set_title(f"{A2_col}={a2:g}", fontsize=11)
        ax.set_ylim(y0, y1)

        # subplot legend: samples/valid only
        txt = f"samples: {samples}\nvalid: {valid}"
        ax.plot([], [], " ", label=txt)
        ax.legend(loc=legend_loc, frameon=True, fontsize=9, handlelength=0, handletextpad=0)

    for k in range(n_panels, 10):
        axes[k].axis("off")

    fig.suptitle(
        f"{A1_col} vs {B1_col}: {center_label} + {int(qlo*100)}–{int(qhi*100)}% band (per {A2_col})",
        fontsize=14
    )
    fig.supxlabel(A1_col)
    fig.supylabel(B1_col)

    # figure-level legend for band + center
    if band_handle is not None and center_handle is not None:
        fig.legend([center_handle, band_handle], [center_label, band_label],
                   loc="upper right", frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(out_dir, f"{A1_col}_vs_{B1_col}_{center}_band_by_{A2_col}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    print("Saved:", out_path)
    print(f"[Y range] mode={y_mode}, y_quantiles={y_quantiles} -> ({y0:.3g}, {y1:.3g})")



# --- your CSV ---
files = [
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch00.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch01.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch02.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch03.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch04.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch05.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch06.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch07.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch08.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch09.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch10.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch11.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch12.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch13.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch14.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch15.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch16.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch17.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch18.csv",
    "../../experimental_data/from_260430/R0_sigma_sensitive_2params_batch19.csv"
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
cols = ["R0", "sigma", "run_seed"]  # change as needed
print(df[cols].nunique(dropna=True))
df.to_csv("../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv", index=False)
print("Saved merged.csv, rows:", len(df))


# ******1. R0 vs avg_time by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******2. R0 vs max_time by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_time",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_time",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_time",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******3. R0 vs num_strains by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="num_strains",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="num_strains",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="num_strains",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******4. R0 vs avg_time_repeat by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time_repeat",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******5. R0 vs var_time_repeat by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_time_repeat",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******6. R0 vs avg_prev by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_prev",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_prev",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_prev",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******7. R0 vs var_prev by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_prev",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_prev",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_prev",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******8. R0 vs avg_div by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_div",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_div",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_div",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******9. R0 vs var_div by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_div",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_div",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="var_div",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******10. R0 vs max_abundance by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_abundance",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_abundance",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_abundance",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******11. R0 vs avg_npmi by sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_npmi",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_npmi",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="avg_npmi",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



# ******12. R0 vs div_all_isolatesby sigma
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="div_all_isolates",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="div_all_isolates",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/R0_vs_ss_panels_by_sigma",
    A1_col="R0",
    A2_col="sigma",
    B1_col="div_all_isolates",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# -----------------------------------------------------------------------

# ******1. sigma vs avg_time by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******2. sigma vs max_time by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_time",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_time",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_time",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******3. sigma vs num_strains by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="num_strains",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="num_strains",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="num_strains",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******4. sigma vs avg_time_repeat by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time_repeat",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******5. sigma vs var_time_repeat by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_time_repeat",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_time_repeat",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******6. sigma vs avg_prev by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_prev",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_prev",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_prev",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******7. sigma vs var_prev by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_prev",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_prev",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_prev",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******8. sigma vs avg_div by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_div",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_div",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_div",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******9. sigma vs var_div by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_div",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_div",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="var_div",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******10. sigma vs max_abundance by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_abundance",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_abundance",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="max_abundance",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******11. sigma vs avg_npmi by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_npmi",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_npmi",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="avg_npmi",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)


# ******12. sigma vs div_all_isolates by R0
plot_2x5_A2_panels_with_reps_v2(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="div_all_isolates",
    rep_col="run_seed",  # or "run_seed" if that's what you saved
    y_mode="robust",
    y_quantiles=(0.01, 0.99),
    y_pad=0.20
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="div_all_isolates",
    rep_col="run_seed",
    min_n=300,
    center="mean",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)
)

plot_2x5_A2_panels_band_only(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    out_dir="../../figures/from_260430/sigma_vs_ss_panels_by_R0",
    A1_col="sigma",
    A2_col="R0",
    B1_col="div_all_isolates",
    rep_col="run_seed",
    min_n=300,
    center="median",
    top_pad=0.20,  # more headroom
    bottom_pad=0.20,
    y_quantiles=(0.01, 0.99)

)



