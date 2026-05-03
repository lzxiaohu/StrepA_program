# file name: pcc_display.py

# Source: ./pcc_display.ipynb


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pearsonr_simple(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 2:
        return np.nan, n
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x*x).sum() * (y*y).sum())
    if denom == 0:
        return np.nan, n
    return float((x*y).sum() / denom), n


def pcc_A1_B1_by_A2_and_seed(
    csv_path,
    A1_col="R0",
    A2_col="sigma",
    B1_col="max_time",
    seed_col="run_seed",
    min_points=4,
    round_A2=10,
    round_A1=10,
):
    df = pd.read_csv(csv_path)

    # numeric safety
    for c in [A1_col, A2_col, B1_col, seed_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[A1_col, A2_col, B1_col, seed_col]).copy()

    # avoid float mismatches
    df[A2_col] = df[A2_col].round(round_A2)
    df[A1_col] = df[A1_col].round(round_A1)

    rows = []
    nan_num = 0
    too_short = 0

    for (a2, seed), g in df.groupby([A2_col, seed_col], sort=True):
        # average duplicates at same A1 within seed
        gg = g.groupby(A1_col, as_index=False)[B1_col].mean()

        x = gg[A1_col].to_numpy(dtype=float)
        y = gg[B1_col].to_numpy(dtype=float)

        # keep only finite pairs (important)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]; y = y[m]
        n = x.size

        # ---- check length BEFORE computing PCC ----
        if n < min_points:
            too_short += 1
            continue

        r, _ = pearsonr_simple(x, y)  # your helper returns (r, n); n==x.size anyway

        if not np.isfinite(r):
            nan_num += 1
            continue

        rows.append((a2, seed, r, n))

    print(f"groups too short (<{min_points}): {too_short}")
    print(f"NaN/inf PCC among groups with n>={min_points}: {nan_num}")
    print(f"kept rows: {len(rows)}")

    return pd.DataFrame(rows, columns=[A2_col, seed_col, "pcc", "n"])


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def kde_all_pcc(
        res_df,
        pcc_col="pcc",
        A1_col="R0",
        B1_col="avg_time",
        bandwidth=0.08,
        gridsize=800,
        dpi=180,
        save_path=None,  # None OR folder OR full .png path
):
    x = res_df[pcc_col].to_numpy(dtype=float)
    total_sampes = x.size
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        raise ValueError("Not enough PCC points for KDE.")

    X = x.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)

    xs = np.linspace(x.min(), x.max(), gridsize).reshape(-1, 1)
    dens = np.exp(kde.score_samples(xs))

    # summary stats
    x_mean = float(np.mean(x))
    x_med = float(np.median(x))
    x_mode = float(xs[int(np.argmax(dens)), 0])

    plt.figure(figsize=(6.6, 4.3))
    plt.hist([], color="black", label=f"samples (n={total_sampes})")
    plt.hist(x, bins=30, density=True, alpha=0.30, edgecolor="black",
             label=f"valid (n={n})")

    plt.plot(xs.ravel(), dens, linewidth=2.2, color="C1", label=f"KDE (bw={bandwidth:g})")
    plt.axvline(x_mode, linestyle="-", color="C1", linewidth=1.6, label=f"KDE mode={x_mode:.3g}")
    plt.axvline(x_mean, linestyle="--", linewidth=1.4, label=f"Mean={x_mean:.3g}")
    plt.axvline(x_med, linestyle=":", linewidth=1.6, label=f"Median={x_med:.3g}")

    plt.xlabel("PCC")
    plt.ylabel("Density")
    plt.title(f"PCC between {A1_col} and {B1_col} (all seeds)")
    plt.legend(loc="center left", fontsize=8, bbox_to_anchor=(0.72, 0.78), borderaxespad=0.)
    plt.tight_layout()

    # --- adaptive save behavior ---
    if save_path is None:
        plt.show()
        return None

    # If save_path is a directory (or has no .png), treat as folder
    if save_path.endswith(os.sep) or (os.path.isdir(save_path)) or (not str(save_path).lower().endswith(".png")):
        out_dir = save_path
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{A1_col}_vs_{B1_col}_pcc_kde_all.png".replace(" ", "_")
        out_file = os.path.join(out_dir, fname)
    else:
        # full file path provided
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_file = save_path

    plt.savefig(out_file, dpi=dpi)
    plt.close()
    print("Saved:", out_file)
    return out_file


def boxplot_pcc_by_A2(
    res_df,
    A2_col="sigma",
    pcc_col="pcc",
    A1_col="R0",
    B1_col="avg_time",
    title=None,
    dpi=180,
    save_path=None,
    show=False,
    showfliers=False,
    annotate_fmt="samples={n}\nvalid={v}",   # text shown above each box
    annotate_fontsize=9,
    annotate_y_pad_frac=0.03,      # vertical padding as fraction of y-range
    rotate_xticks=0
):
    a2_vals = np.sort(res_df[A2_col].unique())

    data = []
    samples_list = []
    valid_list = []
    for a2 in a2_vals:
        g = res_df.loc[res_df[A2_col] == a2, pcc_col]
        samples = len(g)
        valid = int(g.notna().sum())
        arr = g.dropna().to_numpy(dtype=float)

        data.append(arr)
        samples_list.append(samples)
        valid_list.append(valid)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    bp = ax.boxplot(data, labels=[f"{v:g}" for v in a2_vals], showfliers=showfliers)
    ax.axhline(0.0, linewidth=1)

    ax.set_xlabel(A2_col)
    ax.set_ylabel(f"Pearson r ({A1_col} vs {B1_col})")
    ax.set_title(title if title is not None else f"PCC between {A1_col} and {B1_col} (grouped by {A2_col})")

    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    # ---- annotate counts close to each box ----
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    y_pad = annotate_y_pad_frac * y_span

    # place annotation just above the top whisker (or top of box if whisker missing)
    for i, (samples, valid) in enumerate(zip(samples_list, valid_list), start=1):
        # whiskers are 2 per box: indices 2*(i-1), 2*(i-1)+1
        w_top = bp["whiskers"][2*(i-1) + 1]
        y_top = float(np.max(w_top.get_ydata())) if w_top.get_ydata().size else y_max

        txt = annotate_fmt.format(n=samples, v=valid)
        ax.text(i, y_top + y_pad, txt, ha="center", va="bottom", fontsize=annotate_fontsize)

    # expand y-limit a bit so annotations are not cut off
    ax.set_ylim(y_min, y_max + 3*y_pad)

    fig.tight_layout()

    # ---- save behavior ----
    if save_path is None:
        plt.show()
        return None

    save_path = str(save_path)
    if save_path.endswith(os.sep) or os.path.isdir(save_path) or (not save_path.lower().endswith(".png")):
        out_dir = save_path
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{A1_col}_vs_{B1_col}_by_{A2_col}_pcc_boxplot.png".replace(" ", "_")
        out_file = os.path.join(out_dir, fname)
    else:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_file = save_path

    plt.savefig(out_file, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print("Saved:", out_file)
    return out_file


def violin_pcc_by_A2(
    res_df,
    A2_col="sigma",
    pcc_col="pcc",
    A1_col="R0",
    B1_col="avg_time",
    title=None,
    show_median=True,
    show_iqr=True,
    dpi=180,
    save_path=None,
    show=False,
    # band-like style
    band_color="C0",
    alpha=0.25,
    edge_color="C0",
    edge_lw=1.0,
    # annotation
    annotate=True,
    annotate_fmt="samples={n}\nvalid={v}",
    annotate_fontsize=9,
    annotate_y_pad_frac=0.03,
    legend_loc=None,   # optional, e.g. "upper right"
):
    a2_vals = np.sort(res_df[A2_col].unique())
    positions = np.arange(1, len(a2_vals) + 1)

    data = []
    samples_list = []
    valid_list = []
    for a2 in a2_vals:
        g = res_df.loc[res_df[A2_col] == a2, pcc_col]
        samples = len(g)
        valid = int(g.notna().sum())
        arr = g.dropna().to_numpy(dtype=float)

        data.append(arr)
        samples_list.append(samples)
        valid_list.append(valid)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    vp = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # same band-like color for all violins
    for body in vp["bodies"]:
        body.set_facecolor(band_color)
        body.set_edgecolor(edge_color)
        body.set_alpha(alpha)
        body.set_linewidth(edge_lw)

    # median / IQR overlays
    for i, y in enumerate(data, start=1):
        if y.size == 0:
            continue
        if show_iqr:
            q1, q3 = np.quantile(y, [0.25, 0.75])
            ax.plot([i, i], [q1, q3], linewidth=3, color=edge_color, alpha=0.9)
        if show_median:
            med = np.median(y)
            ax.scatter([i], [med], s=24, zorder=3, color=edge_color)

    ax.axhline(0.0, linewidth=1, color="k", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{v:g}" for v in a2_vals])
    ax.set_xlabel(A2_col)
    ax.set_ylabel(f"Pearson r ({A1_col} vs {B1_col})")
    ax.set_title(title if title is not None else f"PCC between {A1_col} and {B1_col} (grouped by {A2_col})")

    # ---- annotate n/v above each violin ----
    if annotate:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min if y_max > y_min else 1.0
        y_pad = annotate_y_pad_frac * y_span

        # choose top position per violin from its data
        for i, (y, n, v) in enumerate(zip(data, samples_list, valid_list), start=1):
            if y.size == 0:
                y_top = y_min
            else:
                y_top = float(np.nanmax(y))
            txt = annotate_fmt.format(n=n, v=v)
            ax.text(i, y_top + y_pad, txt, ha="center", va="bottom", fontsize=annotate_fontsize)

        # expand ylim so text isn't clipped
        ax.set_ylim(y_min, y_max + 3*y_pad)

    if legend_loc:
        ax.legend(loc=legend_loc, frameon=False)

    fig.tight_layout()

    # ----- adaptive save behavior -----
    if save_path is None:
        plt.show()
        return None

    save_path = str(save_path)
    if save_path.endswith(os.sep) or os.path.isdir(save_path) or (not save_path.lower().endswith(".png")):
        out_dir = save_path
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{A1_col}_vs_{B1_col}_by_{A2_col}_pcc_violin.png".replace(" ", "_")
        out_file = os.path.join(out_dir, fname)
    else:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_file = save_path

    fig.savefig(out_file, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print("Saved:", out_file)
    return out_file


# *******1 compute PCC between R0 and avg_time
A1_col="R0"
A2_col="sigma"
B1_col="avg_time"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(
    res,
    A1_col=A1_col,
    B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"
)

boxplot_pcc_by_A2(
    res,
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"
)

violin_pcc_by_A2(
    res,
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"
)


# *******2 compute PCC between R0 and max_time
A1_col="R0"
A2_col="sigma"
B1_col="max_time"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)



# *******3 compute PCC between R0 and num_strains
A1_col="R0"
A2_col="sigma"
B1_col="num_strains"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)



# *******4 compute PCC between R0 and avg_time_repeat
A1_col="R0"
A2_col="sigma"
B1_col="avg_time_repeat"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)

# *******5 compute PCC between R0 and var_time_repeat
A1_col="R0"
A2_col="sigma"
B1_col="var_time_repeat"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)


# *******6 compute PCC between R0 and avg_prev
A1_col="R0"
A2_col="sigma"
B1_col="avg_prev"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)


# *******7 compute PCC between R0 and var_prev
A1_col="R0"
A2_col="sigma"
B1_col="var_prev"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)

# *******8 compute PCC between R0 and avg_div
A1_col="R0"
A2_col="sigma"
B1_col="avg_div"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)

# *******9 compute PCC between R0 and var_div
A1_col="R0"
A2_col="sigma"
B1_col="var_div"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)


# *******10 compute PCC between R0 and max_abundance
A1_col="R0"
A2_col="sigma"
B1_col="max_abundance"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)

# *******11 compute PCC between R0 and avg_npmi
A1_col="R0"
A2_col="sigma"
B1_col="avg_npmi"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)

# *******12 compute PCC between R0 and div_all_isolates
A1_col="R0"
A2_col="sigma"
B1_col="div_all_isolates"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_R0_vs_ss_panels_by_sigma/"   # folder -> auto filename
)


# *******1 compute PCC between sigma and avg_time
A1_col="sigma"
A2_col="R0"
B1_col="avg_time"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******2 compute PCC between sigma and max_time
A1_col="sigma"
A2_col="R0"
B1_col="max_time"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)


# *******3 compute PCC between sigma and num_strains
A1_col="sigma"
A2_col="R0"
B1_col="num_strains"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******4 compute PCC between sigma and avg_time_repeat
A1_col="sigma"
A2_col="R0"
B1_col="avg_time_repeat"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)


# *******5 compute PCC between sigma and var_time_repeat
A1_col="sigma"
A2_col="R0"
B1_col="var_time_repeat"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******6 compute PCC between sigma and avg_prev
A1_col="sigma"
A2_col="R0"
B1_col="avg_prev"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)


# *******7 compute PCC between sigma and var_prev
A1_col="sigma"
A2_col="R0"
B1_col="var_prev"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******8 compute PCC between sigma and avg_div
A1_col="sigma"
A2_col="R0"
B1_col="avg_div"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******9 compute PCC between sigma and var_div
A1_col="sigma"
A2_col="R0"
B1_col="var_div"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******10 compute PCC between sigma and max_abundance
A1_col="sigma"
A2_col="R0"
B1_col="max_abundance"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******11 compute PCC between sigma and avg_npmi
A1_col="sigma"
A2_col="R0"
B1_col="avg_npmi"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)

# *******12 compute PCC between sigma and avg_time
A1_col="sigma"
A2_col="R0"
B1_col="div_all_isolates"

res = pcc_A1_B1_by_A2_and_seed(
    csv_path="../../experimental_data/from_260430/R0_sigma_sensitive_2params_merged.csv",
    A1_col=A1_col,
    A2_col=A2_col,
    B1_col=B1_col,
    seed_col="run_seed",   # change to "rep" if that’s what you have
    min_points=4
)

kde_all_pcc(res,
            A1_col=A1_col,
            B1_col=B1_col,
            save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

boxplot_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
                  save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/")

violin_pcc_by_A2(res,
                  A1_col=A1_col,
                  A2_col=A2_col,
                  B1_col=B1_col,
    save_path="../../figures/from_260430/pcc_sigma_vs_ss_panels_by_R0/"   # folder -> auto filename
)
