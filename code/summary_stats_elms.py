# file name: summary_stats_elms.py

# Packages:
import numpy as np

# *** 1. AvgTimeObsStr
# Take SSPrev_obs>0 (presence/absence). Sum across time for each strain ⇒ “how many time points this strain was seen”. Then average over strains that were ever seen.
# → “On average, how long (in time points) an observed strain stayed detectable.”
def avg_time_obs_str(SSPrev_obs: np.ndarray, *, nan_if_none: bool = True) -> float:
    """
    AvgTimeObsStr: average # of time points a *seen* strain is present (>0).

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.
    nan_if_none: if True, return np.nan when no strain is ever seen; else 0.0.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    # presence/absence per strain over time
    present = (X > 0).astype(np.int32)  # (S, T)
    per_strain_counts = present.sum(axis=1)  # (S,)
    seen = per_strain_counts[per_strain_counts > 0]
    if seen.size == 0:
        return float("nan") if nan_if_none else 0.0
    return float(seen.mean())


# *** 2.  MaxTimeObsStr
# From the same per-strain counts: take the maximum.
# → “Longest observed persistence (in time points) among all strains.”

def max_time_obs_str(SSPrev_obs: np.ndarray, *, nan_if_none: bool = True) -> float:
    """
    MaxTimeObsStr: longest observed persistence (in time points) among all strains.

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.
    nan_if_none: if True, return np.nan when no strain is ever seen; else 0.0.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    present = (X > 0).astype(np.int32)  # (S, T) presence/absence
    per_strain_counts = present.sum(axis=1)  # time-points seen for each strain
    seen = per_strain_counts[per_strain_counts > 0]
    if seen.size == 0:
        return float("nan") if nan_if_none else 0.0
    return float(seen.max())


# *** 3. NumStrainsObs
# Count how many strains were ever observed (any nonzero across time).
# → “Total observed strain richness.”

def num_strains_obs_str(SSPrev_obs: np.ndarray) -> int:
    """
    Count how many strains were ever observed (any nonzero across time).

    SSPrev_obs : (n_strains, T) array of counts per strain per time step.

    Returns
    -------
    int
        Number of strains with at least one nonzero entry.
    """
    X = np.asarray(SSPrev_obs)
    return int(np.any(X > 0, axis=1).sum())


# *** 4. AvgTimeRepeatInf
# Apply timerepeat(SSPrev_obs): for each strain’s presence/absence row, take indices where present and compute gaps between successive presences; pool all strains’ gaps together and take the mean.
# → “Average time between repeat detections (recurrences) across strains.”

def avg_time_repeat_inf_numpy(SSPrev_obs: np.ndarray, *, nan_if_empty: bool = True) -> float:
    """
    Average time between repeat detections across strains.
    For each strain (row), take presence indices, compute gaps (np.diff),
    pool all gaps across strains, then return the mean gap.
    """
    X = (np.asarray(SSPrev_obs, float) > 0)  # (S, T) boolean presence
    gaps_all = []
    for row in X:
        idx = np.flatnonzero(row)  # times where present
        if idx.size >= 2:
            gaps_all.append(np.diff(idx))  # gaps between presences for this strain
    if not gaps_all:
        return float("nan") if nan_if_empty else 0.0
    gaps = np.concatenate(gaps_all).astype(float)
    return float(gaps.mean())


# *** 5. VarTimeRepeatInf
# Variance of those pooled inter-occurrence gaps.
# → “How variable the recurrence intervals are.”

def var_time_repeat_inf_numpy(SSPrev_obs, *, nan_if_empty: bool = True) -> float:
    """
    Variance (sample, ddof=1) of pooled gaps between repeat detections across strains.
    """
    X = (np.asarray(SSPrev_obs, float) > 0)  # (S, T) boolean presence
    gaps_all = []
    for row in X:
        idx = np.flatnonzero(row)
        if idx.size >= 2:
            gaps_all.append(np.diff(idx))
    if not gaps_all:
        return float("nan") if nan_if_empty else 0.0
    gaps = np.concatenate(gaps_all).astype(float)
    return float(np.var(gaps, ddof=1)) if gaps.size > 1 else (float("nan") if nan_if_empty else 0.0)


# *** 6. AvgPrev
# Mean of Prevalence_obs over time.
# → “Average prevalence in the observation window.”

# def avg_prev_numpy(Prevalence_obs) -> float:
#     """
#     Average prevalence over time.
#     Accepts array-like (T,) or (T,1). NaNs ignored.
#     """
#     y = np.asarray(Prevalence_obs, dtype=float).ravel()
#     return float(np.nanmean(y)) if y.size else float("nan")

def avg_prev_numpy(SSPrev_obs) -> float:
    """
    Average prevalence over time.
    Accepts array-like (T,) or (T,1). NaNs ignored.
    """
    
    Prevalence_obs = SSPrev_obs.sum(axis=0)
    # print("Prevalence_obs:", Prevalence_obs)
    y = np.asarray(Prevalence_obs, dtype=float).ravel()
    return float(np.nanmean(y)) if y.size else float("nan")


# *** 7. VarPrev
# Variance of Prevalence_obs over time.
# → “How much prevalence fluctuates.”

# def var_prev_numpy(Prevalence_obs, ddof: int = 1) -> float:
#     """
#     Variance of prevalence over time (ignores NaNs).
#     Prevalence_obs: array-like (T,) or (T,1)
#     ddof: 1 for sample variance (MATLAB-like), 0 for population variance.
#     """
#     y = np.asarray(Prevalence_obs, dtype=float).ravel()
#     if y.size == 0:
#         return float("nan")
#     # need at least 2 valid points for sample variance
#     if ddof == 1 and np.sum(~np.isnan(y)) < 2:
#         return float("nan")
#     return float(np.nanvar(y, ddof=ddof))

def var_prev_numpy(SSPrev_obs, ddof: int = 1) -> float:
    """
    Variance of prevalence over time (ignores NaNs).
    Prevalence_obs: array-like (T,) or (T,1)
    ddof: 1 for sample variance (MATLAB-like), 0 for population variance.
    """
    
    Prevalence_obs = SSPrev_obs.sum(axis=0)
    # print("Prevalence_obs:", Prevalence_obs)
    y = np.asarray(Prevalence_obs, dtype=float).ravel()
    if y.size == 0:
        return float("nan")
    # need at least 2 valid points for sample variance
    if ddof == 1 and np.sum(~np.isnan(y)) < 2:
        return float("nan")
    return float(np.nanvar(y, ddof=ddof))


# *** 8. AvgDiv
# Mean of Diversity_obs over time (your diversity metric per time step, e.g., reciprocal Simpson).
# → “Average strain diversity over time.”

# def avg_div_numpy(Diversity_obs) -> float:
#     """
#     Average strain diversity over time (ignores NaNs).
#     Diversity_obs: array-like of shape (T,) or (T,1)
#     """
#     y = np.asarray(Diversity_obs, dtype=float).ravel()
#     return float(np.nanmean(y)) if y.size else float("nan")

def avg_div_numpy(SSPrev_obs) -> float:
    """
    Average strain diversity over time (ignores NaNs).
    Diversity_obs: array-like of shape (T,) or (T,1)
    """
    Diversity_obs = (SSPrev_obs > 0).astype(np.int32)
    Diversity_obs = Diversity_obs.sum(axis=0)
    y = np.asarray(Diversity_obs, dtype=float).ravel()
    return float(np.nanmean(y)) if y.size else float("nan")


# *** 9. VarDiv
# Variance of Diversity_obs.
# → “How much strain diversity fluctuates.”

# def var_div_numpy(Diversity_obs, ddof: int = 1) -> float:
#     """
#     Variance of diversity over time (ignores NaNs).
#     Diversity_obs: array-like (T,) or (T,1)
#     ddof: 1 for sample variance (MATLAB-like), 0 for population variance.
#     """
#     y = np.asarray(Diversity_obs, dtype=float).ravel()
#     if y.size == 0:
#         return float("nan")
#     if ddof == 1 and np.sum(~np.isnan(y)) < 2:
#         return float("nan")
#     return float(np.nanvar(y, ddof=ddof))

def var_div_numpy(SSPrev_obs, ddof: int = 1) -> float:
    """
    Variance of diversity over time (ignores NaNs).
    Diversity_obs: array-like (T,) or (T,1)
    ddof: 1 for sample variance (MATLAB-like), 0 for population variance.
    """
    Diversity_obs = (SSPrev_obs > 0).astype(np.int32)
    Diversity_obs = Diversity_obs.sum(axis=0)
    # print("Diversity_obs: ", Diversity_obs)
    y = np.asarray(Diversity_obs, dtype=float).ravel()
    if y.size == 0:
        return float("nan")
    if ddof == 1 and np.sum(~np.isnan(y)) < 2:
        return float("nan")
    return float(np.nanvar(y, ddof=ddof))


# *** 10. MaxAbundance
# max(max(SSPrev_obs)): the single highest count seen for any strain at any time.
# → “Peak abundance of a strain at a time point.”

def max_abundance_numpy(SSPrev_obs) -> float:
    """
    Peak abundance of a strain at any time.
    SSPrev_obs: array-like (n_strains, T). NaNs ignored.
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    if X.size == 0 or np.isnan(X).all():
        return float("nan")
    return float(np.nanmax(X))


# *** 11. AvgNPMI (MI(SSPrev_obs))
# Average normalized pointwise mutual information across strain pairs using presence/absence over time. Positive ⇒ co-occur more than chance; ~0 ⇒ independent; negative ⇒ avoid each other.
# → “Average pairwise co-occurrence beyond chance.”

def avg_npmi_numpy(SSPrev_obs) -> float:
    """
    Average normalized PMI across *pairs of strains* (lower triangle).
    SSP: array-like, shape (S, T), nonnegative counts per strain (rows) over time (cols).
    Returns NaN if there are <2 strains or total count is zero.
    """
    SSP = np.asarray(SSPrev_obs, dtype=float)
    if SSP.ndim != 2:
        raise ValueError("SSP must be 2D (strains x time)")

    S, T = SSP.shape
    if S < 2:
        return np.nan

    TotalO = SSP.sum()
    if TotalO <= 0:
        return np.nan

    # Marginal probabilities P(X=i)
    PX = SSP.sum(axis=1) / TotalO  # shape (S,)

    # Pairwise joint probabilities P(X=i,Y=j) via sum_t min(SSP[i,t], SSP[j,t]) / TotalO
    # Build (S,S,T) of pairwise minima and sum over time (OK for S ~ 42, T modest)
    # mins[i,j,t] = min(SSP[i,t], SSP[j,t])
    A = SSP[:, None, :]            # (S,1,T)
    B = SSP[None, :, :]            # (1,S,T)
    mins = np.minimum(A, B)        # (S,S,T)
    PXY = mins.sum(axis=2) / TotalO  # (S,S)

    # Compute NPMI on the strict lower triangle (j < i)
    # NPMI(i,j) = (log PXY - log PX_i - log PX_j) / (-log PXY), if PXY>0
    # If PXY==0 => -1; if PX_i + PX_j == 0 => 0 (per original logic)
    i_idx, j_idx = np.tril_indices(S, k=-1)
    pxy = PXY[i_idx, j_idx]
    pxi = PX[i_idx]
    pxj = PX[j_idx]

    # Start with all -1 (the value when pxy == 0)
    npmi = np.full(pxy.shape, -1.0, dtype=float)

    # Valid where PX_i + PX_j > 0
    valid_marg = (pxi + pxj) > 0
    # Among those, where PXY > 0
    valid_joint = valid_marg & (pxy > 0)

    # Cases where PX_i + PX_j == 0 → 0 (match MATLAB intent)
    npmi[~valid_marg] = 0.0

    # Compute NPMI where joint > 0
    if np.any(valid_joint):
        lj = np.log(pxy[valid_joint])
        li = np.log(pxi[valid_joint])
        ljj = np.log(pxj[valid_joint])
        denom = -lj  # -log(PXY)
        # Guard tiny denom just in case
        denom = np.where(denom == 0.0, np.finfo(float).eps, denom)
        npmi[valid_joint] = (lj - li - ljj) / denom

    # Average over all pairs (S choose 2), same denominator as MATLAB:
    n_pairs = (S * S - S) // 2
    # Note: positions not in lower triangle don't contribute (we only built npmi over those)
    return float(npmi.sum() / n_pairs)


# *** 12. DivAllIsolates
# First collapse time by summing counts per strain: sum(SSPrev_obs, 2) (i.e., total isolates per strain across all time). Then compute your diversity function div(...) on that vector.
# → “Overall strain diversity across the entire study period (richness + evenness), time-collapsed.”

def div_all_isolates_numpy(SSPrev_obs) -> float:
    """
    Overall strain diversity across the entire study period.
    Steps:
      1) Collapse time: totals per strain = sum over columns (time).
      2) Reciprocal Simpson: D = N*(N-1) / sum_i n_i*(n_i-1)
         (returns 0 if no isolates; equals richness when all n_i ∈ {0,1})
    """
    X = np.asarray(SSPrev_obs, dtype=float)
    if X.size == 0:
        return 0.0
    totals = X.sum(axis=1)  # per-strain totals across time
    N = totals.sum()
    if N <= 1:
        return 0.0
    denom = np.sum(totals * (totals - 1.0))
    if denom <= 0:
        # all totals are 0 or 1 → diversity equals number of nonzero strains
        return float((totals > 0).sum())
    return float(N * (N - 1.0) / denom)