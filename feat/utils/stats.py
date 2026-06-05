"""
Feat utility and helper functions for performing statistics.
"""

import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
import torch

__all__ = [
    "wavelet",
    "calc_hist_auc",
    "softmax",
    "cluster_identities",
    "regress",
    "downsample",
    "upsample",
    "set_decomposition_algorithm",
    "clean_signal",
]


def wavelet(freq, num_cyc=3, sampling_freq=30.0):
    """Create a complex Morlet wavelet.

    Creates a complex Morlet wavelet by windowing a cosine function by a Gaussian. All formulae taken from Cohen, 2014 Chaps 12 + 13

    Args:
        freq: (float) desired frequency of wavelet
        num_cyc: (float) number of wavelet cycles/gaussian taper. Note that smaller cycles give greater temporal precision and that larger values give greater frequency precision; (default: 3)
        sampling_freq: (float) sampling frequency of original signal.

    Returns:
        wav: (ndarray) complex wavelet
    """
    dur = (1 / freq) * num_cyc
    time = np.arange(-dur, dur, 1.0 / sampling_freq)

    # Cosine component
    sin = np.exp(2 * np.pi * 1j * freq * time)

    # Gaussian component
    sd = num_cyc / (2 * np.pi * freq)  # standard deviation
    gaus = np.exp(-(time**2.0) / (2.0 * sd**2.0))

    return sin * gaus


def calc_hist_auc(vals, hist_range=None):
    """Calculate histogram area under the curve.

    This function follows the bag of temporal feature analysis as described in Bartlett, M. S., Littlewort, G. C., Frank, M. G., & Lee, K. (2014). Automatic decoding of facial movements reveals deceptive pain expressions. Current Biology, 24(7), 738-743. The function receives convolved data, squares the values, finds 0 crossings to calculate the AUC(area under the curve) and generates a 6 exponentially-spaced-bin histogram for each data.

    Args:
        vals:

    Returns:
        Series of histograms
    """
    # Square values
    vals = [elem**2 if elem > 0 else -1 * elem**2 for elem in vals]
    # Get 0 crossings
    crossings = np.where(np.diff(np.sign(vals)))[0]
    pos, neg = [], []
    for i in range(len(crossings)):
        if i == 0:
            cross = vals[: crossings[i]]
        elif i == len(crossings) - 1:
            cross = vals[crossings[i] :]
        else:
            cross = vals[crossings[i] : crossings[i + 1]]
        if cross:
            auc = simps(cross)
            if auc > 0:
                pos.append(auc)
            elif auc < 0:
                neg.append(np.abs(auc))
    if not hist_range:
        hist_range = np.logspace(0, 5, 7)  # bartlett 10**0~ 10**5

    out = pd.Series(
        np.hstack([np.histogram(pos, hist_range)[0], np.histogram(neg, hist_range)[0]])
    )
    return out


def softmax(x):
    """
    Softmax function to change log likelihood evidence values to probabilities.
    Use with Evidence values from FACET.

    Args:
        x: value to softmax
    """
    return 1.0 / (1 + 10.0**-(x))


def regress(X, y, mode="ols", **kwargs):
    """Ordinary least squares multiple regression.

    Drop-in replacement for nltools.stats.regress for the ols path; other
    modes (e.g., robust, ridge) are not implemented.

    Args:
        X: [n, p] design matrix (numpy array or array-like).
        y: [n] or [n, k] response.
        mode: only ``"ols"`` is supported.

    Returns:
        (beta, se, t_stats, p_vals, df, residuals).
        beta/se/t_stats/p_vals shape: [p, k] (or [p] if y is 1-D).
        df is a scalar (n - p). residuals shape: [n, k] (or [n]).
    """
    if mode != "ols":
        raise NotImplementedError(
            f"mode={mode!r} is not supported by feat.utils.stats.regress; only 'ols'"
        )
    if kwargs:
        raise TypeError(f"unexpected keyword args: {sorted(kwargs)}")
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    one_d = y_arr.ndim == 1
    if one_d:
        y_arr = y_arr[:, None]

    n, p = X_arr.shape
    XtX_inv = np.linalg.pinv(X_arr.T @ X_arr)
    beta = XtX_inv @ X_arr.T @ y_arr
    res = y_arr - X_arr @ beta
    df = n - p
    sigma2 = (res ** 2).sum(axis=0) / max(df, 1)
    diag = np.diag(XtX_inv)
    se = np.sqrt(np.outer(diag, sigma2))
    # t-stats; protect against div-by-zero for zero columns
    se_safe = np.where(se == 0, np.finfo(se.dtype).tiny, se)
    t_stats = beta / se_safe

    from scipy.stats import t as _t

    # Use sf (1 - cdf) directly to avoid precision loss when |t| is large.
    p_vals = 2 * _t.sf(np.abs(t_stats), df)

    if one_d:
        beta = beta.squeeze(-1)
        se = se.squeeze(-1)
        t_stats = t_stats.squeeze(-1)
        p_vals = p_vals.squeeze(-1)
        res = res.squeeze(-1)
    return beta, se, t_stats, p_vals, df, res


def downsample(data, sampling_freq, target, target_type="samples", method="mean"):
    """Block-aggregate downsample a DataFrame's rows.

    Drop-in replacement for ``nltools.stats.downsample``. ``target`` is
    interpreted by ``target_type``:

    - ``'samples'`` (default, matches nltools): ``target`` is the number
      of consecutive rows aggregated per output row.
    - ``'seconds'``: ``target`` is the duration of each output bin in
      seconds; bin size in samples = ``round(target * sampling_freq)``.
    - ``'hz'``: ``target`` is the desired output sampling rate in Hz;
      bin size in samples = ``round(sampling_freq / target)``.

    Output row count is ``ceil(n_input / bin_size)``; the final bin can
    contain fewer than ``bin_size`` rows (matches nltools).

    Args:
        data: pandas.DataFrame (or 2-D array-like) where rows are time samples.
        sampling_freq: original sampling frequency in Hz.
        target: see ``target_type``.
        target_type: ``'samples'`` (default), ``'seconds'``, or ``'hz'``.
        method: ``'mean'`` (default) or ``'median'``.
    """
    if target_type == "samples":
        n_samples = int(target)
    elif target_type == "seconds":
        n_samples = int(round(float(target) * float(sampling_freq)))
    elif target_type == "hz":
        if target > sampling_freq:
            raise ValueError(
                f"target ({target}) must be <= sampling_freq ({sampling_freq}) "
                f"when target_type='hz'"
            )
        n_samples = int(round(float(sampling_freq) / float(target)))
    else:
        raise ValueError(
            f"target_type must be 'samples', 'seconds', or 'hz', not {target_type!r}"
        )

    if n_samples <= 0:
        raise ValueError(f"computed bin size must be > 0, got {n_samples}")

    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', not {method!r}")

    if n_samples == 1:
        return data.copy() if hasattr(data, "copy") else np.array(data, copy=True)

    is_df = isinstance(data, pd.DataFrame)
    arr = data if is_df else pd.DataFrame(np.asarray(data))
    n = len(arr)
    group_idx = np.arange(n) // n_samples
    grouped = arr.groupby(group_idx)
    out = grouped.mean() if method == "mean" else grouped.median()
    out.reset_index(drop=True, inplace=True)
    return out if is_df else out.values


def upsample(data, sampling_freq, target, target_type="hz", **kwargs):
    """Upsample a DataFrame's rows by Fourier-domain resampling.

    Drop-in replacement for nltools.stats.upsample.

    Args:
        data: pandas.DataFrame (or 2-D array-like).
        sampling_freq: original sampling frequency in Hz.
        target: target frequency or duration; interpretation set by target_type.
        target_type: 'hz' (target is target sampling rate in Hz),
            'samples' (target is the desired sample count),
            'seconds' (target is the period of the upsampled signal in seconds).

    Returns:
        Same type as input, with the new row count.
    """
    from scipy.signal import resample

    if kwargs:
        raise TypeError(f"unexpected keyword args: {sorted(kwargs)}")

    n_in = len(data)
    if target_type == "hz":
        factor = target / sampling_freq
        n_out = int(round(n_in * factor))
    elif target_type == "samples":
        n_out = int(round(target))
    elif target_type == "seconds":
        n_out = int(round(target * sampling_freq))
    else:
        raise ValueError(
            f"target_type must be 'hz', 'samples', or 'seconds'; got {target_type!r}"
        )

    if isinstance(data, pd.DataFrame):
        upsampled = resample(data.to_numpy(), n_out, axis=0)
        return pd.DataFrame(upsampled, columns=data.columns)
    return resample(np.asarray(data), n_out, axis=0)


def set_decomposition_algorithm(algorithm="pca", n_components=None, *args, **kwargs):
    """Return an unfit sklearn decomposition object by name.

    Drop-in replacement for nltools.utils.set_decomposition_algorithm.

    Args:
        algorithm: one of 'pca', 'ica', 'nnmf', 'fa'.
        n_components: passed through to the sklearn class.
        Additional args/kwargs are forwarded to the sklearn class constructor.
    """
    from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis

    algo = algorithm.lower()
    if algo == "pca":
        return PCA(n_components=n_components, *args, **kwargs)
    if algo == "ica":
        return FastICA(n_components=n_components, *args, **kwargs)
    if algo == "nnmf":
        return NMF(n_components=n_components, *args, **kwargs)
    if algo == "fa":
        return FactorAnalysis(n_components=n_components, *args, **kwargs)
    raise ValueError(
        f"Unknown algorithm {algorithm!r}; use 'pca', 'ica', 'nnmf', or 'fa'"
    )


def clean_signal(
    signals,
    *,
    detrend=True,
    standardize=True,
    confounds=None,
    low_pass=None,
    high_pass=None,
    ensure_finite=False,
    sampling_freq=1.0,
    runs=None,
):
    """Clean a 2D time-series signal: detrend, filter, regress confounds, standardize.

    Drop-in replacement for the parts of ``nilearn.signal.clean`` that
    ``Fex.clean`` uses, so py-feat can avoid taking on nilearn (and its
    transitive nibabel/joblib/sklearn deps) just for time-series cleanup.

    Operations are applied in nilearn's order:
        1. Detrend (linear, optional)
        2. Butterworth low/high/bandpass filter (optional, uses ``filtfilt``)
        3. Regress out confounds (optional). Confounds are filtered with the
           same Butterworth before regression so the filter and confound-
           removal operators stay orthogonal (Lindquist et al. 2018).
        4. Standardize (zero-mean unit-variance, optional)
        5. Replace NaN/Inf with zero (optional)

    Args:
        signals: ``[T, n_signals]`` array (or 1-D, treated as ``[T, 1]``).
        detrend: subtract a linear trend from each column.
        standardize: rescale each column to zero-mean unit-variance
            (using sample std, ``ddof=1``).
        confounds: optional ``[T, n_conf]`` confounds regressed out via OLS
            (intercept added).
        low_pass: low-pass cutoff in Hz (Butterworth, order 5).
        high_pass: high-pass cutoff in Hz.
        ensure_finite: replace NaN/Inf with zero in the output.
        sampling_freq: sampling rate in Hz (used for filter design).
        runs: optional 1-D label array. If given, each unique label is
            cleaned independently and the segments are concatenated back
            in original order.

    Returns:
        ``np.ndarray`` of shape ``[T, n_signals]``.
    """
    from scipy.signal import butter, detrend as _detrend, filtfilt

    signals = np.asarray(signals, dtype=np.float64)
    one_d = signals.ndim == 1
    if one_d:
        signals = signals[:, None]

    if runs is not None:
        runs = np.asarray(runs)
        if runs.shape[0] != signals.shape[0]:
            raise ValueError(
                f"runs length ({runs.shape[0]}) must match signals "
                f"length ({signals.shape[0]})"
            )
        out = np.empty_like(signals)
        for r in np.unique(runs):
            mask = runs == r
            sub_conf = confounds[mask] if confounds is not None else None
            out[mask] = clean_signal(
                signals[mask],
                detrend=detrend,
                standardize=standardize,
                confounds=sub_conf,
                low_pass=low_pass,
                high_pass=high_pass,
                ensure_finite=ensure_finite,
                sampling_freq=sampling_freq,
                runs=None,
            )
        return out.squeeze(-1) if one_d else out

    if detrend:
        signals = _detrend(signals, axis=0, type="linear")

    # Filter design (used for both signals and confounds, per nilearn).
    filter_b_a = None
    if low_pass is not None or high_pass is not None:
        nyq = sampling_freq / 2.0
        if low_pass is not None and high_pass is not None:
            filter_b_a = butter(
                N=5, Wn=[high_pass / nyq, low_pass / nyq], btype="bandpass"
            )
        elif low_pass is not None:
            filter_b_a = butter(N=5, Wn=low_pass / nyq, btype="lowpass")
        else:
            filter_b_a = butter(N=5, Wn=high_pass / nyq, btype="highpass")

    if filter_b_a is not None:
        signals = filtfilt(*filter_b_a, signals, axis=0)

    if confounds is not None:
        confounds = np.asarray(confounds, dtype=np.float64)
        if confounds.ndim == 1:
            confounds = confounds[:, None]
        if confounds.shape[0] != signals.shape[0]:
            raise ValueError(
                f"confounds length ({confounds.shape[0]}) must match signals "
                f"length ({signals.shape[0]})"
            )
        # Filter confounds with the same Butterworth so the filter and
        # confound-removal operators stay orthogonal (nilearn's behavior).
        if filter_b_a is not None:
            confounds = filtfilt(*filter_b_a, confounds, axis=0)
        # Append intercept and regress out via least squares.
        X = np.hstack([confounds, np.ones((confounds.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(X, signals, rcond=None)
        signals = signals - X @ beta

    if standardize:
        mean = signals.mean(axis=0)
        std = signals.std(axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)  # avoid div-by-zero on constant cols
        signals = (signals - mean) / std

    if ensure_finite:
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

    return signals.squeeze(-1) if one_d else signals


def _normalize_rows(emb):
    """L2-normalize each row of an ``[N, D]`` float array."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    return emb / norms


def _cluster_gallery(normed, threshold):
    """Incremental 'leader' clustering. Each embedding (in order) joins the
    nearest running identity centroid whose cosine similarity is >= threshold,
    otherwise it starts a new identity; the matched centroid is updated as a
    running mean and re-normalized.

    O(N·K) time and O(K·D) memory (K = number of identities) — no N×N matrix,
    so it scales to arbitrarily large inputs. Single-pass, so labels are
    order-dependent (a property shared by any streaming scheme).
    """
    N, D = normed.shape
    labels = np.empty(N, dtype=np.int64)
    cap = 64
    cent = np.zeros((cap, D), dtype=normed.dtype)   # normalized running-mean centroids
    counts = np.zeros(cap, dtype=np.int64)
    K = 0
    for i in range(N):
        e = normed[i]
        if K:
            sims = cent[:K] @ e
            k = int(sims.argmax())
            if sims[k] >= threshold:
                labels[i] = k
                counts[k] += 1
                c = cent[k] + (e - cent[k]) / counts[k]
                n = float(np.linalg.norm(c))
                cent[k] = c / (n if n > 1e-12 else 1e-12)
                continue
        if K == cap:
            cap *= 2
            new_cent = np.zeros((cap, D), dtype=normed.dtype)
            new_cent[:K] = cent[:K]
            new_counts = np.zeros(cap, dtype=np.int64)
            new_counts[:K] = counts[:K]
            cent, counts = new_cent, new_counts
        cent[K] = e
        counts[K] = 1
        labels[i] = K
        K += 1
    return labels


def _cluster_hdbscan(normed, threshold, min_cluster_size):
    """Density-based clustering (HDBSCAN). On L2-normalized vectors squared
    Euclidean distance is monotonic in cosine similarity (||a-b||² = 2-2·cos),
    so we use the Euclidean metric — letting HDBSCAN use a space tree instead
    of a dense cosine matrix — and map the cosine threshold to a cluster-
    selection epsilon. Returns -1 for noise points (low-confidence / outliers),
    surfaced to the caller as ``"Unknown"``.
    """
    import math
    from sklearn.cluster import HDBSCAN

    eps = math.sqrt(max(0.0, 2.0 - 2.0 * threshold))
    clusterer = HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        metric="euclidean",
        cluster_selection_epsilon=eps,
        copy=False,                 # `normed` is a fresh array; no need to copy
    )
    return clusterer.fit_predict(normed).astype(np.int64)


def _cluster_connected_components(normed, threshold, chunk_size):
    """Single-linkage connected components on the thresholded cosine graph.
    Transitive (A~B, B~C ⇒ A,B,C share a label). EXACT (matches pre-0.7
    labels) but materializes an N×N boolean adjacency — O(N²) memory, so it
    OOMs on long videos. Kept for backwards-compatible / exact results; prefer
    'gallery' or 'hdbscan' at scale.
    """
    t = torch.from_numpy(np.ascontiguousarray(normed))
    N = t.shape[0]
    thresholded = torch.zeros((N, N), dtype=torch.bool)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        thresholded[start:end] = (t[start:end] @ t.T) > threshold
    visited = torch.zeros(N, dtype=torch.bool)
    labels = np.full(N, -1, dtype=np.int64)
    nxt = 0
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        labels[i] = nxt
        while stack:
            cur = stack.pop()
            neighbors = (thresholded[cur] & ~visited).nonzero(as_tuple=True)[0].tolist()
            for j in neighbors:
                visited[j] = True
                labels[j] = nxt
                stack.append(j)
        nxt += 1
    return labels


def cluster_identities(face_embeddings, threshold=0.8, method="gallery",
                       min_cluster_size=2, chunk_size: int = 4096):
    """Cluster face identities from their embeddings.

    Args:
        face_embeddings: ``[N, D]`` Fex / numpy array / torch tensor of
            embeddings. Rows with non-finite values (e.g. no-detection frames)
            are labelled ``NaN`` and excluded from clustering.
        threshold: cosine-similarity cutoff for the same person (``gallery`` /
            ``connected``; also mapped to an epsilon for ``hdbscan``).
        method: ``"gallery"`` (default) — single-pass incremental clustering,
            O(K·D) memory, scales to huge inputs; ``"hdbscan"`` — density-based,
            emits ``"Unknown"`` for noise; ``"connected"`` — legacy exact
            single-linkage connected components (O(N²) memory, can OOM).
        min_cluster_size: minimum cluster size for ``hdbscan``.
        chunk_size: matmul block size for ``connected``.

    Returns:
        list of length ``N`` of ``"Person_<k>"`` (k by first appearance),
        ``"Unknown"`` (hdbscan noise), or ``float('nan')`` (non-finite row).
    """
    from feat.data import Fex

    if isinstance(face_embeddings, Fex):
        emb = face_embeddings.astype(float).values
    elif isinstance(face_embeddings, np.ndarray):
        emb = face_embeddings
    elif isinstance(face_embeddings, torch.Tensor):
        emb = face_embeddings.detach().cpu().numpy()
    else:
        emb = np.asarray(face_embeddings)
    emb = np.asarray(emb, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"face_embeddings must be [N, D]; got shape {emb.shape}")
    N = emb.shape[0]
    if N == 0:
        return []

    finite = np.isfinite(emb).all(axis=1)
    out = [float("nan")] * N            # non-finite (no-detection) rows -> NaN
    if not finite.any():
        return out
    normed = _normalize_rows(emb[finite]).astype(np.float32)

    if method == "gallery":
        raw = _cluster_gallery(normed, threshold)
    elif method == "hdbscan":
        raw = _cluster_hdbscan(normed, threshold, min_cluster_size)
    elif method == "connected":
        raw = _cluster_connected_components(normed, threshold, chunk_size)
    else:
        raise ValueError(
            f"unknown method {method!r}; use 'gallery', 'hdbscan', or 'connected'"
        )

    # Renumber to Person_<k> by order of first appearance (stable across
    # methods); HDBSCAN noise (-1) -> 'Unknown'.
    remap = {}
    nxt = 0
    for pos, lab in zip(np.nonzero(finite)[0], raw):
        lab = int(lab)
        if lab == -1:
            out[pos] = "Unknown"
            continue
        if lab not in remap:
            remap[lab] = nxt
            nxt += 1
        out[pos] = f"Person_{remap[lab]}"
    return out
