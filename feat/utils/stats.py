"""
Feat utility and helper functions for performing statistics.
"""

import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
import torch
from torch.nn.functional import cosine_similarity

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


def cluster_identities(face_embeddings, threshold=0.8, chunk_size: int = 4096):
    """Cluster face identities based on cosine similarity of embeddings.

    Treats the thresholded cosine-similarity matrix as the adjacency matrix
    of an undirected graph, and labels each connected component as one
    identity. Two embeddings need not be directly above-threshold to share
    a label - any chain of above-threshold edges puts them in the same
    cluster (transitivity).

    Args:
        face_embeddings: ``[N, D]`` tensor (or Fex / numpy array) of
            embeddings.
        threshold: cosine-similarity cutoff above which two embeddings are
            considered the same person.
        chunk_size: number of rows per matmul block when ``N`` is large.
            Memory peak per block is O(chunk_size × N) bytes. Default 4096.

    Returns:
        list of length ``N`` with strings ``"Person_<k>"``, where the
        cluster ids ``k`` follow the order in which clusters first appear.
    """
    from feat.data import Fex

    if isinstance(face_embeddings, Fex):
        face_embeddings = torch.tensor(face_embeddings.astype(float).values)
    elif isinstance(face_embeddings, np.ndarray):
        face_embeddings = torch.tensor(face_embeddings)

    # Build the boolean adjacency matrix via L2-normalized matmul, in
    # row-chunks. Previous implementation used broadcasted cosine_similarity
    # which materialized a (N, N, D) tensor — at N=57k that's ~14 TB and OOMs.
    # The matmul form is O(N×N) memory (~13 GB at N=57k, fp32) plus one
    # bool matrix of the same shape (~3 GB). The chunked write avoids two
    # full fp32 copies (raw sim then threshold) sitting in RAM together.
    if face_embeddings.dim() != 2:
        raise ValueError(
            f"face_embeddings must be [N, D]; got shape {tuple(face_embeddings.shape)}"
        )
    N = face_embeddings.size(0)
    if N == 0:
        return []
    if face_embeddings.dtype != torch.float32:
        face_embeddings = face_embeddings.float()
    norms = face_embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normed = face_embeddings / norms
    # Keep the dense bool adjacency on CPU (per-row chunked write).
    normed_cpu = normed.cpu()
    thresholded_matrix = torch.zeros((N, N), dtype=torch.bool)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        sim_chunk = normed_cpu[start:end] @ normed_cpu.T  # (chunk, N)
        thresholded_matrix[start:end] = sim_chunk > threshold
        del sim_chunk

    # Track visited as a bool tensor instead of a Python set. The previous
    # implementation rebuilt `~torch.tensor([idx in visited for idx ...])`
    # on every BFS pop, costing O(N) per pop and overall O(N^3) worst case
    # plus N+ tensor allocations per detect() call.
    visited = torch.zeros(N, dtype=torch.bool, device=thresholded_matrix.device)
    cluster_indices = [-1] * N
    next_cluster_idx = 0

    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        cluster_indices[i] = next_cluster_idx
        while stack:
            current = stack.pop()
            # Neighbors above threshold AND not yet visited.
            mask = thresholded_matrix[current] & ~visited
            neighbors = mask.nonzero(as_tuple=True)[0].tolist()
            for neighbor in neighbors:
                stack.append(neighbor)
                visited[neighbor] = True
                cluster_indices[neighbor] = next_cluster_idx
        next_cluster_idx += 1
    return [f"Person_{x}" for x in cluster_indices]
