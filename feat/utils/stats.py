"""
Feat utility and helper functions for performing statistics. 
"""

import numpy as np
import pandas as pd
from scipy.integrate import simps
import torch
from torch.nn.functional import cosine_similarity

__all__ = ["wavelet", "calc_hist_auc", "softmax", "cluster_identities"]


def wavelet(freq, num_cyc=3, sampling_freq=30.0):
    """Create a complex Morlet wavelet.

    Creates a complex Morlet wavelet by windowing a cosine function by a Gaussian. All formulae taken from Cohen, 2014 Chaps 12 + 13

    Args:
        freq: (float) desired frequence of wavelet
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
    return 1.0 / (1 + 10.0 ** -(x))


def cluster_identities(face_embeddings, threshold=0.8):
    """Function to cluster face identities based on cosine similarity of embeddings

    Args:
        face_embeddings (torch.tensor): an observation by embedding torch tensor
        threshold (float): a threshold to determine which embeddings are the same person

    Returns:
        a list of of identities
    """
    from feat.data import Fex

    if isinstance(face_embeddings, Fex):
        face_embeddings = torch.tensor(face_embeddings.astype(float).values)
    elif isinstance(face_embeddings, np.ndarray):
        face_embeddings = torch.tensor(face_embeddings)

    similarity_matrix = cosine_similarity(
        face_embeddings[None, :], face_embeddings[:, None], dim=-1
    )

    thresholded_matrix = similarity_matrix > threshold

    # Clustering
    visited = set()
    clusters = []
    cluster_indices = [-1 for _ in range(face_embeddings.size(0))]  # Initialize list

    for i in range(thresholded_matrix.size(0)):
        if i not in visited:
            # New cluster
            cluster = {i}
            stack = [i]
            visited.add(i)
            current_cluster_idx = len(
                clusters
            )  # This will be the index for the current cluster
            cluster_indices[i] = current_cluster_idx
            while stack:
                current = stack.pop()
                neighbors = (
                    thresholded_matrix[current]
                    & ~torch.tensor(
                        [idx in visited for idx in range(thresholded_matrix.size(0))]
                    )
                ).nonzero(as_tuple=True)[0]
                for neighbor in neighbors:
                    stack.append(neighbor.item())
                    cluster.add(neighbor.item())
                    visited.add(neighbor.item())
                    cluster_indices[
                        neighbor.item()
                    ] = current_cluster_idx  # Update the cluster index for the item
            clusters.append(cluster)
    return [f"Person_{x}" for x in cluster_indices]
