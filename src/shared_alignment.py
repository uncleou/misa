"""Sample-centroid losses, independent of the model's fusion states."""

import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


def shared_centroids(shared_t, shared_a, shared_v, normalize_centroid=True):
    """Average unit modality vectors; only legacy CU renormalizes the mean."""
    normalized = [F.normalize(h, p=2, dim=-1, eps=1e-8)
                  for h in (shared_t, shared_a, shared_v)]
    centroids = sum(normalized) / 3.0
    if normalize_centroid:
        centroids = F.normalize(centroids, p=2, dim=-1, eps=1e-8)
    return centroids


def gather_centroids(centroids):
    """Gather the global batch with autograd, including uneven local batches.

    Every rank evaluates the same global centroid loss. The differentiable gather sums
    backward contributions across ranks; the solver's subsequent parameter
    gradient average cancels that factor. Do NOT divide the loss by world_size.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return centroids

    from torch.distributed.nn.functional import all_gather

    size = torch.tensor([centroids.size(0)], device=centroids.device, dtype=torch.long)
    sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, size)
    sizes = [int(value.item()) for value in sizes]
    padded = F.pad(centroids, (0, 0, 0, max(sizes) - centroids.size(0)))
    gathered = all_gather(padded.contiguous())
    return torch.cat([part[:length] for part, length in zip(gathered, sizes)], dim=0)


def gather_centroids_and_labels(centroids, labels):
    """Gather paired rows in rank order; labels are detached supervision only.

    Packing the label column with the centroid guarantees the same ordering and
    padding removal, even for uneven local batches. Only centroid columns have
    an autograd path back to the encoder.
    """
    if labels.numel() != centroids.size(0):
        raise ValueError('Expected one scalar sentiment label per centroid')
    labels = labels.detach().to(centroids).reshape(-1, 1)
    gathered = gather_centroids(torch.cat((centroids, labels), dim=1))
    return gathered[:, :-1], gathered[:, -1].detach()


def sentiment_centroid_loss(centroids, labels, sigma=1.0, tau=1.0):
    """Mean KL(q || p) over rows, excluding self from BOTH softmaxes.

    q_ij ~ exp(-abs(y_i-y_j)/sigma)
    p_ij ~ exp(-squared_distance(c_i,c_j)/tau)
    Centroids are the means of unit modality vectors, NOT renormalized.
    """
    for name, value in (('sigma', sigma), ('tau', tau)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    count = centroids.size(0)
    if labels.numel() != count:
        raise ValueError('Expected one scalar sentiment label per centroid')
    if count < 3:
        # B=2 has one neighbor per row, so both distributions equal [1].
        return centroids.sum() * 0.0

    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=centroids.device)
    distances = (centroids[:, None, :] - centroids[None, :, :]).square().sum(dim=-1)
    logits_p = (-distances / tau)[off_diagonal].reshape(count, count - 1)
    log_p = F.log_softmax(logits_p, dim=-1)
    with torch.no_grad():
        labels = labels.detach().to(centroids).reshape(-1)
        gaps = (labels[:, None] - labels[None, :]).abs()
        logits_q = (-gaps / sigma)[off_diagonal].reshape(count, count - 1)
        log_q = F.log_softmax(logits_q, dim=-1)
        q = log_q.exp()
    # Removing the diagonal before softmax avoids 0 * (-inf - -inf).
    return (q * (log_q - log_p)).sum(dim=-1).mean()


def rnc_centroid_loss(centroids, labels, tau=2.0, distance='l2'):
    """Single-view Rank-N-Contrast on raw means of unit modality vectors.

    For each ordered pair (i,j), the denominator contains k != i with
    |y_i-y_k| >= |y_i-y_j|, including j and ALL ties. Average over N*(N-1).
    This adapts the ranking objective, not the original two-view training scheme:
    https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py

    Sorting label gaps and gathering tie-aware suffix log-sums avoids an N^3
    comparison mask. l2 follows the official metric; l2sq is a controlled variant.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('rnc_tau must be finite and positive')
    if distance not in ('l2', 'l2sq'):
        raise ValueError("rnc_distance must be 'l2' or 'l2sq'")
    if centroids.ndim != 2:
        raise ValueError('Expected centroids of shape [N, D]')
    count = centroids.size(0)
    if labels.numel() != count:
        raise ValueError('Expected one scalar sentiment label per centroid')
    if count < 3:
        return centroids.sum() * 0.0

    # Direct Euclidean kernel gives finite gradients even for coincident points.
    distances = torch.cdist(centroids, centroids, p=2,
                           compute_mode='donot_use_mm_for_euclid_dist')
    if distance == 'l2sq':
        distances = distances.square()
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=centroids.device)
    logits = (-distances / tau)[off_diagonal].reshape(count, count - 1)
    with torch.no_grad():
        targets = labels.detach().to(centroids).reshape(-1)
        gaps = (targets[:, None] - targets[None, :]).abs()
        gaps = gaps[off_diagonal].reshape(count, count - 1)
        sorted_gaps, order = gaps.sort(dim=-1)
        # Every tied candidate must start at the first member of its tie group.
        starts = torch.searchsorted(sorted_gaps.contiguous(),
                                    sorted_gaps.contiguous(), right=False)
    sorted_logits = logits.gather(1, order)
    log_suffix = torch.logcumsumexp(sorted_logits.flip(1), dim=1).flip(1)
    log_denominators = log_suffix.gather(1, starts)
    return (log_denominators - sorted_logits).mean()


@torch.no_grad()
def sentiment_centroid_stats(centroids, labels):
    """Nearest-neighbor label gap, raw centroid variance, and mean norm."""
    count = centroids.size(0)
    if count < 2:
        return 0.0, 0.0, centroids.norm(dim=-1).mean().item() if count else 0.0
    distances = (centroids[:, None, :] - centroids[None, :, :]).square().sum(dim=-1)
    distances.fill_diagonal_(float('inf'))
    nearest = distances.argmin(dim=-1)
    labels = labels.detach().to(centroids).reshape(-1)
    return (
        (labels - labels[nearest]).abs().mean().item(),
        centroids.var(dim=0, unbiased=False).mean().item(),
        centroids.norm(dim=-1).mean().item(),
    )


def centroid_uniformity(centroids, tau=2.0):
    """Log mean exp(-tau * squared distance), over unordered non-self pairs."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('tau must be finite and positive')
    count = centroids.size(0)
    if count < 2:
        # No pairs: finite zero, still connected to autograd.
        return centroids.sum() * 0.0
    pairs = torch.triu_indices(count, count, offset=1, device=centroids.device)
    distances = (centroids[pairs[0]] - centroids[pairs[1]]).square().sum(dim=-1)
    return torch.logsumexp(-tau * distances, dim=0) - math.log(distances.numel())


@torch.no_grad()
def centroid_stats(centroids):
    """Mean off-diagonal cosine and mean per-coordinate batch variance."""
    count = centroids.size(0)
    if count < 2:
        return 0.0, 0.0
    cosines = centroids @ centroids.t()
    mean_cosine = (cosines.sum() - cosines.diagonal().sum()) / (count * (count - 1))
    return mean_cosine.item(), centroids.var(dim=0, unbiased=False).mean().item()
