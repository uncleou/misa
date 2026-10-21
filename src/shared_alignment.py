"""Sample-centroid uniformity, independent of the model's fusion states."""

import math

import torch
import torch.distributed as dist
from torch.nn import functional as F


def shared_centroids(shared_t, shared_a, shared_v):
    """Normalize each modality, average equally, then normalize each centroid."""
    normalized = [F.normalize(h, p=2, dim=-1, eps=1e-8)
                  for h in (shared_t, shared_a, shared_v)]
    return F.normalize(sum(normalized) / 3.0, p=2, dim=-1, eps=1e-8)


def gather_centroids(centroids):
    """Gather the global batch with autograd, including uneven local batches.

    Every rank evaluates the same global CU. The differentiable gather sums
    backward contributions across ranks; the solver's subsequent parameter
    gradient average cancels that factor. Do NOT divide CU by world_size.
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
