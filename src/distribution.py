"""Histogram-loss (HL) helpers for the optional distribution prediction head.

An HL head emits one logit per ordered score bin instead of a single scalar.
Targets are Gaussian soft labels over bin edges (HL-Gauss), trained with
cross-entropy; the continuous prediction is the softmax expectation over bin
centres. Everything here is stateless and driven by the config fields
``hl_bins``/``hl_min``/``hl_max``/``hl_sigma`` so the scalar route is untouched.
"""
import math

import torch
import torch.nn.functional as F


def validate_head_config(config):
    bins = int(config.hl_bins)
    if bins < 2:
        raise ValueError('hl_bins must be at least 2')
    if not (math.isfinite(config.hl_min) and math.isfinite(config.hl_max)):
        raise ValueError('hl_min/hl_max must be finite')
    if not config.hl_min < config.hl_max:
        raise ValueError('hl_min must be strictly below hl_max')
    for name in ('hl_sigma', 'hl_reg_weight'):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    if config.hl_sigma <= 0:
        raise ValueError('hl_sigma must be strictly positive')


def bin_centers(config, device=None, dtype=torch.float32):
    """K evenly spaced bin centres covering [hl_min, hl_max] inclusive."""
    return torch.linspace(config.hl_min, config.hl_max, int(config.hl_bins),
                          device=device, dtype=dtype)


def bin_edges(config, device=None, dtype=torch.float32):
    """K+1 boundaries at bin midpoints, extended by half a step at each end."""
    centers = bin_centers(config, device=device, dtype=dtype)
    step = (config.hl_max - config.hl_min) / (int(config.hl_bins) - 1)
    return torch.cat([
        (centers[:1] - step / 2),
        (centers[:-1] + centers[1:]) / 2,
        (centers[-1:] + step / 2),
    ])


def soft_targets(labels, config):
    """Gaussian CDF mass per bin, renormalized so each row sums to one."""
    labels = labels.reshape(-1)
    device, dtype = labels.device, labels.dtype
    edges = bin_edges(config, device=device, dtype=dtype)
    sigma = labels.new_tensor(float(config.hl_sigma))
    z = (edges.unsqueeze(0) - labels.unsqueeze(1)) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    q = cdf[:, 1:] - cdf[:, :-1]
    q = q.clamp_min(0.0)
    return q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def expectation(logits, config):
    """Continuous prediction as the softmax-weighted bin-centre mean."""
    probs = F.softmax(logits, dim=-1)
    centers = bin_centers(config, device=logits.device, dtype=logits.dtype)
    return (probs * centers).sum(dim=-1)


def cross_entropy(logits, labels, config):
    target = soft_targets(labels, config).to(logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def hl_task_loss(logits, labels, config):
    """Distribution cross-entropy plus optional L1 on the expected value."""
    labels = labels.reshape(-1)
    loss = cross_entropy(logits, labels, config)
    if config.hl_reg_weight > 0:
        loss = loss + config.hl_reg_weight * (expectation(logits, config) - labels).abs().mean()
    return loss


def outside_range(labels, config):
    """Count and extreme values of labels outside the supported score range."""
    labels = labels.reshape(-1)
    mask = (labels < config.hl_min) | (labels > config.hl_max)
    count = int(mask.sum().item())
    if count == 0:
        return 0, None, None
    return count, float(labels.min().item()), float(labels.max().item())
