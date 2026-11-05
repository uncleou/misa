"""Frozen-representation private probe: fixed masks, one shared head, train-only scaling.

The five groups share one input layout ``[s_t, s_a, s_v, p_t, p_a, p_v]`` of width
``6 * d`` and a fixed binary mask per group. Masked dimensions are zeroed in both
training and validation, so every group sees the same input definition throughout.
Every group uses the same ``Linear(6d, 64) -> ReLU -> Linear(64, 1)`` head, but
parameters on masked dimensions receive no gradient, so effective capacity differs;
this is not an exact equal-capacity control.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

VERSION = 'private_probe_v2'

FEATURE_ORDER = ('utt_shared_t', 'utt_shared_a', 'utt_shared_v',
                 'utt_private_t', 'utt_private_a', 'utt_private_v')
SHARED = FEATURE_ORDER[:3]
PRIVATE = FEATURE_ORDER[3:]

# Ordered to match FEATURE_ORDER: [s_t, s_a, s_v, p_t, p_a, p_v].
GROUPS = {
    'shared': (1, 1, 1, 0, 0, 0),
    'shared_pt': (1, 1, 1, 1, 0, 0),
    'shared_pa': (1, 1, 1, 0, 1, 0),
    'shared_pv': (1, 1, 1, 0, 0, 1),
    'shared_all': (1, 1, 1, 1, 1, 1),
}
DEFAULT_SEEDS = (11, 22, 33)
HIDDEN_WIDTH = 64
SETTINGS = dict(optimizer='Adam', learning_rate=1e-3, batch_size=64,
                max_epochs=100, patience=10, weight_decay=0.0,
                selection='dev_mse', normalization='train_standardize', dropout=0.0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                          encoding='utf-8')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def feature_dim(split):
    return split[FEATURE_ORDER[0]].shape[1]


def validate_split(split, hidden_size=None):
    count = split['labels'].numel()
    if count == 0 or split['labels'].shape != (count,):
        raise ValueError('Expected nonempty scalar regression labels')
    if len(split['sample_ids']) != count or len(set(split['sample_ids'])) != count:
        raise ValueError('Missing or duplicate sample IDs')
    dim = feature_dim(split)
    if hidden_size is not None and dim != hidden_size:
        raise ValueError('Feature width {} != hidden_size {}'.format(dim, hidden_size))
    for key in FEATURE_ORDER:
        if split[key].shape != (count, dim):
            raise ValueError('{} requires [N, {}] features: {}'.format(VERSION, dim, key))
    for key in FEATURE_ORDER + ('labels',):
        value = split[key]
        if value.requires_grad or not torch.isfinite(value).all():
            raise ValueError('Features/labels must be detached and finite: ' + key)


def stack_features(split):
    """Full unmasked [N, 6d] input in FEATURE_ORDER layout."""
    return torch.cat([split[key] for key in FEATURE_ORDER], dim=1)


def fit_standardizer(train_split):
    """Mean/std fitted on training features only; zero-variance dims fall back to 1."""
    features = stack_features(train_split).double()
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean.float(), std.float()


def group_mask(group, dim, dtype=torch.float32):
    pattern = GROUPS[group]
    mask = torch.tensor(pattern, dtype=dtype)
    return mask.repeat_interleave(dim)


def group_input(split, group, standardizer):
    """Standardize with training statistics, then apply the fixed group mask."""
    mean, std = standardizer
    features = (stack_features(split) - mean) / std
    return features * group_mask(group, feature_dim(split), features.dtype)


def make_head(dim, hidden=HIDDEN_WIDTH):
    input_dim = len(FEATURE_ORDER) * dim
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def regression_metrics(truth, prediction):
    """All-sample regression; explicit zero-label conventions for binary metrics."""
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape:
        raise ValueError('Expected matching nonempty prediction and label vectors')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions or labels')
    corr = None
    if len(truth) > 1 and np.std(truth) > 0 and np.std(prediction) > 0:
        corr = float(np.corrcoef(truth, prediction)[0, 1])
    truth_a7 = np.round(np.clip(truth, a_min=-3., a_max=3.))
    pred_a7 = np.round(np.clip(prediction, a_min=-3., a_max=3.))
    metrics = dict(mse=float(np.mean((truth - prediction) ** 2)),
                   mae=float(np.mean(np.abs(truth - prediction))), corr=corr,
                   acc3=float(np.mean(pred_a7 == truth_a7)))
    for suffix, keep, positive in (
            ('nonzero', truth != 0, lambda x: x > 0),
            ('with_zero', np.ones(len(truth), dtype=bool), lambda x: x >= 0)):
        actual, predicted = positive(truth[keep]), positive(prediction[keep])
        metrics['acc2_' + suffix] = float(np.mean(actual == predicted)) if keep.any() else None
        metrics['f1_' + suffix] = float(f1_score(actual, predicted, average='weighted',
                                                zero_division=0)) if keep.any() else None
    return metrics


def predict(head, features, device, batch_size=1024):
    head.eval()
    with torch.no_grad():
        return torch.cat([head(x.to(device)).reshape(-1).cpu()
                          for x in features.split(batch_size)])


def train_head(train, dev, group, seed, device, standardizer, settings=None):
    settings = SETTINGS if settings is None else settings
    set_seed(seed)
    dim = feature_dim(train)
    head = make_head(dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    x_train = group_input(train, group, standardizer)
    x_dev = group_input(dev, group, standardizer)
    # Independent RNG: minibatch order does not depend on head initialization size.
    order_rng = torch.Generator().manual_seed(seed)
    best_mse, stale, history = float('inf'), 0, []
    best_state, best_epoch = None, None
    for epoch in range(1, settings['max_epochs'] + 1):
        head.train()
        order = torch.randperm(len(x_train), generator=order_rng)
        squared_error = 0.0
        for indices in order.split(settings['batch_size']):
            target = train['labels'][indices].to(device)
            optimizer.zero_grad()
            prediction = head(x_train[indices].to(device)).reshape(-1)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite probe loss')
            loss.backward()
            optimizer.step()
            squared_error += loss.item() * len(indices)
        dev_prediction = predict(head, x_dev, device)
        dev_mse = regression_metrics(dev['labels'].numpy(), dev_prediction.numpy())['mse']
        history.append(dict(epoch=epoch, train_mse=squared_error / len(x_train), dev_mse=dev_mse))
        if dev_mse < best_mse:
            best_mse, best_epoch, stale = dev_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= settings['patience']:
                break
    head.load_state_dict(best_state)
    prediction = predict(head, x_dev, device)
    metrics = regression_metrics(dev['labels'].numpy(), prediction.numpy())
    row = dict(group=group, seed=seed, input_dim=x_train.shape[1], hidden_dim=HIDDEN_WIDTH,
               parameters=sum(p.numel() for p in head.parameters()), best_epoch=best_epoch,
               epochs_run=len(history), **metrics)
    return row, best_state, prediction, history


def summarize(rows, groups=None):
    present = [g for g in (groups or GROUPS) if any(r['group'] == g for r in rows)]
    baseline_mae = {row['seed']: row['mae'] for row in rows if row['group'] == 'shared'}
    baseline_corr = {row['seed']: row['corr'] for row in rows if row['group'] == 'shared'}
    for row in rows:
        base_mae = baseline_mae.get(row['seed'])
        row['delta_mae_vs_shared'] = None if base_mae is None else base_mae - row['mae']
        base_corr = baseline_corr.get(row['seed'])
        row['delta_corr_vs_shared'] = (
            None if row['corr'] is None or base_corr is None else row['corr'] - base_corr)
    metric_names = ('mse', 'mae', 'corr', 'acc3', 'acc2_nonzero', 'f1_nonzero',
                    'acc2_with_zero', 'f1_with_zero', 'delta_mae_vs_shared', 'delta_corr_vs_shared')
    summary = []
    for group in present:
        selected = [row for row in rows if row['group'] == group]
        result = dict(group=group, parameters=selected[0]['parameters'], seeds=len(selected))
        for metric in metric_names:
            values = [row[metric] for row in selected if row[metric] is not None]
            result[metric + '_mean'] = float(np.mean(values)) if values else None
            result[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result[metric + '_n'] = len(values)
        summary.append(result)
    return summary


def run_probes(cache, output, device, standardizer, groups=tuple(GROUPS), seeds=DEFAULT_SEEDS,
               settings=None):
    output = Path(output)
    for split in ('train', 'dev'):
        validate_split(cache[split])
    rows = []
    for group in groups:
        for seed in seeds:
            row, state, prediction, history = train_head(
                cache['train'], cache['dev'], group, seed, device, standardizer, settings)
            stem = '{}_s{}'.format(group, seed)
            torch.save(dict(version=VERSION, group=group, seed=seed,
                            state_dict=state, metrics=row), output / (stem + '.pt'))
            write_json(output / (stem + '_history.json'), history)
            write_csv(output / (stem + '_dev_predictions.csv'), [
                dict(sample_id=sample_id, label=float(label), prediction=float(pred))
                for sample_id, label, pred in zip(cache['dev']['sample_ids'],
                                                 cache['dev']['labels'], prediction)])
            rows.append(row)
            print('{} seed={} epoch={} dev_mae={:.6f}'.format(
                group, seed, row['best_epoch'], row['mae']), flush=True)
    summary = summarize(rows, groups)
    write_csv(output / 'per_seed.csv', rows)
    write_csv(output / 'summary.csv', summary)
    write_json(output / 'summary.json',
               dict(version=VERSION, per_seed=rows, summary=summary,
                    groups=list(groups), seeds=list(seeds)))
    return rows, summary
