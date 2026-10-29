"""cmd_private_probe_v1: frozen CMD features, five matched-budget train/dev probes."""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
VERSION = 'cmd_private_probe_v1'
SEEDS = (336, 337, 338, 339, 340)
SETTINGS = dict(optimizer='Adam', learning_rate=1e-3, batch_size=64,
                max_epochs=100, patience=10, weight_decay=0.0,
                selection='dev_mse', normalization='none', dropout=0.0)
SHARED = ('utt_shared_t', 'utt_shared_a', 'utt_shared_v')
PRIVATE = ('utt_private_t', 'utt_private_a', 'utt_private_v')
FEATURES = SHARED + PRIVATE
GROUPS = {'S': SHARED, 'S_T': SHARED + PRIVATE[:1],
          'S_A': SHARED + PRIVATE[1:2], 'S_V': SHARED + PRIVATE[2:],
          'S_T_A_V': FEATURES}
WIDTHS = {'S': 128, 'S_T': 96, 'S_A': 96, 'S_V': 96, 'S_T_A_V': 64}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_split(model, loader, device):
    """Cache all six pre-fusion states in exactly the collated label/ID order."""
    model.requires_grad_(False)
    model.eval()
    chunks = {key: [] for key in FEATURES}
    labels, sample_ids = [], []
    with torch.no_grad():
        for batch in loader:
            t, v, a, y, lengths, bt, types, mask, ids = batch
            model(t.to(device), v.to(device), a.to(device), lengths,
                  bt.to(device), types.to(device), mask.to(device))
            for key in FEATURES:
                chunks[key].append(getattr(model, key).detach().cpu().float().clone())
            labels.append(y.detach().cpu().float().reshape(-1))
            sample_ids.extend(ids)
    if not labels:
        raise ValueError('Cannot probe an empty split')
    result = {key: torch.cat(value) for key, value in chunks.items()}
    result['labels'] = torch.cat(labels)
    result['sample_ids'] = sample_ids
    validate_split(result)
    return result


def validate_split(split):
    count = split['labels'].numel()
    if count == 0 or split['labels'].shape != (count,):
        raise ValueError('Expected nonempty scalar regression labels')
    if len(split['sample_ids']) != count or len(set(split['sample_ids'])) != count:
        raise ValueError('Missing or duplicate sample IDs')
    for key in FEATURES:
        if split[key].shape != (count, 128):
            raise ValueError('{} requires [N, 128] features: {}'.format(VERSION, key))
    for key in FEATURES + ('labels',):
        if split[key].requires_grad or not torch.isfinite(split[key]).all():
            raise ValueError('Features/labels must be detached and finite: ' + key)


def group_input(split, group):
    return torch.cat([split[key] for key in GROUPS[group]], dim=1)


def make_head(group):
    return nn.Sequential(nn.Linear(128 * len(GROUPS[group]), WIDTHS[group]),
                         nn.ReLU(), nn.Linear(WIDTHS[group], 1))


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
    metrics = dict(mse=float(np.mean((truth - prediction) ** 2)),
                   mae=float(np.mean(np.abs(truth - prediction))), corr=corr)
    for suffix, keep, positive in (
            ('nonzero', truth != 0, lambda x: x > 0),
            ('with_zero', np.ones(len(truth), dtype=bool), lambda x: x >= 0)):
        actual, predicted = positive(truth[keep]), positive(prediction[keep])
        metrics['acc2_' + suffix] = float(accuracy_score(actual, predicted)) if keep.any() else None
        metrics['f1_' + suffix] = float(f1_score(actual, predicted, average='weighted',
                                               zero_division=0)) if keep.any() else None
    return metrics


def predict(head, features, device, batch_size=1024):
    head.eval()
    with torch.no_grad():
        return torch.cat([head(x.to(device)).reshape(-1).cpu()
                          for x in features.split(batch_size)])


def train_head(train, dev, group, seed, device, settings=None):
    settings = SETTINGS if settings is None else settings
    set_seed(seed)
    head = make_head(group).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    x_train, x_dev = group_input(train, group), group_input(dev, group)
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
    row = dict(group=group, seed=seed, input_dim=x_train.shape[1], hidden_dim=WIDTHS[group],
               parameters=sum(p.numel() for p in head.parameters()), best_epoch=best_epoch,
               epochs_run=len(history), **metrics)
    return row, best_state, prediction, history


def summarize(rows):
    baseline = {row['seed']: row['mae'] for row in rows if row['group'] == 'S'}
    metric_names = ('mse', 'mae', 'corr', 'acc2_nonzero', 'f1_nonzero',
                    'acc2_with_zero', 'f1_with_zero', 'delta_mae_vs_S')
    summary = []
    for row in rows:
        row['delta_mae_vs_S'] = baseline[row['seed']] - row['mae']
    for group in GROUPS:
        selected = [row for row in rows if row['group'] == group]
        result = dict(group=group, parameters=selected[0]['parameters'], seeds=len(selected))
        for metric in metric_names:
            values = [row[metric] for row in selected if row[metric] is not None]
            result[metric + '_mean'] = float(np.mean(values)) if values else None
            result[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result[metric + '_n'] = len(values)
        summary.append(result)
    return summary


def run_probes(cache, output, device, seeds=SEEDS, settings=None):
    for split in ('train', 'dev'):
        validate_split(cache[split])
    rows = []
    for group in GROUPS:
        for seed in seeds:
            row, state, prediction, history = train_head(
                cache['train'], cache['dev'], group, seed, device, settings)
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
    summary = summarize(rows)
    write_csv(output / 'per_seed.csv', rows)
    write_csv(output / 'summary.csv', summary)
    write_json(output / 'summary.json', dict(version=VERSION, per_seed=rows, summary=summary))
    return rows, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=(VERSION,), default=VERSION)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Original CMD state_dict, e.g. checkpoints/model_mosi_cmd_s336.std')
    parser.add_argument('--data', choices=('mosi', 'mosei'), default='mosi')
    parser.add_argument('--output', type=Path, help='New output directory; never overwrite an existing run')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--extract_batch_size', type=int, default=64)
    args = parser.parse_args(argv)
    if args.extract_batch_size <= 0:
        parser.error('--extract_batch_size must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Use python, not torchrun: this diagnostic is single-process')
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Place the trained CMD checkpoint at: {}'.format(checkpoint))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    output = args.output or PROJECT_DIR / 'results' / VERSION / (
        args.data + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Delayed imports: --help and unit tests do not load a tokenizer or download BERT.
    from config import get_config
    from data_loader import get_loader
    from models import MISA

    set_seed(SEEDS[0])
    configs = {split: get_config(
        parse=False, data=args.data, mode=split, name=VERSION, model='MISA',
        use_bert=True, use_cmd_sim=True, shared_align_loss='cmd',
        cu_weight=0.0, sent_weight=0.0, rnc_weight=0.0,
        hidden_size=128, embedding_size=300, rnncell='lstm', activation='relu',
        dropout=0.5, num_classes=1, seed=SEEDS[0], batch_size=args.extract_batch_size)
        for split in ('train', 'dev')}
    loaders = {split: get_loader(config, shuffle=False, diagnostic=True)
               for split, config in configs.items()}
    model = MISA(configs['train'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    config = configs['train']
    manifest = dict(
        version=VERSION, dataset=args.data, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_provenance='User-supplied CMD; historical loss cannot be inferred from state_dict',
        model_config={key: getattr(config, key) for key in (
            'model', 'use_bert', 'use_cmd_sim', 'shared_align_loss', 'hidden_size',
            'embedding_size', 'visual_size', 'acoustic_size', 'rnncell', 'dropout',
            'num_classes', 'cu_weight', 'sent_weight', 'rnc_weight')},
        activation='relu', seeds=list(SEEDS), settings=SETTINGS,
        groups={key: dict(features=list(value), hidden_width=WIDTHS[key])
                for key, value in GROUPS.items()},
        extract_batch_size=args.extract_batch_size, device=str(device),
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        data_sha256={name: sha256(config.dataset_dir / name)
                     for name in ('train.pkl', 'dev.pkl', 'embedding_and_mapping.pt')},
        status='extracting')
    write_json(output / 'manifest.json', manifest)
    cache = {}
    for split, loader in loaders.items():
        print('Extracting {} frozen features...'.format(split), flush=True)
        cache[split] = extract_split(model, loader, device)
    torch.save(dict(metadata=manifest, **cache), output / 'features.pt')
    del model, state, loaders
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    manifest.update(status='probing', samples={k: len(v['labels']) for k, v in cache.items()},
                    features_sha256=sha256(output / 'features.pt'))
    write_json(output / 'manifest.json', manifest)
    run_probes(cache, output, device)
    manifest['status'] = 'complete'
    write_json(output / 'manifest.json', manifest)
    print('Completed {}: {}'.format(VERSION, output), flush=True)


if __name__ == '__main__':
    main()
