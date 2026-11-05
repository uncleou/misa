"""private_probe_v2: freeze a checkpoint, cache six pre-fusion states, train masked heads.

Reads only the existing train/dev caches, never test.pkl or the original Solver.
The original model, training objective and default configuration are unchanged.
See probe_utils.py for the fixed mask, shared head and train-only standardization.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_utils
from probe_utils import FEATURE_ORDER, GROUPS, VERSION

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_SEED = 336
ARCH_FIELDS = (
    'use_bert', 'use_cmd_sim', 'shared_align_loss', 'atp_variant', 'hidden_size',
    'embedding_size', 'rnncell', 'dropout', 'num_classes', 'variant',
    'private_selection', 'scps_hidden', 'scps_init_gate', 'scps_init_std',
    'gate_lr_mult', 'scps_diagnostics', 'reverse_grad_weight', 'diff_weight',
    'diff_sp_weight', 'diff_pp_weight')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    probe_utils.set_seed(seed)


def write_json(path, value):
    probe_utils.write_json(path, value)


def extract_split(model, loader, device):
    """Cache all six pre-fusion states in exactly the collated label/ID order."""
    model.requires_grad_(False)
    model.eval()
    chunks = {key: [] for key in FEATURE_ORDER}
    labels, sample_ids = [], []
    with torch.no_grad():
        for batch in loader:
            t, v, a, y, lengths, bt, types, mask, ids = batch
            model(t.to(device), v.to(device), a.to(device), lengths,
                  bt.to(device), types.to(device), mask.to(device))
            for key in FEATURE_ORDER:
                chunks[key].append(getattr(model, key).detach().cpu().float().clone())
            labels.append(y.detach().cpu().float().reshape(-1))
            sample_ids.extend(ids)
    if not labels:
        raise ValueError('Cannot probe an empty split')
    result = {key: torch.cat(value) for key, value in chunks.items()}
    result['labels'] = torch.cat(labels)
    result['sample_ids'] = sample_ids
    probe_utils.validate_split(result)
    return result


def _activation_name(value, activation_dict):
    matches = [k for k, cls in activation_dict.items() if value in (k, str(cls))]
    if not matches:
        raise ValueError('Unsupported saved activation: {}'.format(value))
    return matches[0]


def restore_architecture(checkpoint, activation_dict):
    """Prefer config_<name>.json next to model_<name>.std; else fixed CMD defaults."""
    name = checkpoint.name
    candidate = checkpoint.with_name('config_' + name[len('model_'):-len('.std')] + '.json') \
        if name.startswith('model_') and name.endswith('.std') else None
    saved, source = {}, 'defaults'
    if candidate is not None and candidate.is_file():
        saved = json.loads(candidate.read_text(encoding='utf-8'))
        source = str(candidate)
    options = {key: saved[key] for key in ARCH_FIELDS if key in saved}
    options.setdefault('data', saved.get('data', 'mosi'))
    options.setdefault('shared_align_loss', 'cmd')
    options.setdefault('use_bert', True)
    options.setdefault('use_cmd_sim', True)
    options.setdefault('hidden_size', 128)
    options.setdefault('embedding_size', 300)
    options.setdefault('rnncell', 'lstm')
    options.setdefault('dropout', 0.5)
    options.setdefault('model', 'MISA')
    if 'activation' in saved:
        options['activation'] = _activation_name(saved['activation'], activation_dict)
    else:
        options.setdefault('activation', 'relu')
    return options, source


def build_configs(data, options, batch_size, seed, activation_dict):
    from config import get_config
    activation = options.get('activation', 'relu')
    common = {key: value for key, value in options.items()
              if key in ARCH_FIELDS or key == 'activation'}
    configs = {}
    original_argv = sys.argv
    try:
        for split in ('train', 'dev'):
            sys.argv = [original_argv[0]]
            configs[split] = get_config(
                parse=False, data=data, mode=split, name=VERSION, model='MISA',
                batch_size=batch_size, seed=seed, **common)
    finally:
        sys.argv = original_argv
    return configs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=(VERSION, 'cmd_private_probe_v1',), default=VERSION,
                        help='Recorded in the manifest; only the v2 protocol is implemented.')
    parser.add_argument('--data', choices=('mosi', 'mosei'), default='mosei')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Trained state_dict, e.g. checkpoints/model_mosei_cmd_s336.std')
    parser.add_argument('--groups', nargs='+', choices=tuple(GROUPS), default=list(GROUPS),
                        help='Probe groups to run; default is all five.')
    parser.add_argument('--probe_seeds', nargs='+', type=int, default=list(probe_utils.DEFAULT_SEEDS),
                        help='Seeds for the small prediction heads (not the encoder).')
    parser.add_argument('--output_dir', type=Path,
                        help='New output directory; never overwrite an existing run.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--extract_batch_size', type=int, default=64)
    args = parser.parse_args(argv)
    if args.extract_batch_size <= 0:
        parser.error('--extract_batch_size must be positive')
    if not args.probe_seeds:
        parser.error('--probe_seeds must not be empty')
    if len(set(args.groups)) != len(args.groups):
        parser.error('--groups must not repeat a group')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Use python, not torchrun: this diagnostic is single-process')
    return args


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError('Place the trained checkpoint at: {}'.format(checkpoint))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    output = (args.output_dir or PROJECT_DIR / 'results' / VERSION / (
        args.data + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Delayed imports: --help and unit tests do not load a tokenizer or download BERT.
    from config import activation_dict
    from data_loader import get_loader
    from models import MISA

    seed = probe_utils.DEFAULT_SEEDS[0]
    set_seed(seed)
    options, config_source = restore_architecture(checkpoint, activation_dict)
    configs = build_configs(args.data, options, args.extract_batch_size, seed, activation_dict)
    config = configs['train']
    loaders = {split: get_loader(split_config, shuffle=False, diagnostic=True)
               for split, split_config in configs.items()}
    model = MISA(config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    manifest = dict(
        version=VERSION, dataset=args.data, checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_provenance='User-supplied; historical loss cannot be inferred from state_dict',
        architecture_source=config_source,
        model_config={key: getattr(config, key) for key in (
            'model', 'variant', 'use_bert', 'use_cmd_sim', 'shared_align_loss', 'atp_variant',
            'private_selection', 'hidden_size', 'embedding_size', 'visual_size',
            'acoustic_size', 'rnncell', 'dropout', 'num_classes')},
        activation=options.get('activation', 'relu'),
        seeds=list(args.probe_seeds), settings=probe_utils.SETTINGS,
        groups={group: dict(mask=list(GROUPS[group]), hidden_width=probe_utils.HIDDEN_WIDTH)
                for group in args.groups},
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
    standardizer = probe_utils.fit_standardizer(cache['train'])
    manifest.update(status='probing', samples={k: len(v['labels']) for k, v in cache.items()},
                    features_sha256=sha256(output / 'features.pt'),
                    standardization='train_mean_std_fit_on_train_only')
    write_json(output / 'manifest.json', manifest)
    probe_utils.run_probes(cache, output, device, standardizer, groups=args.groups,
                           seeds=tuple(args.probe_seeds))
    manifest['status'] = 'complete'
    write_json(output / 'manifest.json', manifest)
    print('Completed {}: {}'.format(VERSION, output), flush=True)


if __name__ == '__main__':
    main()
