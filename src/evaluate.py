"""Evaluate a saved MISA checkpoint and write the test metrics to a file."""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_DIR)

from config import get_config, activation_dict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--name', required=True,
                        help='Checkpoint name (checkpoints/model_<name>.std).')
    parser.add_argument('--shared_align_loss', choices=('cmd', 'atp'), default=None)
    parser.add_argument('--variant', choices=('default', 'misa_pdr', 'misa_pdr_zero', 'misa_scps'), default=None)
    parser.add_argument('--private_selection', choices=('none', 'private_only', 'shared_context'), default=None)
    parser.add_argument('--scps_hidden', type=int, default=None)
    parser.add_argument('--output', default=None,
                        help='Output file; defaults to results/<name>_test.txt.')
    return parser.parse_args()


def build_eval_config(args, project_dir=PROJECT_DIR):
    """Restore architecture from saved metadata; fail on conflicting overrides."""
    path = Path(project_dir) / 'checkpoints' / f'config_{args.name}.json'
    saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    fields = ('data', 'variant', 'private_selection', 'scps_hidden', 'scps_init_gate',
              'use_bert', 'use_cmd_sim', 'shared_align_loss', 'atp_variant',
              'hidden_size', 'embedding_size', 'rnncell', 'dropout', 'num_classes',
              'diff_weight', 'diff_sp_weight', 'diff_pp_weight', 'cu_weight',
              'sent_weight', 'rnc_weight', 'model', 'reverse_grad_weight')
    options = {k: saved[k] for k in fields if k in saved}
    if 'activation' in saved:
        value = saved['activation']
        matches = [k for k, cls in activation_dict.items() if value in (k, str(cls))]
        if not matches:
            raise ValueError(f'Unsupported saved activation: {value}')
        options['activation'] = matches[0]
    for key in ('data', 'variant', 'shared_align_loss', 'private_selection', 'scps_hidden'):
        value = getattr(args, key, None)
        if value is not None:
            if key in options and options[key] != value:
                raise ValueError(f'{key} conflicts with saved checkpoint config')
            options[key] = value
    options.setdefault('data', 'mosi')
    options.setdefault('shared_align_loss', 'cmd')
    options.setdefault('use_bert', True)
    options.setdefault('use_cmd_sim', True)
    options.setdefault('num_classes', 2 if options['data'] == 'ur_funny' else 1)
    options.update(mode='test', name=args.name)
    # Do not let unrelated evaluator CLI flags leak into training configuration.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return get_config(parse=False, **options)
    finally:
        sys.argv = original_argv


def main():
    args = parse_args()
    config = build_eval_config(args)
    from data_loader import get_loader
    from solver import Solver

    test_loader = get_loader(config, shuffle=False)
    solver = Solver(config, config, config, None, None, test_loader, is_train=False)

    with redirect_stdout(io.StringIO()):
        solver.build()

    metrics = io.StringIO()
    with redirect_stdout(metrics):
        eval_loss, accuracy = solver.eval(mode='test', to_print=True)

    output_path = (Path(args.output) if args.output
                   else PROJECT_DIR / 'results' / f'{args.name}_test.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        f'checkpoint: checkpoints/model_{args.name}.std\n'
        f'dataset: {config.data}\n'
        f'variant: {config.variant}\n'
        f'private_selection: {config.private_selection}\n'
        f'shared_align_loss: {config.shared_align_loss}\n'
        f'eval_loss: {eval_loss:.6f}\n'
        f'accuracy: {accuracy:.6f}\n\n'
    )
    output_path.write_text(summary + metrics.getvalue(), encoding='utf-8')
    print(summary + metrics.getvalue())
    print(f'Results written to {output_path}')


if __name__ == '__main__':
    main()
