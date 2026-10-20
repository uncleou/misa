"""Evaluate a saved MISA checkpoint and write the test metrics to a file."""
import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_DIR)

from config import get_config
from data_loader import get_loader
from solver import Solver


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='mosi')
    parser.add_argument('--name', required=True,
                        help='Checkpoint name (checkpoints/model_<name>.std).')
    parser.add_argument('--shared_align_loss', choices=('cmd', 'atp'), default='atp')
    parser.add_argument('--output', default=None,
                        help='Output file; defaults to results/<name>_test.txt.')
    return parser.parse_args()


def main():
    args = parse_args()

    config = get_config(
        parse=False,
        mode='test',
        data=args.data,
        name=args.name,
        shared_align_loss=args.shared_align_loss,
        use_bert=True,
        use_cmd_sim=True,
    )

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
        f'dataset: {args.data}\n'
        f'shared_align_loss: {args.shared_align_loss}\n'
        f'eval_loss: {eval_loss:.6f}\n'
        f'accuracy: {accuracy:.6f}\n\n'
    )
    output_path.write_text(summary + metrics.getvalue(), encoding='utf-8')
    print(summary + metrics.getvalue())
    print(f'Results written to {output_path}')


if __name__ == '__main__':
    main()
