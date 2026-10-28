import os
import argparse
import math
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn

# path to a pretrained word embedding file
word_emb_path = '/home/devamanyu/glove.840B.300d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)

        self.shared_align_loss = getattr(self, 'shared_align_loss', 'cmd')
        if self.shared_align_loss not in ('cmd', 'atp'):
            raise ValueError("shared_align_loss must be 'cmd' or 'atp'")
        if self.shared_align_loss == 'atp' and not self.use_cmd_sim:
            raise ValueError(
                "ATP requires --use_cmd_sim true; false enables domain-adversarial alignment."
            )

        self.cu_weight = getattr(self, 'cu_weight', 0.0)
        self.cu_tau = getattr(self, 'cu_tau', 2.0)
        if not math.isfinite(self.cu_weight) or self.cu_weight < 0:
            raise ValueError('cu_weight must be finite and non-negative')
        if not math.isfinite(self.cu_tau) or self.cu_tau <= 0:
            raise ValueError('cu_tau must be finite and positive')

        self.sent_weight = getattr(self, 'sent_weight', 0.0)
        self.sent_sigma = getattr(self, 'sent_sigma', 1.0)
        self.sent_tau = getattr(self, 'sent_tau', 1.0)
        if not math.isfinite(self.sent_weight) or self.sent_weight < 0:
            raise ValueError('sent_weight must be finite and non-negative')
        for name in ('sent_sigma', 'sent_tau'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.rnc_weight = getattr(self, 'rnc_weight', 0.0)
        self.rnc_tau = getattr(self, 'rnc_tau', 2.0)
        self.rnc_distance = getattr(self, 'rnc_distance', 'l2')
        if not math.isfinite(self.rnc_weight) or self.rnc_weight < 0:
            raise ValueError('rnc_weight must be finite and non-negative')
        if not math.isfinite(self.rnc_tau) or self.rnc_tau <= 0:
            raise ValueError('rnc_tau must be finite and positive')
        if self.rnc_distance not in ('l2', 'l2sq'):
            raise ValueError("rnc_distance must be 'l2' or 'l2sq'")
        if sum(w > 0 for w in (self.cu_weight, self.sent_weight, self.rnc_weight)) > 1:
            raise ValueError('Choose only one of CU, sentiment centroid KL, or RNC; set other weights to 0')
        if self.sent_weight > 0 and self.data.lower() not in ('mosi', 'mosei'):
            raise ValueError('Sentiment centroid loss requires continuous MOSI/MOSEI labels')
        if self.rnc_weight > 0 and self.data.lower() not in ('mosi', 'mosei'):
            raise ValueError('RNC requires continuous MOSI/MOSEI labels')

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)
    parser.add_argument(
        '--shared_align_loss', choices=('cmd', 'atp'), default='cmd',
        help='Shared alignment loss when use_cmd_sim=true.'
    )
    parser.add_argument(
        '--atp_variant', choices=('text', 'all'), default='all',
        help="ATP variant: 'text' anchors audio/video to text; 'all' aligns all three pairs."
    )

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--seed', type=int, default=336)
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=6)

    parser.add_argument('--diff_weight', type=float, default=0.3)
    parser.add_argument('--sim_weight', type=float, default=1.0,
                        help='Weight of the selected shared alignment loss.')
    parser.add_argument('--cu_weight', type=float, default=0.0,
                        help='Centroid uniformity weight; 0 preserves the existing loss.')
    parser.add_argument('--cu_tau', type=float, default=2.0,
                        help='Positive scale in exp(-cu_tau * squared centroid distance).')
    parser.add_argument('--sent_weight', type=float, default=0.0,
                        help='Sentiment centroid KL weight; exclusive with positive CU/RNC weights.')
    parser.add_argument('--sent_sigma', type=float, default=1.0,
                        help='Label neighborhood width in exp(-abs(label difference)/sigma).')
    parser.add_argument('--sent_tau', type=float, default=1.0,
                        help='Representation temperature in exp(-squared centroid distance/tau).')
    parser.add_argument('--rnc_weight', type=float, default=0.0,
                        help='Single-view centroid RNC weight; exclusive with positive CU/KL weights.')
    parser.add_argument('--rnc_tau', type=float, default=2.0,
                        help='Positive RNC temperature dividing the representation distance.')
    parser.add_argument('--rnc_distance', choices=('l2', 'l2sq'), default='l2',
                        help='RNC metric: Euclidean (official) or squared Euclidean (control).')
    parser.add_argument('--sp_weight', type=float, default=0.0)
    parser.add_argument('--recon_weight', type=float, default=1.0)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
    parser.add_argument('--activation', type=str, default='relu')

    # Model
    parser.add_argument('--model', type=str,
                        default='MISA', help='one of {MISA, }')

    # Data
    parser.add_argument('--data', type=str, default='mosi')

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    print(kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = 1
        kwargs.batch_size = 64
    elif kwargs.data == "mosei":
        kwargs.num_classes = 1
        kwargs.batch_size = 16
    elif kwargs.data == "ur_funny":
        kwargs.num_classes = 2
        kwargs.batch_size = 32
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
