import os
import pickle
import random
import numpy as np

from config import get_config, activation_dict
from data_loader import get_loader
from solver import Solver

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist


if __name__ == '__main__':

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    
    # Setting the config for each stage
    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # Setting random seed
    random_seed = train_config.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    if distributed:
        train_config.batch_size = max(1, train_config.batch_size // world_size)

    print(train_config)

    # Creating pytorch dataloaders
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)


    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    try:
        solver.train()
    finally:
        if distributed:
            dist.destroy_process_group()
