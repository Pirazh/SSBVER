import os, sys, shutil
import os.path as osp
import time
import json
import random

import numpy as np
import torch
from torch import nn


def fix_random_seeds(seed=0):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_configs(args):
    if args.is_train:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)
            f.close()

def create_folder(args):
    if args.is_train:
        if osp.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            os.makedirs(args.output_dir)
