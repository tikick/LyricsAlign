# Some of the code comes from https://github.com/jhuang448/LyricsAlignment-MTL

import torch
import numpy as np
from prettytable import PrettyTable


def set_seed(seed=97):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_num_params = 0
    for name, params in model.named_parameters():
        if not params.requires_grad:
            continue
        num_params = params.numel()
        table.add_row([name, num_params])
        total_num_params += num_params
    print(table)
    print(f'Total Trainable Params: {total_num_params}')
    return total_num_params