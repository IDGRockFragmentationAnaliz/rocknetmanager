import os
import time
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(state, epoch, models_folder: Path):
    models_folder = str(models_folder)

    filename = 'checkpoint_%03d.pth' % epoch
    model_filename = os.path.join(models_folder, filename)
    latest_filename = os.path.join(models_folder, 'latest.txt')

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # write new checkpoint
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


# remove old model
# if saveID is not None and (saveID + 1) % keep_freq != 0:
#     filename = 'checkpoint_%03d.pth' % saveID
#     model_filename = os.path.join(model_dir, filename)
#     if os.path.exists(model_filename):
#         os.remove(model_filename)
#         print('=> removed checkpoint %s' % model_filename)

