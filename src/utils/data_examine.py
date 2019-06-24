import json
import os
import numpy as np
from src.utils.util import log
from src.core.dataloader import load

def explore_json(root_dir, json_file):
    json_path = os.path.join(root_dir, json_file)

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    import IPython; IPython.embed()

def normalizing_params(data_names):
    total_mean = []
    total_std = []

    for data_name in data_names:
        data_params = {
            'data_name': data_name,
            'mode': 'train',
            'root_dir': '/mnt/nfs/work1/ds4cg',
            'batch_size': 512,
            'num_workers': 4,
            'label_type': 'binary'
        }
        dataloader = load(**data_params)
        num_batches = len(dataloader['train'])
        for i, (image, label) in enumerate(dataloader['train']):
            numpy_image = image.numpy()

            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std = np.std(numpy_image, axis=(0,2,3))

            total_mean.append(batch_mean)
            total_std.append(batch_std)

            log.infov('{}/{}'.format(i, num_batches))
    mean = np.mean(total_mean, axis=0) / 255.0
    std = np.mean(total_std, axis=0) / 255.0

    log.infov('mean: {}, std: {}'.format(mean, std))


