import os

from src.utils.util import log
from src.data.dataloader import load


def build(data_config):
    if 'name' not in data_config:
        log.error('Specify a data name')

    data_params = {
        'data_name': data_config['name'],
        'mode': data_config['mode']
        'root_dir': data_config.get('root_dir', '/mnt/nfs/work1/ds4cg'),
        'batch_size': data_config.get('batch_size', 128),
        'num_workers': data_config.get('num_workers', 4),
        'label_type': data_config.get('label_type', 'binary')
    }

    dataloader = load(**data_params)
    return dataloader

