import os
import csv
import yaml
import logging
import torch
from torchvision import transforms
from colorlog import ColoredFormatter


# Logging
# =======

def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('better')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
logging.Logger.infov = _infov


# general utils
# =============

def load_config(config_name):
    root = os.getcwd()
    config_path = os.path.join(root, 'configs', config_name + '.yml')
    with open(config_path) as file:
        config = yaml.load(file)
    return config


def generate_tag(tag):
    if not tag:
        import random, string
        letters = string.ascii_lowercase
        tag = ''.join(random.choice(letters) for i in range(5))
        log.warn("Tag is not specified. Random tag '{}' is assigned".format(tag))
    else:
        log.warn("Tag '{}' is specified".format(tag))
    return tag


def setup(mode, model_name, tag):
    directory = dir_path(mode, model_name, tag)
    os.makedirs(directory, exist_ok=True)

    log.info("Directory {} to save checkpoints/results is ready".format(directory))


def dir_path(mode, model_name, tag):
    if mode == 'train':
        root = 'checkpoints'
    else:
        root = 'results'

    directory = os.path.join(root, model_name + '_' + tag)
    return directory


def save_results(mode, model_name, tag, data_name, results):
    headers = {
        'wildcam': ['id', 'animal_present']
    }
    results = headers[data_name] + results

    save_dir = dir_path(mode, model_name, tag)
    save_path = os.path.join(save_dir, 'results.csv')

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)


def save_roc(probs, labels):
    raise NotImplementedError()


def check_eval_type(data_name):
    if data_name == 'wildcam':
        is_label_available = False
    elif data_name in ['nacti', 'tnc']:
        is_label_available = True
    else:
        log.error('Specify right data name - nacti, wildcam, tnc'); exit()
    return is_label_available


# Custom transforms
# =================

class NormalizePerImage(object):
    """Normalize the given image using its mean and standard deviation
    """

    def __call__(self, image_tensor):
        mean = torch.mean(image_tensor, dim=(1,2), keepdim=True)
        std = torch.std(image_tensor, dim=(1,2), keepdim=True)

        return (image_tensor - mean) / std
