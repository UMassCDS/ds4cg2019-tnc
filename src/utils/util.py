import os
import csv
import yaml
import time
import fnmatch
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
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def load_checkpoint(checkpoint_path):
    if checkpoint_path is '':
        log.warn('No checkpoint is specified')
        checkpoint = None
    elif checkpoint_path.endswith('.pth'):
        checkpoint = torch.load(checkpoint_path)
    elif os.path.isdir(checkpoint_path):
        checkpoint = torch.load(latest_checkpoint(checkpoint_path))
    else:
        log.error('Specify right checkpoint location')
    return checkpoint


def latest_checkpoint(checkpoint_dir):
    is_new_checkpoint = False
    last_checkpoint_path = read_last_checkpoint(checkpoint_dir)

    while not is_new_checkpoint:
        checkpoints = fnmatch.filter(os.listdir(checkpoint_dir), '*.pth')
        if len(checkpoints) <= 0:
            log.warn('No new checkpoint is available with for 5 minutes')
            time.sleep(300)
            continue

        tags = [checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0]
                for checkpoint_path in checkpoints]

        index, highest_step = -1, -1
        for i, tag in enumerate(tags):
            try:
                step = int(tag)
                if step > highest_step:
                    index = i
            except:
                continue

        try:
            last_step = int(last_checkpoint_path.split('/')[-1].split('_')[-1].split('.')[0])
        except:
            is_new_checkpoint = True

        if not is_new_checkpoint and highest_step > last_step:
            is_new_checkpoint = True

        if not is_new_checkpoint:
            log.warn('No new checkpoint is available with for 5 minutes')
            time.sleep(300)

    checkpoint_path = os.path.join(checkpoint_dir, checkpoints[index])
    write_last_checkpoint(checkpoint_dir, checkpoint_path)
    log.infov('Loading {}'.format(checkpoint_path))

    return checkpoint_path


def read_last_checkpoint(checkpoint_dir):
    checkpoint_history_path = os.path.join(checkpoint_dir, 'checkpoint')

    if not os.path.exists(checkpoint_history_path):
        return None

    with open(checkpoint_history_path, 'r') as f:
        checkpoint_history = f.readline()

    return checkpoint_history


def write_last_checkpoint(checkpoint_dir, checkpoint_path):
    checkpoint_history_path = os.path.join(checkpoint_dir, 'checkpoint')
    with open(checkpoint_history_path, 'w') as f:
        f.write(checkpoint_path)


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

    if mode == 'eval':
        directory = dir_path('train', model_name, tag)
        os.makedirs(directory, exist_ok=True)


def dir_path(mode, model_name, tag):
    if mode == 'train':
        root = 'checkpoints'
    else:
        root = 'results'

    directory = os.path.join(root, model_name + '_' + tag)
    return directory


def save_results(mode, model_name, tag, data_name, results):
    headers = {
        'wildcam': [['id', 'animal_present']]
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
