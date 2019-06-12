import torch.nn as nn

from src.utils.util import log

CRITERIONS = {
    'cross_entropy': nn.BCEWithLogitsLoss
}

def build(train_config):
    # set cross entropy as a default
    if 'criterion' not in train_config:
        train_config['criterion'] = {'name': 'cross_entropy'}

    criterion_name = train_config['criterion'].get('name', 'cross_entropy')

    if criterion_name in CRITERIONS:
        criterion = CRITERIONS[criterion_name]()
    else:
        log.error('Enter valid criterion name among {}'.format(CRITERIONS)); exit()

    log.infov('{} criterion is built'.format(criterion_name.upper()))
    return criterion
