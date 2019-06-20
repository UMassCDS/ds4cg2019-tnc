import torch.nn.functional as F
from src.utils.util import log
from src.core.losses import BinaryClassFocalLoss, MultiClassFocalLoss


CRITERIONS = {'binary_cross_entropy': F.binary_cross_entropy_with_logits,
              'multi_cross_entropy': F.cross_entropy,
              'binary_focal_loss': BinaryClassFocalLoss,
              'multi_focal_lss': MultiClassFocalLoss}

def build(train_config, label_type):
    # set cross entropy as a default
    if 'criterion' not in train_config:
        criterion_config = {'name': 'cross_entropy'}
    criterion_name = label_type + '_' + criterion_config.pop('name')

    if criterion_config:
        criterion = CRITERIONS[criterion_name](**criterion_config)
    else:
        criterion = CRITERIONS[criterion_name]()

    log.infov('{} is built'.format(criterion_name))
    return criterion

