from torch import optim
from src.utils.util import log

OPTIMIZERS = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def build(train_config, model_params, checkpoint):
    if 'optimizer' not in train_config:
        log.error('Specify an optimizer'); exit()

    optim_config = train_config['optimizer']
    optimizer_name = optim_config.pop('name', 'sgd')
    lr_schedule = optim_config.pop('lr_schedule', None)
    optim_config['params'] = model_params

    if optimizer_name in OPTIMIZERS:
        optimizer = OPTIMIZERS[optimizer_name](**optim_config)
    else:
        log.error('Specify valid optimizer name among {}'.format(OPTIMIZERS)); exit()

    if lr_schedule is not None:
        lr_schedule['optimizer'] = optimizer
        # TODO: check if it's okay to assgin the same name as input variable
        optimizer = optim.lr_scheduler.StepLR(**lr_schedule)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    log.infov('{} optimizer is built'.format(optimizer_name.upper()))

    return optimizer
