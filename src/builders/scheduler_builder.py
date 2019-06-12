from torch import optim

from src.utils.util import log

SCHEDULERS = {
    'step_lr': optim.lr_scheduler.StepLR
}

def build(train_config, optimizer, checkpoint):
    if 'scheduler' not in train_config:
        log.infov('No scheduler is specified')
        return None

    schedule_config = train_config['scheduler']
    scheduler_name = schedule_config.pop('name', 'step_lr')
    schedule_config['optimizer'] = optimizer

    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**schedule_config)
    else:
        log.error(
            'Specify valid scheduler name among {}'.format(SCHEDULERS.keys())
        ); exit()

    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    log.infov('{} scheduler is built'.format(scheduler_name.upper()))

    return scheduler
