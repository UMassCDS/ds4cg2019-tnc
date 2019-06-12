import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.datasets import NACTI, TNC, WILDCAM
from src.utils.util import log, NormalizePerImage


DATASETS = {
    'nacti': NACTI,
    'tnc': TNC,
    'wildcam': WILDCAM,
}

def load(mode, data_name, root_dir, batch_size, num_workers, label_type):
    if data_name == 'nacti':
      if mode == 'eval':
        log.error('Evaluation dataset for {} is not available'.format(data_name)); exit()
      dataloaders = load_nacti(mode, data_name, root_dir, batch_size, num_workers, label_type)
    elif data_name == 'wildcam':
      dataloaders = load_wildcam(mode, data_name, root_dir, batch_size, num_workers, label_type)
    elif data_name == 'tnc':
      # TODO: add tnc dataset
      dataloaders = None
    else:
      log.error('Specify right data name for {} - nacti, tnc, wildcam'.format(mode))

    if mode == 'train':
      dataloader = {'train': dataloaders[0], 'val': dataloaders[1]}
    elif mode == 'eval':
      dataloader = {'eval': dataloaders}
    else:
      log.error('Specify right mode - train, eval')

    return dataloader


def load_nacti(mode, data_name, root_dir, batch_size, num_workers, label_type):
    '''
    Only train and validation sets are available at this point.
    '''
    data_dir = os.path.join(root_dir, data_name)

    transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
    ])

    train_val_json = 'nacti_metadata.json'
    dataset = DATASETS[data_name](data_dir=data_dir,
                                  metadata_file=train_val_json,
                                  label_type=label_type,
                                  transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,
                                              [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=num_workers)
    return (train_dataloader, val_dataloader)


def load_wildcam(mode, data_name, root_dir, batch_size, num_workers, label_type):
    '''
    For evaluation dataset, there is no labels. It will return None for each label
    '''
    data_dir = os.path.join(root_dir, data_name)

    if mode == 'train':
      train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        NormalizePerImage()
      ])

      val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NormalizePerImage()
      ])

      train_json = 'train_annotations.json'
      val_json = 'val_annotations.json'
      train_dataset = DATASETS[data_name](data_dir=data_dir,
                                          metadata_file=train_json, label_type=label_type,
                                          transform=train_transform)
      val_dataset = DATASETS[data_name](data_dir=data_dir,
                                        metadata_file=val_json, label_type=label_type,
                                        transform=val_transform)
      train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
      val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
      return (train_dataloader, val_dataloader)
    elif mode == 'eval':
      eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        NormalizePerImage()
      ])

      eval_dataset = DATASETS[data_name](data_dir=data_dir,
                                         metadata_file=None, label_type=None,
                                         transform=eval_transform, mode='eval')
      eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)
      return eval_dataloader


def load_tnc(data_name, root_dir, batch_size, num_workers, label_type, mode):
    data_dir = os.path.join(root_dir, data_name)
    return

