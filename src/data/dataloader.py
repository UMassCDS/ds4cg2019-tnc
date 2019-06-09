import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.datasets import NACTI, TNC, WILDCAM
from src.utils.util import log


DATASETS = {
    'nacti': NACTI,
    'tnc': TNC,
    'wildcam': WILDCAM,
}

def load(data_name, root_dir, batch_size, num_workers, label_type):
    data_dir = os.path.join(root_dir, data_name)

    if data_name == 'nacti':
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
    elif data_name == 'wildcam':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_json = 'train_annotations.json'
        val_json = 'val_annotations.json'
        train_dataset = DATASETS[data_name](data_dir=data_dir,
                                            metadata_file=train_json,
                                            label_type=label_type,
                                            transform=train_transform)
        val_dataset = DATASETS[data_name](data_dir=data_dir,
                                          metadata_file=val_json,
                                          label_type=label_type,
                                          transform=val_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    elif data_name == 'tnc':
        return
    else:
        log.error('Specify right data name for train/val - nacti, tnc, wildcam')

    # TODO: For eval, always use TNC testset
    eval_dataloader = None

    dataloader = {'train': train_dataloader,
                  'val': val_dataloader,
                  'eval': eval_dataloader}
    return dataloader







