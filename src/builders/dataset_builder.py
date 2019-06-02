import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.utils.util import log
from src.datasets import NACTI, TNC


DATASETS = {
	'nacti': NACTI,
	'tnc': TNC,
}


def build(data_config):
    if 'name' not in data_config:
        log.error('Specify a data name')

    data_name = data_config['name']

    data_dir = data_config.get('data_dir', '/mnt/nfs/work1/ds4cg/wbae/data/nacti')

    batch_size = data_config.get('batch_size', 128)
    num_workers = data_config.get('num_workers', 4)
    label_type = data_config.get('label_type', 'binary')
    
    transform = transforms.Compose([
    		transforms.Resize((224, 224)),
    		transforms.ToTensor()
    	])

    # TODO: add eval using tnc dataset
    dataset = DATASETS[data_name](data_dir=data_dir,
    							  label_type=label_type,
    							  transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset,
    										  [train_size, val_size])
    training_dataloader = DataLoader(train_dataset, batch_size=batch_size,
    								 shuffle=True, num_workers=num_workers)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size,
    								   shuffle=False, num_workers=num_workers)
    
    dataloader = {'train': training_dataloader,
    			  'val': validation_dataloader,
    			  'eval': None}
    return dataloader



