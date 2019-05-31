import os

from torch.utils.data import Dataset, Dataloader
from torchvision import transforms
from src.utils.util import log


DATASETS = {
	'nacti': NACTI,
	'tnc': TNC,
}


def build(data_config):
	if 'name' not in data_config:
        log.error('Specify a data name')

    data_name = data_config['name']
    root_dir = data_config.get('rood_dir', 'data/nacti')
    batch_size = data_config.get('batch_size', 128)
    num_workers = data_config.get('num_workers', 4)
    label_type = data_config.get('label_type', 'binary')
    
    transform = transforms.Compose([
    		transforms.Resize((256, 256)),
    		transforms.ToTensor()
    	])

    # TODO: add eval using tnc dataset
    dataset = DATASETS[data_name](root_dir=root_dir,
    							  type=label_type,
    							  transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
    														   [train_size, val_size])
    training_dataloader = Dataloader(train_dataset, batch_size=batch_size,
    								 shuffle=True, num_workers=num_workers)
    validation_dataloader = Dataloader(val_dataset, batch_size=batch_size,
    								   shuffle=False, num_workers=num_workers)
    
    dataloader = {'train': training_dataloader,
    			  'val': validation_dataloader,
    			  'eval': None}
	return dataloader



class NACTI(Dataset):
	# 0: not animal, 1: animal
	BINARY = {1:1, 3:1, 4:1, 5:1, 6:1, 7:1, 9:1, 10:1, 11:1, 12:1, 13:1,
	          14:1, 15:1, 16:0, 17:1, 18:1, 19:1, 21:1, 22:1, 23:1, 24:1,
	          26:1, 27:1, 28:1, 29:1, 30:1, 31:1, 32:1, 33:1, 34:1, 35:1,
	          36:1, 37:1, 38:1, 40:1, 41:1, 42:1, 43:1, 44:1, 46:1, 47:1,
			  50:1, 53:1, 54:1, 55:1, 56:1, 57:1, 58:1, 59:1, 60:1, 62:1,
			  63:1, 64:1, 65:1, 66:1, 67:1, 68:0, 69:1, 70:1}
	MULTI = {}
	LABEL_TYPES = {'binary': BINARY, 'multi': MULTI}

	def __init__(self, root_dir, label_type, tranform=None):
		self.root_dir = root_dir
		
		# load meta data
		metadata_path = os.path.join(self.root_dir, 'nacti_metadata.json')
		with open(metadata_path, 'r') as f:
			self.metadata = json.load(f)
		
		# initialize label map from the original label to a new label
		if label_type not in LABEL_TYPES:
			log.error('Specify right type for NACTI dataset - binary or multi')
		self.label_map = LABEL_TYPES[label_type]
		
		# transformer
		self.transform = transform

	def __len__(self):
		return len(self.metadata['annotations'])

	def __getitem__(self, idx):
		# read an image
		image_path = os.path.join(self.root_dir,
								  metadata['images'][idx]['file_name'])
		image = io.imread(image_path)

		# get a label
		original_label = metadata['annotations'][idx]['category_id']
		label = self.label_map[original_label]

		if self.transform:
			image = self.transform(image)

		return image, label

class TNC(Dataset):

	def __init__(self, root_dir, json_file, transform=None):
		self.root_dir = root_dir
		self.json_file = json_file
		self.transform = transform

	def __len__(self):
		return

	def __getitem__(self, idx):
		sample = None

		if self.transform:
			sample = self.transform(sample)
		return

