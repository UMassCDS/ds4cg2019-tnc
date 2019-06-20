import os
import json
import fnmatch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.util import log

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

    def __init__(self, data_dir, metadata_file, label_type, transform=None):
        self.data_dir = data_dir

        # load meta data
        meatadata_path = os.path.join(self.data_dir, metadata_file)
        with open(meatadata_path, 'r') as f:
            self.metadata = json.load(f)

        # initialize label map from the original label to a new label
        if label_type not in self.LABEL_TYPES:
            log.error('Specify right label type for NACTI dataset - binary or multi')
        self.label_map = self.LABEL_TYPES[label_type]

        # transformer
        self.transform = transform

    def __len__(self):
        return len(self.metadata['annotations'])

    def __getitem__(self, idx):
        # read an image
        image_path = os.path.join(self.data_dir,
                                  self.metadata['images'][idx]['file_name'])
        image = Image.open(image_path)

        # get a label
        original_label = self.metadata['annotations'][idx]['category_id']
        label = self.label_map[original_label]

        if self.transform:
            image = self.transform(image)

        return (image, label)

class TNC(Dataset):
    BINARY = {0: 0, 1: 1}
    LABEL_TYPES = {'binary': BINARY}

    def __init__(self, data_dir, metadata_file, transform=None, mode='train'):
        self.data_dir = data_dir

        self.metadata = metadata_file
        self.transform = transform

    def __len__(self):
        length = len(self.metadata['annotations'])
        return length

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir,
                                  self.metadata['images'][idx]['id'])
        # get a label
        original_label = self.metadata['annotations'][idx]['category_id']
        label = self.label_map[original_label]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return (image, label)

class WILDCAM(Dataset):
    # 0: not animal, 1: animal
    BINARY = {0: 0, 1: 1}
    LABEL_TYPES = {'binary': BINARY}
    MODES = {'train', 'eval'}

    def __init__(self, data_dir, metadata_file, label_type, transform=None, mode='train'):
        # train / eval
        if mode not in self.MODES:
          log.error('Specify right mode for WILDCAM dataset - train, eval'); exit()
        self.mode = mode
        self.data_dir = data_dir

        if self.mode != 'eval':
          # load meta data
          metadata_path = os.path.join(self.data_dir, metadata_file)
          with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

          # initialize label map from the original label to a new label
          if label_type not in self.LABEL_TYPES:
            log.error('Specify right label type for WILDCAM dataset - binary')
          self.label_map = self.LABEL_TYPES[label_type]
        else:
          self.data_dir = os.path.join(data_dir, 'test')
          self.metadata = fnmatch.filter(os.listdir(self.data_dir), '*.jpg')

        # transformers
        self.transform = transform

    def __len__(self):
        if self.mode == 'eval':
            length = len(self.metadata)
        else:
            length = len(self.metadata['annotations'])
        return length

    def __getitem__(self, idx):
        # In eval mode, labels are not available
        if self.mode == 'eval':
            image_path = os.path.join(self.data_dir,
                                      self.metadata[idx])
            # image id
            label = self.metadata[idx].split('.')[0]
        else:
            image_path = os.path.join(self.data_dir,
                                      self.metadata['images'][idx]['file_name'])
            # get a label
            original_label = self.metadata['annotations'][idx]['category_id']
            label = self.label_map[original_label]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return (image, label)


