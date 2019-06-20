import json
import os

root = '/mnt/nfs/work1/ds4cg'
data = 'tnc/val.json'
data_path = os.path.join(root, data)

with open(data_path, 'r') as f:
    metadata = json.load(f)

import IPython; IPython.embed()
