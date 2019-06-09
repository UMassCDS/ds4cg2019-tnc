

# DS4CG TNC      
  
  

## Installation

Some python libraries are required to use our pipeline. All the necessary python libraries are specified in requirements.txt. Run the following commend to install the libraries at once.
```bash
bash scripts/setup.sh
```

## Datasets

### iWildCam2018

As a pretraining step of our model, we first train the model on iWildCam2018 dataset. For detailed information about iWildCam2018, please refer to <a href='https://github.com/visipedia/iwildcam_comp/tree/master/2018'>iWildCam2018 github repo</a>. Run the following commend to download iWildCam2018 train, validation and test sets and corresponding annotation file. Note that download_path flag indicates the directory to download the dataset. If the flag is not provided, the dataset will be downloaded to $repo/data/wildcam directory. People who are using Gypsum do not need to download the dataset.

```bash
bash scripts/download --download_path data/wildcam
```

### TNC 
TBD


## Training

Outside of Gypsum:
```bash
bash scripts/run_train.sh --config resnet18 --tag test
```

In Gypsum:

```bash
sbatch -p titanx-short --gres=gpu:1 --output=out/test.out scripts/run_train.sh --config resnet18 --tag test 
```

## Evaluation

Outside of Gypsum:
```bash
bash scripts/run_eval.sh --config resnet18 --tag test
```

In Gypsum:

```bash
sbatch -p titanx-short --gres=gpu:1 --output=out/test.out scripts/run_eval.sh --config resnet18 --tag test 
```