import torch
import torch.nn as nn
from torchvision import models

from src.utils.util import log

# TODO: upgrade torchvision to the latest version - mobilenet and resentxt is not available for 0.3.0 version
SUPERVISED_MODELS = {
    'alexnet': models.alexnet, 
    'vgg11': models.vgg11_bn,
    'vgg13': models.vgg13_bn,
    'vgg16': models.vgg16_bn,
    'vgg19': models.vgg19_bn,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'squeezenet1.0': models.squeezenet1_0,
    'squeezenet1.1': models.squeezenet1_1,
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'inception': models.inception_v3,
    #'mobilenet': models.mobilenet_v2,
    #'resnetxt50': models.resnext50_32x4d
}

SEMI_MODELS = {
}

def build(model_config):
    if 'name' not in model_config:
        log.error('Specify a model name')
    model_name = model_config['name']
        
    if model_name in SUPERVISED_MODELS:
        log.infov('{} model is built'.format(model_name.upper()))
        return build_supervised_model(model_name, model_config)
    elif model_name in SEMI_MODELS:
    	log.infov('{} model is built'.format(model_name.upper()))
    	return build_semi_model(model_name, model_config)
    else:
        SUPERVISED_MODELS.update(SEMI_MODELS)
        log.error('Enter valid model name among {}'.format(SUPERVISED_MODELS))
        exit()


# Supervised Model
# ================

def build_supervised_model(model_name, model_config):
    misc = {}
    if 'inception' in model_name:
        misc['is_inception'] = True

    # build a model
    model = SUPERVISED_MODELS[model_name](pretrained=False) # imagenet pretrained is False
        
    # load a pretrained model
    pretrained = model_config.get('pretrained', False)
    if pretrained:
        path = model_confg['checkpoint_path']
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        log.infov('Checkpoint is loaded')
        misc['checkpoint'] = checkpoint
    
    # freeze parameters except the last layer (classifier)
    freeze = model_config.get('freeze', False)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        log.warn('Recommend not to freeze the model parameters')

    # modify the last layer (classifier)
    num_classes = model_config.get('num_classes', 2)
    model = modify_classifier(model, model_name, num_classes)

    log.info(
        'Model config - pretrained: {0}, freeze: {1}, num of classes: {2}'\
        .format(pretrained, freeze, num_classes))

    return model, misc

def modify_classifier(model, model_name, num_classes):
    if 'alexnet' in model_name:
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif 'vgg' in model_name:
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif 'resnet' in model_name:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'squeezenet' in model_name:
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
    elif 'densnet' in model_name:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    elif 'inception' in model_name:
        num_features = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_features, num_classes)
        # Handle the primary net
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features,num_classes)
    elif 'mobilenet' in model_name:
        raise NotImplementedError()
    elif 'resnetxt' in model_name:
        raise NotImplementedError()
    else:
    	log.error('Invalid model name {} for modifying a classifier'.format(model_name))
    	exit()
    return model

# Semi-supervised Model
# ================

def build_semi_model(model_name, model_config):
    raise NotImplementedError()
