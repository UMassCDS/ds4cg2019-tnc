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
    #'mobilenet': models.mobilenet_v2,
    #'resnetxt50': models.resnext50_32x4d
}

SEMI_MODELS = {
}

def build(model_config, checkpoint):
    if 'name' not in model_config:
        log.error('Specify a model name')
    model_name = model_config['name']

    # build model
    if model_name in SUPERVISED_MODELS:
        log.infov('{} model is built'.format(model_name.upper()))
        model = build_supervised_model(model_name, model_config)
    elif model_name in SEMI_MODELS:
        log.infov('{} model is built'.format(model_name.upper()))
        model = build_semi_model(model_name, model_config)
    else:
        SUPERVISED_MODELS.update(SEMI_MODELS)
        log.error(
            'Enter valid model name among {}'.format(SUPERVISED_MODELS)
        ); exit()

    # load model
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        log.infov('Model is built using the given checkpoint')
    else:
        log.infov('Model is built without checkpoint')

    # parallelize model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        log.warn("{} GPUs will be used.".format(torch.cuda.device_count()))

    return model


# Supervised Model
# ================

def build_supervised_model(model_name, model_config):
    # build a model
    model = SUPERVISED_MODELS[model_name](pretrained=False) # imagenet pretrained is False

    # modify the last layer (classifier)
    num_classes = model_config.get('num_classes', 1)
    model = modify_classifier(model, model_name, num_classes)

    return model

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
