from __future__ import print_function
from __future__ import division

import os
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from src.utils import util
from src.utils.util import log
from src.builders import model_builder, dataset_builder, \
    optimizer_builder, criterion_builder, scheduler_builder

log.info("PyTorch Version: {}".format(torch.__version__))
log.info("Torchvision Version: {}".format(torchvision.__version__))


class BaseEngine(object):

    def __init__(self, mode, config_name, tag):
        self.mode = mode
        self.tag = util.generate_tag(tag)

        # assign configurations
        config = util.load_config(config_name)
        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        # misc information
        self.model_name = self.model_config['name']
        self.num_classes = self.model_config['num_classes']

        # setup a directory to store checkpoints or evaluation results
        util.setup(self.mode, self.model_name, self.tag)

        # store data name to data config depending on which mode we are on
        if self.mode == 'train':
            data_name = self.train_config['data']
        elif self.mode == 'eval':
            data_name = self.eval_config['data']
        else:
            log.error('Specify right mode - train, eval'); exit()

        self.data_config['name'] = data_name
        self.data_config['mode'] = mode

        # determine which device to use - cpu or gpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if device == "cpu":
            log.warn("GPU is not available. Please check the configuration.")
        else:
            log.warn("GPU is available.")

        self.writer = SummaryWriter()

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


class Engine(BaseEngine):

    def __init__(self, mode, config_name, tag):
        super(Engine, self).__init__(mode, config_name, tag)

        # build dataloader
        self.dataloader = dataset_builder.build(self.data_config)

        # load checkpoint
        if self.mode == 'train':
            self.checkpoint = util.load_checkpoint(self.train_config['checkpoint_path'])
        elif self.mode == 'eval':
            self.checkpoint = util.load_checkpoint(self.eval_config['checkpoint_path'])

        # build model
        self.model = model_builder.build(self.model_config, self.checkpoint)
        self.model.to(self.device)

        # build optimizer/criterion
        if self.mode == 'train':
            self.optimizer = optimizer_builder.build(
                self.train_config, self.model.parameters(), self.checkpoint)
            self.scheduler = scheduler_builder.build(
                self.train_config, self.optimizer, self.checkpoint)
            self.criterion = criterion_builder.build(self.train_config)


    def train(self):
        start_epoch = 0 if self.checkpoint is None else self.checkpoint['epoch']
        num_epochs = self.train_config.get('num_epochs', 50)
        if num_epochs < start_epoch:
            num_epochs = start_epoch + 50

        log.info(
            "Training for {} epochs starts from epoch {}".format(num_epochs, start_epoch)
        )

        val_accuracies = []
        best_acc = 0.0

        for epoch in range(start_epoch, num_epochs):
            train_start = time.time()

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss = self._train_one_epoch()
            self._save_model(epoch)

            time_elapsed = time.time() - train_start
            log.info(
                'Epoch {} completed in {} - train loss: {:4f}'\
                .format(epoch, time_elapsed, train_loss)
            )

            val_start = time.time()
            val_loss, val_acc = self.validate()

            # save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                self._save_model()

            val_accuracies.append(val_acc)
            time_elapsed = time.time() - val_start

            log.infov(
                'Epoch {} completed in {} - val loss: {:4f}, val accuracy {:4f}'\
                .format(epoch, time_elapsed, val_loss, val_acc)
            )


    def _train_one_epoch(self):
        total_loss, num_corrects = 0.0, 0
        num_batches = len(self.dataloader['train'])
        self.model.train()

        for i, (inputs, labels) in enumerate(self.dataloader['train']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).unsqueeze(-1).float()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward propagation
            loss.backward()
            self.optimizer.step()

             # Statistics
            total_loss += loss.item() * inputs.size(0)

            log.info(
                'Train batch {}/{} - loss: {:4f}'\
                .format(i+1, num_batches, loss)
            )
            #self.writer.add_scalar('training_loss', loss, i)
            if i == 10:
                self._save_model(0)
        train_loss = total_loss / len(self.dataloader['train'].dataset)
        return train_loss


    def validate(self):
        total_loss, num_corrects = 0.0, 0
        self.model.eval()

        for inputs, labels in self.dataloader['val']:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).unsqueeze(-1).float()

            # Forward propagation
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Use softmax when the num of classes > 1 else sigmoid
            if self.num_classes > 1:
                _, predictions = torch.max(outputs, 1)
            else:
                probabilities = torch.sigmoid(outputs)
                predictions = torch.gt(probabilities, 0.5).float()

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            num_corrects += torch.sum((predictions == labels).int())

        val_loss = total_loss / len(self.dataloader['val'].dataset)
        val_acc = num_corrects.double() / len(self.dataloader['val'].dataset)
        return val_loss, val_acc


    def evaluate(self):
        # check whether labels are available for evaluation
        data_name = self.eval_config['data']
        is_label_available = util.check_eval_type(data_name)

        num_batches = len(self.dataloader['eval'])
        use_roc = self.eval_config.get('use_roc', False)
        stop = False

        while not stop:
            self._evaluate_once(data_name, is_label_available, num_batches, use_roc)
            stop = self._reload_model()


    def _evaluate_once(self, data_name, is_label_available, num_batches, use_roc):
        if is_label_available:
            total_loss, num_corrects = 0.0, 0
        else:
            prediction_results = []

        self.model.eval()
        for i, (inputs, labels) in enumerate(self.dataloader['eval']):
            inputs = inputs.to(self.device)
            if is_label_available:
                labels = labels.to(self.device).unsqueeze(-1).float()

            # Forward propagation
            outputs = self.model(inputs)
            if is_label_available:
                loss = self.criterion(outputs, labels)

            # Use softmax when the num of classes > 1 else sigmoid
            if self.num_classes > 1:
                _, predictions = torch.max(outputs, 1)
            else:
                probabilities = torch.sigmoid(outputs)
                predictions = torch.gt(probabilities, 0.5)
                if use_roc and is_label_available:
                    # TODO: implement save_roc
                    util.save_roc(probabilities, labels)

            # Statistics
            if is_label_available:
                total_loss += loss.item() * inputs.size(0)
                num_corrects += torch.sum(predictions == labels.data)
            else:
                predictions = predictions.data.cpu().numpy().flatten()
                ids = [id for id in labels]
                for prediction, id in zip(predictions, ids):
                    prediction_results.append([id, int(prediction)])

            log.info(
                'Evaluation batch {}/{}'.format(i+1, num_batches)
            )

        if is_label_available:
            eval_loss = total_loss / len(self.dataloader['eval'].dataset)
            eval_acc = num_corrects.double() / len(self.dataloader['eval'].dataset)
            # TODO: add step
            self.writer.add_scalar('eval accuracy', eval_acc)
        else:
            util.save_results(self.mode, self.model_name, self.tag,
                              data_name, prediction_results)

    # TODO: create model saver/loader
    def _save_model(self, epoch=None):
        if epoch is not None:
            checkpoint_tag = str(epoch)
        else:
            checkpoint_tag = 'best'

        save_dir = util.dir_path(self.mode, self.model_name, self.tag)
        checkpoint_path = os.path.join(
            save_dir, 'checkpoint' + '_' + checkpoint_tag + '.pth')

        model_params = {
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        if torch.cuda.device_count() > 1:
            model_params['model_state_dict'] = self.model.module.state_dict()
        else:
            model_params['model_state_dict'] = self.model.state_dict()

        if self.scheduler is not None:
            model_params['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(model_params, checkpoint_path)


    def _reload_model(self):
        stop = False
        checkpoint_path = self.eval_config['checkpoint_path']
        if checkpoint_path.endswith('.pth'):
            stop = True
            return stop
        self.checkpoint = util.load_checkpoint(self.eval_config['checkpoint_path'])

        # build model
        self.model = model_builder.build(self.model_config, self.checkpoint)
        self.model.to(self.device)
        return stop
