import time
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import numpy as np


class MetricsWriter():
    def __init__(self, model, model_name='model_name', save_treshold=5):
        self.loss_val = list()
        self.loss_train = list()
        self.acc_val = list()
        self.f1 = list()
        
        self.epoch = 0
        self.start_epoch = 0

        self.save_treshold = save_treshold
        self.model_name = model_name
        self.model_type, self.module = model.get_info()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter('runs/' + self.model_name + '{}_{}_{}'.format(self.model_type, self.module, self.timestamp))

    def get_list_metrics(self):
        return self.loss_val, self.loss_train, self.acc_val

    def print_epoch(self, epoch):
        self.start_epoch = time.time()
        self.epoch = epoch
        print('EPOCH {}:'.format(epoch))

    def print_time(self):
        end_epoch = time.time()
        elapsed = end_epoch - self.start_epoch
        print("Time per epoch {}s".format(elapsed))

    def get_best_epoch_by_map(self):
        return self.f1.index(max(self.f1))

    def writer_step(self, loss, vloss, recall, precision, f1, acc):
        print('LOSS train {} valid {}'.format(loss, vloss))
        print('Accuracy valid {}'.format(acc))
        print('Recall valid {}'.format(recall))
        print('Precision valid {}'.format(precision))
        print('Val F1->{}'.format(f1))
        
        self.loss_train.append(loss)
        self.loss_val.append(vloss)
        self.acc_val.append(acc)
        self.f1.append(f1)

        self.writer.add_scalars('Training vs. Validation Loss',
                    { 'Training': loss, 'Validation': vloss },
                    self.epoch)
        self.writer.add_scalars('Validation Metrics',
                    { 'Validation Recall': recall, 'Validation Precision': precision, 'Validation F1': f1
                    }, self.epoch)
        self.writer.add_scalars('Validation Accuracy',
                    { 'Validation Accuracy': acc
                    }, self.epoch)


    def save_model(self, model):
        if (self.epoch) % self.save_treshold == 0:
            model_path = 'model_svs/' + self.model_name + '_{}_{}_{}_{}'.format(self.model_type, 
                                                                                                self.module, self.timestamp, 
                                                                                                self.epoch)
            torch.save(model.state_dict(), model_path)


class Metrics():
    def __init__(self):
        self.all_actual = np.array([])
        self.all_predicted = np.array([])

    def batch_step(self, actualv, predictedv):
        self.all_actual = np.concatenate([self.all_actual, actualv.detach().cpu().numpy().ravel()])   # actualv.numpy(force=True)])
        self.all_predicted = np.concatenate([self.all_predicted, predictedv.detach().cpu().numpy().ravel()])  # .ravel() 2d to 1d (test)

    def get_metrics(self):
        recall = recall_score(self.all_actual, self.all_predicted)
        precision = precision_score(self.all_actual, self.all_predicted)
        f1 = f1_score(self.all_actual, self.all_predicted)
        accuracy = accuracy_score(self.all_actual, self.all_predicted)
        return recall, precision, f1, accuracy