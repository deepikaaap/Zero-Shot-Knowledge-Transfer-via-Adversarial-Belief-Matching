import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import random


class AggregateScalar(object):
    """
    Computes and stores the average and std of stream.
    Mostly used to average losses and accuracies.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0001  # to avoid division with 0
        self.sum = 0

    def update(self, val, weight=1):
        """
        :param val: new running value
        :param w: weight, e.g batch size
        """
        self.sum += weight * (val)
        self.count += weight

    def avg(self):
        return self.sum / self.count


def compute_regularization(self, W, l):
    """
        compute the regularization term: l * ||W||^2
    """
    return l * np.sum(np.square(W))


def compute_ce_loss(self, Y, P):
    p_one_hot = np.sum(np.prod((Y, P), axis=0), axis=0)
    loss = np.sum(0 - np.log(p_one_hot))
    return loss


def compute_accuracy(self, predictions_, y_labels):
    """
        Percentage of correctly classified predictions
    """

    predictions = np.argmax(predictions_, axis=0)
    corrects = np.where(predictions - y_labels == 0)
    num_of_corrects = len(corrects[0])

    return (num_of_corrects / np.size(y_labels[0]))


def accuracy(predictions_, target, topk=(1,)):
    """
      Computes the precision@k for the specified values of k
    """
    if len(target.shape) > 1:
        target = np.argmax(target, axis=0)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = predictions_.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def str2bool(v):
    """
      used in argparse, to pass booleans
      codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise UserWarning


def delete_files_from_name(folder_path, file_name, type='contains'):
    """ Delete log files based on their name"""
    assert type in ['is', 'contains']
    for f in os.listdir(folder_path):
        if (type == 'is' and file_name == f) or (type == 'contains' and file_name in f):
            os.remove(os.path.join(folder_path, f))


def set_tensorflow_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    scalar = AggregateScalar()
    print(scalar)