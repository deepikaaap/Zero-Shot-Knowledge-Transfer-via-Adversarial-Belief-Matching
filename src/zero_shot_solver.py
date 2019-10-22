import os
import shutil
from time import time
# from models.selector import Selector
import tensorflow as tf
import numpy as np
import keras.optimizers as optim
from keras.callbacks import LearningRateScheduler
import math
from keras.callbacks import Callback
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from data_loader import *
from models.wide_resnet import *
import keras.losses as Loss
import keras.backend as K


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class ZeroShotKTSolver():

    def __init__(self, args):
        self.args = args

        # Load dataset
        if args.existing_dataset:
            args.dataset_path = os.path.join(args.dataset_path, args.dataset)

        train_batches, test_batches, len_train_batch = load_dataset(args.batch_size, args.shuffle,
                                                                    args.existing_dataset,
                                                                    dataset=args.dataset,
                                                                    dataset_path=args.dataset_path)

        # Create the required folder structure
        nb_classes = len(test_batches[0][1][0])
        MODEL_PATH = os.environ.get('MODEL_PATH', args.path_to_save_model)
        mk_dir(args.path_to_save_checkpoint)
        CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', os.path.join(args.path_to_save_checkpoint,
                                                                         'WRN-{0}-{1}'.format(args.model_depth,
                                                                                              args.model_width)))
        mk_dir(CHECKPOINT_PATH)
        mk_dir(MODEL_PATH)

        if os.path.exists(args.saved_model):
            model = load_model(args.saved_model)
        else:
            # Build teacher and student models
            self.student = WideResNet(args.kernel_init, args.gamma_init, args.dropout, args.student_learning_rate,
                                      args.weight_decay,
                                      args.momentum)
            self.generator = WideResNet(args.kernel_init, args.gamma_init, args.dropout, args.generator_learning_rate,
                                        args.weight_decay,
                                        args.momentum)

            self.genarator_model = self.generator.build_wide_resnet(args.input_shape, nb_classes=nb_classes,
                                                                    d=args.model_depth, k=args.model_width)
            self.student_model = self.student.build_wide_resnet(args.input_shape, nb_classes=nb_classes,
                                                                d=args.model_depth, k=args.model_width)

            # Optimizers and schedulers
            self.optimizer_generator = optim.Adam(args.generator_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.optimizer_student = optim.Adam(args.student_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.scheduler_generator = CosineAnnealingScheduler(1000, args.generator_learning_rate, 0)
            self.scheduler_student = CosineAnnealingScheduler(1000, args.student_learning_rate, 0)

            # Training
            self.student_model.compile(optimizer=self.optimizer_student, loss="categorical_crossentropy",
                                       metrics=['accuracy'])
            self.genarator_model.compile(optimizer=self.optimizer_generator, loss="categorical_crossentropy",
                                         metrics=['accuracy'])

            self.generator_callbacks = [
                ModelCheckpoint('models/%s/model.hdf5' % args.name, verbose=1, save_best_only=True,
                                save_weights_only=False), self.scheduler_generator]
            self.student_callbacks = [
                ModelCheckpoint('models/%s/model.hdf5' % args.name, verbose=1, save_best_only=True,
                                save_weights_only=False), self.scheduler_student]

            self.genarator_model.fit_generator(train_batches, steps_per_epoch=len_train_batch, epochs=args.epochs,
                                               callbacks=generator_callbacks,
                                               validation_data=test_batches[0])

    def run(self, n_g, n_s, teacher_epochs, args):
        self.args = args

        total_iterations = np.ceil(len(args.dataset) / args.teacher_batch_size) * args.teacher_epochs

        if (args.dataset == 'cifar10'):

            for i in range(0, total_iterations):

                # n_g steps for generator per iter until total iterations
                for gen_step in range(0, n_g):
                    loss_generator = -1 * Loss.kullback_leibler_divergence()
                    self.genarator_model.compile(optimizer=sgd, loss=loss_generator, metrics=['accuracy'])
                    self.genarator_model.fit_generator(train_batches, steps_per_epoch=len_train_batch,
                                                       epochs=args.epochs, callbacks=generator_callbacks,
                                                       validation_data=test_batches[0])

                # n_s steps for student per iter until total iterations
                for stud_step in range(0, n_s):
                    self.student_model.compile(optimizer=sgd, loss=self.KT_loss_student(), metrics=['accuracy'])
                    self.student_model.fit_generator(train_batches, steps_per_epoch=len_train_batch, epochs=args.epochs,
                                                     callbacks=student_callbacks,
                                                     validation_data=test_batches[0])


        else:
            raise Exception('Train your model using cifar10 dataset.')

    def find_student_loss(self):

        student_loss = Loss.kullback_leibler_divergence() + self.compute_attention()

        return (-1 * student_loss)

#   def compute_attention(self):