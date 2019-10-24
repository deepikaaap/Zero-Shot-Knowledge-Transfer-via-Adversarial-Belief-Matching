import tensorflow as tf
import numpy as np
import keras.optimizers as optim
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
import math
from keras.callbacks import Callback
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from data_loader import *
from models.wide_resnet import *
from models.generator import *
import keras.losses as Loss
import keras.backend as K
from utils.helpers import *
from utils.cosine_annealing import *
import tensorflow.keras.backend as TK


def negative_kullback_leibler_divergence(layer):
    negative_loss = -1 * Loss.kullback_leibler_divergence(predicted, true)

    return negative_loss


def compute_attention(student_activations_list, teacher_activations_list, beta):
    if len(student_activations_list) != len(teacher_activations_list):
        raise Exception('Teacher should have equal num of activations as student!')
    else:
        attention_loss = 0.0
        for i in range(0, len(student_activations_list)):
            # L2 norm for each activation
            stud_act_tensor = TK.variable(student_activations_list[i])
            stud_act_norm = tf.keras.backend.l2_normalize(stud_act_tensor, axis=0)

            teach_act_tensor = TK.variable(teacher_activations_list[i])
            teach_act_norm = tf.keras.backend.l2_normalize(teach_act_tensor, axis=0)

            difference_AT = (stud_act_norm - teach_act_norm).pow(2).mean()

            attention_loss += beta * difference_AT

        return attention_loss


class ZeroShotKTSolver():
    def __init__(self, args):
        self.args = args

        # Load dataset
        if self.args.existing_dataset:
            self.args.dataset_path = os.path.join(self.args.dataset_path, self.args.dataset)

        _, test_batches, _ = load_dataset(self.args.batch_size, False,
                                          self.args.existing_dataset,
                                          dataset=self.args.dataset,
                                          dataset_path=self.args.dataset_path)

        # Create the required folder structure
        # TO-DO : Corresponding args to be added in main file
        nb_classes = len(test_batches[0][1][0])
        MODEL_PATH = os.environ.get('MODEL_PATH', self.args.trained_model_path)
        mk_dir(self.args.path_to_save_checkpoint)
        CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', os.path.join(self.args.path_to_save_checkpoint,
                                                                         'WRN-{0}-{1}'.format(
                                                                             self.args.student_model_depth,
                                                                             self.args.student_model_width)))
        mk_dir(CHECKPOINT_PATH)
        mk_dir(MODEL_PATH)
        # Load pre-trained teacher model and set all the layers as non-trainable
        self.teacher_model = load_model(os.path.join(self.args.pretrained_model_path, self.args.pretrained_teacher_model))
        for layer in self.teacher_model.layers:
            layer.trainable = False

        if os.path.exists(self.args.saved_model):
            self.student_model = load_model(os.path.join(MODEL_PATH, self.args.saved_student_model))
            self.generator_model = load_model(os.path.join(MODEL_PATH, self.args.saved_generator_model))
        else:
            # Build student and generator model objects
            if self.args.student_network_model=='WResNet':
                self.student = WideResNet('he_normal', 'uniform', 0.0, self.args.student_learning_rate,
                                          0.0005, 0.1)
                self.student_model = self.student.build_wide_resnet(self.args.input_shape, nb_classes=nb_classes,
                                                                    d=self.args.student_model_depth,
                                                                    k=self.args.student_model_width)
            else:
                print("Not yet implemented")
            self.generator = Generator(self.args)
            self.generator_model = self.generator.build_generator_model()

            # Optimizers and schedulers
            self.optimizer_generator = optim.Adam(self.args.generator_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.optimizer_student = optim.Adam(self.args.student_learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
            self.scheduler_generator = CosineAnnealingScheduler(1000, self.args.generator_learning_rate, 0)
            self.scheduler_student = CosineAnnealingScheduler(1000, self.args.student_learning_rate, 0)

            # Training
            # TO-DO : Loss function to be changed!
            self.student_model.compile(optimizer=self.optimizer_student, loss="kullback_leibler_divergence",
                                       metrics=['accuracy'])
            self.generator_model.compile(optimizer=self.optimizer_generator, loss=negative_kullback_leibler_divergence(student_model.layers[-1]),
                                         metrics=['accuracy'])

            self.generator_callbacks = [
                ModelCheckpoint('models/%s/model.hdf5' % self.args.name, verbose=1, save_best_only=True,
                                save_weights_only=False), self.scheduler_generator]
            self.student_callbacks = [
                ModelCheckpoint('models/%s/model.hdf5' % self.args.name, verbose=1, save_best_only=True,
                                save_weights_only=False), self.scheduler_student]

    def run(self):
        # We are looking to take the same number of steps on the student as was taken on the pretrained teacher.
        total_iterations = np.ceil(self.args.teacher_total_iterations / self.args.student_steps_per_iter)
        output_layers = ['output1', 'output2']  # Layer corresponding every network block
        for i in range(total_iterations):
            # Create a new sample for each iteration
            gen_input = K.random_normal((self.args.batch_size, self.args.z_dim))
            # Just trying to obtain the forward pass output of the generator model, a generated sample
            # To check - Does the generator always create random samples and teacher and student train on that?
            x_sample = generator_model(gen_input)
            # The label for the sampled input is the output from the pre-trained teacher model, that we try imitating.
            teacher_output = teacher_model(x_sample)
            '''
            # Skipping attention for now
            teacher_activations = []
            for layer in self.teacher_model.layers:
                # TO-DO pick only the network block outputs!
                if layer.name in output_layers:
                    teacher_activations.append(layer.output)
            '''
            # steps for generator per iter until total iterations
            for gen_step in range(0, self.args.generator_steps_per_iter):
                # TO-DO :  update with train function
                self.generator_model.train_on_batch(x_sample, teacher_output)

            # student per iter until total iterations
            for stud_step in range(0, self.args.student_steps_per_iter):
                # TO-DO : update with train function
                self.student_model.train_on_batch(x_sample, teacher_output)


        self.student_model.evaluate(test_batches[0][0], test_batches[0][1], len(test_batches[0][0]))
        print('Test loss : %0.5f' % (scores[0]))
        print('Test accuracy = %0.5f' % (scores[1]))



