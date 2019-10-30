import tensorflow as tf
import numpy as np
import keras.optimizers as optim
import tensorflow.keras.optimizers as tfoptim
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from data_loader import *
from models.wide_resnet import *
from models.generator import *
import tensorflow.keras.losses as Loss
import tensorflow.keras.backend as K
from utils.helpers import *
from utils.cosine_annealing import *
import tensorflow.keras.backend as TK
import logging
from sklearn.metrics import confusion_matrix


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
        print("&$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("Executing in eager mode? ", tf.executing_eagerly())
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # Load dataset
        if self.args.existing_dataset:
            self.args.dataset_path = os.path.join(self.args.dataset_path, self.args.dataset)

        _, self.test_batches, _ = load_dataset(self.args.batch_size, False,
                                               self.args.existing_dataset,
                                               dataset=self.args.dataset,
                                               dataset_path=self.args.dataset_path)

        # Create the required folder structure
        # TO-DO : Corresponding args to be added in main file
        nb_classes = len(self.test_batches[0][1][0])
        MODEL_PATH = os.environ.get('MODEL_PATH', self.args.trained_model_path)
        mk_dir(self.args.path_to_save_checkpoint)
        CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', os.path.join(self.args.path_to_save_checkpoint,
                                                                         'WRN-{0}-{1}'.format(
                                                                             self.args.student_model_depth,
                                                                             self.args.student_model_width)))
        mk_dir(CHECKPOINT_PATH)
        mk_dir(MODEL_PATH)
        # Load pre-trained teacher model and set all the layers as non-trainable
        self.teacher_model = load_model(
            os.path.join(self.args.pretrained_model_path, self.args.pretrained_teacher_model))

        for layer in self.teacher_model.layers:
            layer.trainable = False

        if os.path.exists(self.args.saved_student_model) and os.path.exists(self.args.saved_generator_model):
            self.student_model = load_model(os.path.join(MODEL_PATH, self.args.saved_student_model))
            self.generator_model = load_model(os.path.join(MODEL_PATH, self.args.saved_generator_model))
        else:
            # Build student and generator model objects
            if self.args.student_network_model == 'WResNet':
                self.student = WideResNet('he_normal', 'zeros', 0.0, self.args.student_learning_rate,
                                          0.0005, 0.1)
                self.student_model = self.student.build_wide_resnet(self.args.input_shape,
                                                                    nb_classes=nb_classes,
                                                                    d=self.args.student_model_depth,
                                                                    k=self.args.student_model_width)
            else:
                print("Not yet implemented")
            self.generator = Generator(self.args)
            self.generator_model = self.generator.build_generator_model()

        # Learning rate schedulers
        self.optimizer_generator = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.generator_learning_rate)
        self.optimizer_student = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.student_learning_rate)

        self.scheduler_generator = CosineAnnealingScheduler(1000, self.args.generator_learning_rate, 0)
        self.scheduler_student = CosineAnnealingScheduler(1000, self.args.student_learning_rate, 0)

        # Compiling student model
        self.student_model.compile(optimizer=self.optimizer_student, loss="categorical_crossentropy",
                                   metrics=['accuracy'])

    def run(self):
        # We are looking to take the same number of steps on the student as was taken on the pretrained teacher.
        total_iterations = 1000
        # int(np.ceil(self.args.teacher_total_iterations / self.args.student_steps_per_iter))
        logging.debug("Starting to take iteration steps..")
        # counter for iteration steps:
        for current_iteration in range(total_iterations):
            self.optimizer_generator.learning_rate = self.scheduler_generator.find_current_learning_rate(
                current_iteration)
            self.optimizer_student.learning_rate = self.scheduler_student.find_current_learning_rate(current_iteration)

            # Create a new sample for each iteration
            gen_input = K.random_normal((self.args.batch_size, self.args.z_dim))
            # [print(g) for g in self.generator_model(gen_input)]
            logging.debug("In iteration:", current_iteration)

            for stud_step in range(0, self.args.student_steps_per_iter):
                with tf.GradientTape() as gen_tape, tf.GradientTape() as stud_tape:
                    # self.generator_model(gen_input) - Gets the forward pass output of the generator model
                    if stud_step < self.args.generator_steps_per_iter:
                        gen_loss = -1 * Loss.KLD(tf.reshape(self.teacher_model.predict(self.generator_model(gen_input),
                                                                                       batch_size=self.args.batch_size,
                                                                                       steps=1),
                                                            (1, -1)),
                                                 tf.reshape(self.student_model(self.generator_model(gen_input)),
                                                            (1, -1)))
                        grads = gen_tape.gradient(gen_loss, self.generator_model.trainable_variables)
                        grads = [tf.clip_by_value(g, -5, 5) for g in grads]

                    stud_loss = Loss.KLD(tf.reshape(self.teacher_model.predict(self.generator_model(gen_input),
                                                                               batch_size=self.args.batch_size,
                                                                               steps=1), (1, -1)),
                                         tf.reshape(self.student_model(self.generator_model(gen_input)), (1, -1)))
                # print(stud_loss.numpy())
                student_grads = stud_tape.gradient(stud_loss, self.student_model.trainable_variables)
                student_grads = [tf.clip_by_value(g, -5, 5) for g in student_grads]
                '''
                print("GEN GRAD")
                [print(g.numpy()) for g in grads]
                print("###########################################################")
                print("STUD GRADS")
                [print(g.numpy()) for g in student_grads]
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                #Try this to make sure that the teacher model is loaded properly!!!
            ' 
                y_prede=tf.argmax(self.teacher_model.predict(self.test_batches[0][0], batch_size=10000, steps=1),axis=1)
                y_true= tf.argmax(self.test_batches[0][1],axis=1)
                #print(y_prede[:100].numpy())
                #print("################################################")
                #print(y_true[:100].numpy())
                '''
                if stud_step < self.args.generator_steps_per_iter:
                    grads_and_vars = list(zip(grads, self.generator_model.trainable_variables))
                    self.optimizer_generator.apply_gradients(grads_and_vars)
                student_grads_and_vars = list(zip(student_grads, self.student_model.trainable_variables))
                self.optimizer_student.apply_gradients(student_grads_and_vars)
            scores = self.student_model.evaluate(self.test_batches[0][0], self.test_batches[0][1],
                                                 len(self.test_batches[0][0]))
            print('Test loss : %0.5f' % (scores[0]))
            print('Test accuracy = %0.5f' % (scores[1]))

            y_pred = tf.argmax(
                self.student_model.predict(self.test_batches[0][0], batch_size=self.args.batch_size, steps=1),
                axis=1).numpy()
            y_true = tf.argmax(self.test_batches[0][1], axis=1).numpy()
            y_teacher = tf.argmax(
                self.teacher_model.predict(self.test_batches[0][0], batch_size=self.args.batch_size, steps=1),
                axis=1).numpy()

            # print(y_pred)
            # print(y_true)
            # print(y_teacher)



