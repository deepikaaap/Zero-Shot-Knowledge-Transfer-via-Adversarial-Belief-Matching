import argparse
import os
from zero_shot_solver import *
import logging

def main():
    args = parser.parse_args()
    zero_shot_model = ZeroShotKTSolver(args)
    zero_shot_model.run()
    logging.debug("Starting to run Zero shot model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--existing_dataset', default=False, help='Bool value indicating if the dataset is inbuilt or not, False to use inbuilt dataset')
    parser.add_argument('--dataset_path', default=None, help='downloaded path of dataset to be used if ')
    parser.add_argument('--dataset', default='cifar10', help='any inbuilt that is to be dataset to be used')
    parser.add_argument('--student_network_model', default='WResNet', help='the model architecture to be used')
    parser.add_argument('--batch_size', default=128, help='size of the batch of images for training')
    parser.add_argument('--z_dim', default=100)
    parser.add_argument('--student_learning_rate', default=2e-3)  # According to the paper
    parser.add_argument('--generator_learning_rate', default=2e-3)  # According to the paper
    parser.add_argument('--teacher_total_iterations', default=80000, help='Number of iterations used while training the teacher, is number of batches of the image*no of epochs')
    parser.add_argument('--student_steps_per_iter', default=10) # According to the paper
    parser.add_argument('--generator_steps_per_iter', default=1) # According to the paper
    parser.add_argument('--input_shape', default=(32, 32, 3), help='Shape of the input for the WResNet')
    parser.add_argument('--student_model_depth', default=16, help='Depth of the student WResNet model')
    parser.add_argument('--student_model_width', default=2, help='Width of the student WResNet')
    parser.add_argument('--saved_student_model', default='student_model.h5', help='Saved Student model')
    parser.add_argument('--saved_generator_model', default='generator_model.h5', help='Saved generator model')
    parser.add_argument('--pretrained_teacher_model', default='WRN-16-2-200.h5', help='Pre-trained teacher model')
    parser.add_argument('--pretrained_model_path', default="/home/advanceddeeplearning/Zero-Shot-Knowledge-Transfer-via-Adversarial-Belief-Matching/trained_teacher_weights/weights/")
    parser.add_argument('--trained_model_path', default="/home/advanceddeeplearning/Zero-Shot-Knowledge-Transfer-via-Adversarial-Belief-Matching/trained_zero_shot_weights/trained/")
    parser.add_argument('--path_to_save_checkpoint', default="/home/advanceddeeplearning/Zero-Shot-Knowledge-Transfer-via-Adversarial-Belief-Matching/trained_zeroshot_checkpoint/")
    parser.add_argument('--log_directory', default="/home/advanceddeeplearning/Zero-Shot-Knowledge-Transfer-via-Adversarial-Belief-Matching/logs/")

    main()
