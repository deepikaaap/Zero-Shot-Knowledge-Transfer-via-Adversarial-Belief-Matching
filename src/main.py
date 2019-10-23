import argparse
import os


def main():
    args = parser.parse_args()
    if args.existing_dataset:
        args.dataset_path = os.path.join(args.dataset_path, args.dataset)
    print("running the model in ", tf.test.gpu_device_name())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--existing_dataset', default=False, help='Bool value indicating if the dataset is inbuilt or not, False to use inbuilt dataset')
    parser.add_argument('--dataset_path', default=None, help='downloaded path of dataset to be used if ')
    parser.add_argument('--dataset', default='cifar10', help='any inbuilt that is to be dataset to be used')
    parser.add_argument('--student_network_model', default='WResNet', help='the model architecture to be used')
    parser.add_argument('--teacher_network_model', default='WResNet', help='the model architecture to be used')
    parser.add_argument('--batch_size', default='128', help='size of the batch of images for training')
    parser.add_argument('--student_learning_rate', default=2e-3) # According to the paper
    parser.add_argument('--generator_learning_rate', default=2e-3)  # According to the paper
    parser.add_argument('--teacher_total_iterations', default=0.1, help='Number of iterations used while training the teacher, is number of batches of the image*no of epochs')
    parser.add_argument('--student_steps_per_iter', default=10) # According to the paper
    parser.add_argument('--generator_steps_per_iter', default=1) # According to the paper
    parser.add_argument('--pretrained_models', default=True)
    parser.add_argument('--pretrained_model_path', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/trained_teacher_weights/weights/")
    parser.add_argument('--trained_model_path', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/trained_zero_shot_weights/trained/")
    parser.add_argument('--log_directory', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/logs/")

    main()
