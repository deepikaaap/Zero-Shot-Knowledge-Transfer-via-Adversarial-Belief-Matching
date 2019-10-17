import tensorflow as tf

def main():
    args = parser.parse_args()
    print('a {} flower'.format(args.existing_dataset))
    if args.existing_dataset:
    args.dataset_path =  os.path.join(args.dataset_path, args.dataset)
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    print("running the model in ", tf.test.gpu_device_name())


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--existing_dataset', default=False, help='Bool value indicating if the dataset is inbuilt or not, False to use inbuilt dataset')
    parser.add_argument('--dataset_path', default=None, help='downloaded path of dataset to be used if ')
    parser.add_argument('--dataset', default='cifar10', help='any inbuilt that is to be dataset to be used')
    parser.add_argument('--student_network_model', default='WResNet', help='the model architecture to be used')
    parser.add_argument('--teacher_network_model', default='WResNet', help='the model architecture to be used')
    parser.add_argument('--batch_size', default='128', help='size of the batch of images for training')
    parser.add_argument('--student_learning_rate', default=2e-3)
    parser.add_argument('--teacher_learning_rate', default=0.1)
    parser.add_argument('--generator_learning_rate', default=1e-3)
    parser.add_argument('--pretrained_models', default=True)
    parser.add_argument('--pretrained_model_path', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/weights/pretrained/")
    parser.add_argument('--trained_model_path', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/weights/trained/")
    parser.add_argument('--log_directory', default="/home/advanceddeeplearning/ZeroShotKnowledgeTransfer/logs/")
    parser.add_argument('--workers', default=1)

    main()