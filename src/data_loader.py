from keras.preprocessing.image import ImageDataGenerator as ImageDataGenerator
from tensorflow.keras import datasets
from matplotlib import pyplot as plt
import keras.utils.np_utils as kutils


def show_batch(image_batch, label_batch):
    """
    Used for showing a sample of images vailable in the dataset
    :param
      image_batch = batch of images
      label_batch = labels of the corresponding image batch
    """
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')


def load_dataset(batch_size, shuffle, existing_dataset, dataset=None, dataset_path=None):
    """Loads the preprocessed dataset, either downloaded or inbuilt"""

    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)
    if existing_dataset:
        assert dataset_path, "Dataset location not specified!"
        train_data = train_datagen.flow_from_directory(directory=str(dataset_path),
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
        test_data = test_datagen.flow_from_directory(directory=str(dataset_path),
                                                     batch_size=batch_size,
                                                     shuffle=shuffle)
    else:
        assert dataset, "Dataset to be used not specified!"
        if dataset == "cifar10":
            (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        elif dataset == "mnist":
            (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        test_batch_size = len(test_images)
        train_labels = kutils.to_categorical(train_labels)
        test_labels = kutils.to_categorical(test_labels)

        train_datagen.fit(train_images, seed=0, augment=True)
        test_datagen.fit(test_images)
        train_data = train_datagen.flow(train_images, train_labels, batch_size=batch_size, shuffle=shuffle)
        test_data = test_datagen.flow(test_images, test_labels, batch_size=test_batch_size, shuffle=False)

    return train_data, test_data, len(train_images)/batch_size
