import os
from models.wresnet import *

class Selector():

    def __init__(self, pretrained_models_path, model_path, n_classes):

        self.pretrained_models_path = pretrained_models_path
        self.model_path = model_path
        self.n_classes = n_classes

    def select_model(self, dataset, model_name, pretrained=False, pretrained_models_path=None):
        if dataset in ['SVHN', 'CIFAR10']:
            n_classes = self.n_classes

            if model_name == 'WRN-16-1':
                model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0.0)
            elif model_name == 'WRN-16-2':
                model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0.0)
            elif model_name == 'WRN-40-1':
                model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0.0)
            elif model_name == 'WRN-40-2':
                model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0.0)
            else:
                raise NotImplementedError

            # TODO: if pretrained:

        else:
            raise NotImplementedError

        return model


if __name__ == '__main__':

    rand_tensor_ex = tf.random_uniform([64, 3, 32, 32], minval=0, maxval=1, dtype=tf.float32, seed=None, name=None)
    with tf.Session() as sess:  print(rand_tensor_ex.eval())

    selector = Selector()
    model = selector.select_model('CIFAR10', model_name='WRN-16-2')
    output, *act = model(x)
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))