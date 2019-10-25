import math

class CosineAnnealingScheduler():

    def __init__(self, T_max, eta_max, learning_rate, eta_min=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.learning_rate = learning_rate

    def find_current_learning_rate(self, iter):
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * iter / self.T_max)) / 2
        return lr

