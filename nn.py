import numpy as np

class Combiner():
    def __call__(self, x):
        return [np.sum(temp) for temp in x * self.w]

    def __init__(self, w):
        self.w = w

def relu(x):
    return np.maximum(0, x)

class Logistic():
    def __init__(self):
        pass

    def backprop(self, x):
        return self.y * (1 - self.y)

    def __call__(self, x):
        self.y = 1./(1+np.power(np.e, -x))
        return self.y

logistic = Logistic()
