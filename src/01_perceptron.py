import random
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# =====================================================================================================================
# Aufgabe 01 - Perceptron
# 30.10.2021, Thomas Iten
# =====================================================================================================================

class Perceptron(object):
    """The perceptron with an activation of zero or one."""

    def __init__(self, input_dimension, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bias = np.zeros(1)
        self.weights = np.zeros(input_dimension)

    def predict_batch(self, inputs):
        res_vector = np.dot(inputs, self.weights) + self.bias
        activations = [1 if res > 0 else 0 for res in res_vector]
        return np.array(activations)

    def predict(self, inputs):
        res = np.dot(inputs, self.weights) + self.bias
        activation = 1 if res > 0 else 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # update bias and weights
                self.bias    += self.learning_rate * (label - prediction)
                self.weights += self.learning_rate * (label - prediction) * inputs


class Samples:
    """The test data samples generator with a handful of helper function like plotting the test data,
    show the inference and create a comare plot.
    """

    def __init__(self, number_of_samples, fy, random_mode="gauss"):
        self.number_of_samples = number_of_samples
        self.fy = fy
        # init test data
        self.x1 = np.linspace(0, 10, number_of_samples)
        if random_mode == "gauss":
            self.x2_ones  = np.array([self.fy(elem) + abs(random.gauss(20,40)) for elem in self.x1])
            self.x2_zeros = np.array([self.fy(elem) - abs(random.gauss(20,40)) for elem in self.x1])
        elif random_mode == "randint":
            self.x2_ones  = np.array([self.fy(elem) + abs(random.randint(1,5)) for elem in self.x1])
            self.x2_zeros = np.array([self.fy(elem) - abs(random.randint(1,5)) for elem in self.x1])
        elif random_mode == "normal":
            self.x2_ones  = np.array([self.fy(elem) + abs(np.random.standard_normal(1)) for elem in self.x1])
            self.x2_zeros = np.array([self.fy(elem) - abs(np.random.standard_normal(1)) for elem in self.x1])
        else:
            raise Exception("Unknown random_mode")

    def get_class_ones(self):
        return np.column_stack((self.x1, self.x2_ones))

    def get_class_zeros(self):
        return np.column_stack((self.x1, self.x2_zeros))

    def get_features(self):
        return np.vstack((self.get_class_ones(), self.get_class_zeros()))

    def get_labels(self):
        return np.hstack((np.ones(self.number_of_samples), np.zeros(self.number_of_samples))).T

    def plot(self):
        plt.scatter(self.x1, self.x2_zeros, c='b')
        plt.scatter(self.x1, self.x2_ones, c='g')
        plt.plot(self.x1, self.fy(self.x1),linestyle='solid', c='r', linewidth=3, label='decision boundary')
        plt.legend()
        plt.show()

    def compare_plot(self, y_hat):
        plt.scatter(self.x1, self.x2_zeros, c='b')
        plt.scatter(self.x1, self.x2_ones, c='g')
        plt.plot(self.x1, self.fy(self.x1),linestyle='solid', c='yellow', linewidth=5, label='original decision boundary')
        plt.plot(self.x1, y_hat, c='red', linewidth=1, ls = '-.', label='perceptron decision boundary')
        plt.legend()
        plt.show()

    def inference(self, perceptron):
        print("Inference with ", self.number_of_samples, "samples:")
        y  = np.ones(self.number_of_samples)
        y_hat = perceptron.predict_batch(self.get_class_ones())
        print("- accuracy ones :", accuracy_score(y, y_hat))
        y  = np.zeros(self.number_of_samples)
        y_hat = perceptron.predict_batch(self.get_class_zeros())
        print("- accuracy zeros:", accuracy_score(y, y_hat))

# ---------------------------------------------------------------------------------------------------------------------
# Experiments section
# ---------------------------------------------------------------------------------------------------------------------

input_dimension=2
epochs=100
learning_rate=0.1

number_of_samples=100

def fy_formula(x, m, b):
    return m*x + b

def fy_10_5(x):
    return fy_formula(x, 10, 5)

def fy_3_0(x):
    return fy_formula(x, 3, 0)

def fy_m3_2(x):
    return fy_formula(x, -3, 2)

def fy(x):
    """Set your current function for the experiments below."""
    return fy_10_5(x)


# Create test data and train perceptron
samples = Samples(number_of_samples, fy)
samples.plot()


print("#")
print("# Train Perceptron:")
print("#")
perceptron = Perceptron(input_dimension, epochs=epochs, learning_rate=learning_rate)
perceptron.train(samples.get_features(), samples.get_labels())

print("- weights:", perceptron.weights)
print("- bias   :", perceptron.bias)
samples.inference(perceptron)

print("#")
print("# Generate new test data:")
print("#")
s10 = Samples(10, fy)
s10.inference(perceptron)

s100 = Samples(100, fy)
s100.inference(perceptron)

s1000 = Samples(1000, fy)
s1000.inference(perceptron)


print("#")
print("# Compare Prediction")
print("#")
print("see graph")

b = perceptron.bias
w1 = perceptron.weights[0]
w2 = perceptron.weights[1]

y_hat = (-(b/w2)/(b/w1))* samples.x1 + (-b/w2)

compare_samples = Samples(number_of_samples, fy)
compare_samples.compare_plot(y_hat)

# =====================================================================================================================
# The end.