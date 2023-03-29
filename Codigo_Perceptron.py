import numpy as np

# Función de activación escalón unitario
def step_function(x):
    return 1 if x >= 0 else 0

# Clase perceptrón
class Perceptron:
    
    # Inicialización de los pesos y el umbral
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
    
    # Predicción
    def predict(self, input):
        linear_combination = np.dot(self.weights, input) + self.bias
        return step_function(linear_combination)
    
    # Entrenamiento por regla de aprendizaje
    def train(self, training_inputs, labels, num_epochs):
        for epoch in range(num_epochs):
            for input, label in zip(training_inputs, labels):
                prediction = self.predict(input)
                error = label - prediction
                self.weights += self.learning_rate * error * input
                self.bias += self.learning_rate * error

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels, num_epochs=10)

print(perceptron.predict(np.array([0, 0]))) # 0
print(perceptron.predict(np.array([0, 1]))) # 0
print(perceptron.predict(np.array([1, 0]))) # 0
print(perceptron.predict(np.array([1, 1]))) # 1

training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels, num_epochs=10)

print(perceptron.predict(np.array([0, 0]))) # 0
print(perceptron.predict(np.array([0, 1]))) # 1
print(perceptron.predict(np.array([1, 0]))) # 1
print(perceptron.predict(np.array([1, 1]))) # 1

training_inputs = np.array([[0], [1]])
labels = np.array([1, 0])

perceptron = Perceptron(1)
perceptron.train(training_inputs, labels, num_epochs=10)

print(perceptron.predict(np.array([0]))) # 1
print(perceptron.predict(np.array([1]))) # 0
