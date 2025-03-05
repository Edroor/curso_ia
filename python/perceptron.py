import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.zeros(input_size + 1)  # Pesos inicializados en 0 (incluye bias)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0  # Función de activación escalón

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]  # Producto punto + bias
        return self.activation_function(z)

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                y_pred = self.predict(X[i])
                error = y[i] - y_pred
                self.weights[1:] += self.learning_rate * error * X[i]  # Actualiza pesos
                self.weights[0] += self.learning_rate * error  # Actualiza bias

# Datos de entrada (X) y etiquetas (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entrada binaria
y = np.array([0, 0, 0, 1])  # Salida lógica AND

# Inicializa y entrena el perceptrón
perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

# Predicción con el modelo entrenado
for i in range(len(X)):
    print(f"Entrada: {X[i]}, Salida predicha: {perceptron.predict(X[i])}")
