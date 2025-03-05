#Perceptrón Multicapa (MLP) con Sigmoide, la función XOR.

import numpy as np

# Función de activación sigmoide y su derivada correcta
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Se calcula sobre la salida de la sigmoide

# Datos de entrada (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # Salidas esperadas de XOR

# Inicialización de pesos aleatorios
np.random.seed(42)
input_size = 2
hidden_size = 2  # 2 neuronas en la capa oculta
output_size = 1

W1 = np.random.uniform(-1, 1, (input_size, hidden_size))  # Pesos entrada -> oculta
b1 = np.random.uniform(-1, 1, (1, hidden_size))  # Bias de la capa oculta
W2 = np.random.uniform(-1, 1, (hidden_size, output_size))  # Pesos oculta -> salida
b2 = np.random.uniform(-1, 1, (1, output_size))  # Bias de la capa de salida

# Parámetros de entrenamiento
epochs = 10000
learning_rate = 0.1

# Entrenamiento con retropropagación
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)  # Activación sigmoide en capa oculta

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)  # Activación sigmoide en capa de salida

    # Cálculo del error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)  # Gradiente de la capa de salida
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)  # Gradiente de la capa oculta

    # Actualización de pesos y bias
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Mostrar error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Error: {np.mean(np.abs(error))}")

# Predicción final
print("\nResultados después del entrenamiento:")
for i in range(len(X)):
    hidden_output = sigmoid(np.dot(X[i], W1) + b1)
    final_output = sigmoid(np.dot(hidden_output, W2) + b2)
    print(f"Entrada: {X[i]}, Salida predicha: {final_output.round()}")
