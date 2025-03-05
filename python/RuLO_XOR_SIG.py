#Perceptrón Multicapa (MLP) para XOR usando la función de activación ReLU en la capa oculta y sigmoide en la salida.

import numpy as np

# Función de activación ReLU y su derivada
def relu(x):
    return np.maximum(0, x)  # Activa solo valores positivos

def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # Derivada de ReLU (1 si x>0, 0 si x<=0)

# Función sigmoide (para la capa de salida)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Datos de entrada (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # Salidas esperadas de XOR

# Inicialización de pesos con valores pequeños
np.random.seed(42)
input_size = 2
hidden_size = 8  # Aumentamos a 4 neuronas en la capa oculta
output_size = 1

W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))  # Pesos entrada -> oculta
b1 = np.random.uniform(-0.5, 0.5, (1, hidden_size))  # Bias de la capa oculta
W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))  # Pesos oculta -> salida
b2 = np.random.uniform(-0.5, 0.5, (1, output_size))  # Bias de la capa de salida

# Parámetros de entrenamiento
epochs = 20000  # Más iteraciones para garantizar el aprendizaje
learning_rate = 0.05  # Reducimos la tasa de aprendizaje para estabilidad

# Entrenamiento con retropropagación
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = relu(hidden_input)  # Se usa ReLU en la capa oculta

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)  # Se usa sigmoide en la capa de salida

    # Cálculo del error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)  # Gradiente de la capa de salida
    d_hidden = d_output.dot(W2.T) * relu_derivative(hidden_output)  # Gradiente de la capa oculta

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
    hidden_output = relu(np.dot(X[i], W1) + b1)
    final_output = sigmoid(np.dot(hidden_output, W2) + b2)
    print(f"Entrada: {X[i]}, Salida predicha: {final_output.round()}")
