# Reimportar las librerías tras el reinicio del estado de ejecución
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Generación de datos
m = 100
np.random.seed(42)
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)  # Función cuadrática con ruido

# Crear y entrenar una red neuronal con Scikit-Learn
mlp = MLPRegressor(hidden_layer_sizes=(16, 16), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(x, y.ravel())

# Predicción con la red neuronal
x_new = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred_mlp = mlp.predict(x_new)

# Visualización de los resultados
plt.scatter(x, y, color="blue", label="Datos")
plt.plot(x_new, y_pred_mlp, color="green", linewidth=2, label="MLPRegressor (Red Neuronal)")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
