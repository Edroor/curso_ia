import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generación de datos
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)  # Función cuadrática con ruido

# Ajuste de regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)

# Visualización
plt.scatter(x, y, color="blue", label="Datos")
plt.plot(x, y_pred, color="red", linewidth=2, label="Regresión Lineal")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Coeficientes de la regresión
lin_reg.coef_, lin_reg.intercept_
