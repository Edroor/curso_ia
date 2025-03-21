import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x

def mse(y, y_hat):
    return 0.5 * np.mean((y_hat - y.reshape(y_hat.shape)) ** 2)

def grad_mse(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)

class Perceptron():
    def __init__(self, inputs, outputs, activation, loss, grad_loss):
        np.random.seed(42)  # Asegurar reproducibilidad
        inputs += 1  # Para el sesgo
        self.w = np.random.normal(loc=0.0, 
                                  scale=np.sqrt(2 / (inputs + outputs)), 
                                  size=(inputs, outputs))
        self.ws = []
        self.activation = activation
        self.loss = loss
        self.grad_loss = grad_loss
        self.losses = []  # Lista para almacenar la pérdida

    def __call__(self, w, x):
        return self.activation(x @ w)

    def fit(self, x, y, epochs, lr, batch_size=None, verbose=True, log_each=1):
        if batch_size is None:
            batch_size = len(x)
        x = np.c_[np.ones(len(x)), x]  # Agregar columna de sesgo

        for epoch in range(1, epochs + 1):
            # Mini-Batch Gradient Descent con np.array_split
            x_batches = np.array_split(x, len(x) // batch_size)
            y_batches = np.array_split(y, len(y) // batch_size)

            for _x, _y in zip(x_batches, y_batches):
                y_hat = self(self.w, _x)
                l = self.loss(_y, y_hat)
                dldh = self.grad_loss(_y, y_hat)
                dldw = _x.T @ dldh
                self.w -= lr * dldw

            self.ws.append(self.w.copy())
            self.losses.append(l)

            if verbose and not epoch % log_each:
                print(f"Epoch {epoch}/{epochs} Loss {l}")

    def predict(self, x):
        x = np.c_[np.ones(len(x)), x]
        return self(self.w, x)

    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.losses)), self.losses)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss over Epochs")
        plt.show()

    def plot_predictions(self, x, y):
        y_pred = self.predict(x)
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Actual Data", color='blue')
        ax.plot(x, y_pred, label="Predicted Regression", color='red')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.set_title("Actual vs Predicted Regression")
        plt.show()

class LinearRegression(Perceptron):
    def __init__(self, inputs, outputs=1):
        super().__init__(inputs, outputs, linear, mse, grad_mse)

# Datos de entrenamiento
np.random.seed(42)
x = np.random.rand(100)
X = x.reshape(-1, 1)
y = 2 * x + (np.random.rand(100) - 0.5) * 0.5  # y = 2x + ruido

# Crear modelo y entrenar
model = LinearRegression(inputs=1, outputs=1)
epochs, lr = 50, 0.01
model.fit(X, y, epochs, lr, log_each=10)

# Graficar la pérdida en una figura separada
model.plot_loss()

# Graficar y vs predicción en otra figura
model.plot_predictions(x, y)
