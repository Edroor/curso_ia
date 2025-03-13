from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Cargar el dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convertir las matrices de entrada a DataFrame para visualizarlas mejor
df_X_train = pd.DataFrame(X_train, columns=iris.feature_names)
df_X_test = pd.DataFrame(X_test, columns=iris.feature_names)

# Mostrar los datos de entrenamiento
print("Datos de Entrenamiento (X_train):")
print(df_X_train.to_string(index=False))

print("\nDatos de Entrenamiento (y_train):")
print(y_train)

# Mostrar los datos de prueba
print("\nDatos de Prueba (X_test):")
print(df_X_test.to_string(index=False))

print("\nDatos de Prueba (y_test):")
print(y_test)
