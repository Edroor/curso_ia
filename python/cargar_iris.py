import pandas as pd
from sklearn.datasets import load_iris

# Cargar el dataset Iris
iris = load_iris()

# Crear un DataFrame con las características
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Agregar la columna de la clase y convertirla en etiquetas legibles
df['target'] = iris.target
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Mostrar las primeras filas
print(df.head())
print(df.shape)
print(df.to_string())
