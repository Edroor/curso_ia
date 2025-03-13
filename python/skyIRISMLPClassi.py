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


# Crear y entrenar la red neuronal
# En este caso, definimos dos capas ocultas de 10 neuronas cada una, función ReLU, y optimizador Adam.
mlp_clf = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu', solver='adam',
                        max_iter=500, random_state=42)
mlp_clf.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred = mlp_clf.predict(X_test)

# Mostrar los datos de prueba
print("\nDatos de Prueba (X_test):")
print(df_X_test.to_string(index=False))

print("\nDatos de Prueba (y_test):")
print(y_test)

# Imprimir reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualizar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Clasificación Iris")
plt.show()
