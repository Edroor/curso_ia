import torch

# Crear un tensor de 3x3 con números aleatorios
x = torch.rand(3, 3)
print(x)

# Crear un tensor con valores específicos
y = torch.tensor([[1, 2, 0], [3, 4, 0],[0,0,0]])
print(y)

# Operaciones básicas
suma = x + y
producto = x * y
print(suma, producto)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_gpu = x.to(device)  # Mueve el tensor a la GPU
print(x_gpu)
