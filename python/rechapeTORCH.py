import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# Cargar una imagen
imagen = Image.open("8955.jpg")
imagen.show()
# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensiona la imagen
    transforms.ToTensor(),          # Convierte la imagen a tensor
])

# Aplicar transformaciones
tensor_imagen = transform(imagen)
print(tensor_imagen.shape)  # Ver dimensiones del tensor

# Crear la figura con dos subgráficos
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Mostrar la imagen original
axs[0].imshow(imagen)
axs[0].set_title("Imagen Original")
axs[0].axis("off")

# Mostrar la imagen transformada
axs[1].imshow(tensor_imagen.permute(1, 2, 0))  # Cambia el orden de los canales
axs[1].set_title("Imagen Transformada (Tensor)")
axs[1].axis("off")

# Mostrar la figura
plt.show()
