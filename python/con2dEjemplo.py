import numpy as np
import matplotlib.pyplot as plt

def conv2d(image, kernel, stride=1, padding=0):
    # Aplicar padding a la imagen
    if padding > 0:
        image_padded = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        image_padded = image
    
    # Dimensiones de la imagen y del kernel
    h, w = image_padded.shape
    kh, kw = kernel.shape
    
    # Calcular dimensiones de salida
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    
    # Recorrer la imagen para aplicar el kernel
    for i in range(out_h):
        for j in range(out_w):
            # Extraer la región de la imagen correspondiente
            region = image_padded[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output

def relu(x):
    return np.maximum(0, x)

# Crear una imagen de ejemplo (por simplicidad, usamos una imagen aleatoria de 28x28)
image = np.random.rand(28, 28)

# Crear un kernel (filtro) de 3x3, similar al usado en la capa Conv2D
kernel = np.random.rand(3, 3)

# Aplicar la convolución
conv_output = conv2d(image, kernel, stride=1, padding=0)

# Aplicar la función de activación ReLU
activated_output = relu(conv_output)

# Mostrar resultados
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Imagen Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Salida de Convolución")
plt.imshow(conv_output, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Después de ReLU")
plt.imshow(activated_output, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
