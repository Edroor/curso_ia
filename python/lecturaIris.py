import matplotlib.pyplot as plt
from PIL import Image

imagen = Image.open("Iris_setosa.jpg")

# Mostrar la imagen original
plt.imshow(imagen)
plt.title("Iris setosa")
plt.axis("off")
plt.show()
