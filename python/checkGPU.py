import torch
print(torch.cuda.is_available())  # Debe devolver True si CUDA está activo
print(torch.cuda.device_count())  # Número de GPUs disponibles
print(torch.cuda.get_device_name(0))  # Nombre de la GPU
