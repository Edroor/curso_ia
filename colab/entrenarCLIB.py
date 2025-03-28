from transformers import CLIPTokenizer, CLIPTextModel
import torch

# Paso 1: Cargar el tokenizador y el modelo preentrenado de CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Paso 2: Definir el texto de entrada
texto = "Una imagen de un gato en el tejado."

# Paso 3: Tokenizar el texto, convirtiéndolo en tensores adecuados para el modelo
inputs = tokenizer(texto, return_tensors="pt")

# Paso 4: Obtener las representaciones (embeddings) del texto
# Desactivamos el cálculo de gradientes ya que solo hacemos inferencia
with torch.no_grad():
    outputs = model(**inputs)

# 'outputs.last_hidden_state' contiene los embeddings de cada token
# 'outputs.pooler_output' ofrece una representación agregada del texto
print("Forma de los embeddings:", outputs.last_hidden_state.shape)
print("Forma del pooler output:", outputs.pooler_output.shape)

# Liberar objetos grandes
del model
del tokenizer
del inputs
del outputs

# Recolectar basura y limpiar cache de GPU
import gc
gc.collect()
torch.cuda.empty_cache()
