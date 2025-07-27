import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 2. Definir el dispositivo (GPU si está disponible, si no CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# 3. Preparar los datos de prueba

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_data, batch_size=8, shuffle=True)

# 4. Mapeo de etiquetas para entender los resultados
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


# 1. Crear una instancia del modelo
model = NeuralNetwork().to(device)

# 2. Cargar los pesos guardados desde el archivo
try:
    model.load_state_dict(torch.load("model.pth"))
    print("Modelo 'model.pth' cargado exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'model.pth'.")
    print(
        "Asegúrate de que el modelo entrenado esté en la misma carpeta que este script."
    )
    exit()

# 3. Poner el modelo en modo de evaluación
model.eval()


# Obtener un lote de imágenes del cargador de datos de prueba
images, labels = next(iter(test_loader))

# Mover las imágenes a nuestro dispositivo
images = images.to(device)

# Realizar la predicción con el modelo
with torch.no_grad():
    predictions_logits = model(images)
    predicted_indices = predictions_logits.argmax(1)

print("\n--- Mostrando predicciones del modelo para un lote de prueba ---")

# Configurar el gráfico
plt.figure(figsize=(15, 7))
plt.suptitle("Análisis de Predicciones del Modelo", fontsize=16)

# Iterar sobre las imágenes del lote y mostrar los resultados
for i in range(len(images)):
    image = images[i].cpu()
    true_label = labels[i].item()
    predicted_label = predicted_indices[i].item()

    # Determinar el nombre de la clase real y la predicha
    true_name = labels_map[true_label]
    predicted_name = labels_map[predicted_label]

    # Preparar el subplot
    plt.subplot(2, 4, i + 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")

    # Poner el título con el resultado
    if true_label == predicted_label:
        title_color = "green"
        result_text = "ACIERTO"
    else:
        title_color = "red"
        result_text = "ERROR"

    plt.title(
        f"Real: {true_name}\nPred: {predicted_name}\n({result_text})", color=title_color
    )

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajustar para que el suptitle no se solape
plt.show()
