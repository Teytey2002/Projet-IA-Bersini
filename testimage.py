import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from cnn import Net
import os
from PIL import Image

# Définir le chemin vers le modèle et les classes
PATH = './model/model_20240424_094857_21'
classes = {0: 'cat', 1: 'dog'}
class_names = ['cat', 'dog']  # Noms des classes dans l'ordre de l'index

# Définir la transformation
transform = transforms.Compose([
    transforms.Resize(300),  # Redimensionner pour correspondre à l'entrée attendue du modèle
    transforms.CenterCrop(300),  # Rogner au centre
    transforms.ToTensor(),  # Convertir en tenseur
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Charger le modèle
net = Net()
net.load_state_dict(torch.load(PATH))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net.eval()  # Mettre le modèle en mode évaluation

# Statistiques
correct = 0
total = 0

# Fonction pour tester une image
def test_single_image(image_path):
    global correct, total
    image = Image.open(image_path)  # Charger l'image avec PIL
    image_tensor = transform(image).unsqueeze(0).to(device)  # Appliquer la transformation et ajouter une dimension batch

    with torch.no_grad():  # Pas besoin de calculer les gradients
        output = net(image_tensor)
        _, predicted = torch.max(output, 1)  # Obtenir l'index de la classe prédite
        predicted_class = classes[predicted.item()]

    # Obtenir la classe réelle à partir du nom du dossier
    actual_class = os.path.basename(os.path.dirname(image_path))

    # Incrémenter le total et le nombre de corrects
    total += 1
    if predicted_class == actual_class:
        correct += 1

# Parcourir tout le dossier de test
test_dir = '/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet/dogs-vs-cats/test/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtrer pour les images
            image_path = os.path.join(root, file)
            test_single_image(image_path)

# Afficher l'exactitude
if total > 0:
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f} %')
else:
    print("No images to test.")
