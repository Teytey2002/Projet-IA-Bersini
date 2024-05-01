import torch
from torchvision import datasets, models, transforms

# Vérifiez si CUDA est disponible et définissez la variable device en conséquence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargez le modèle pré-entraîné ResNet50
model = models.resnet50(pretrained=True)

# Geler tous les paramètres du modèle
for param in model.parameters():
    param.requires_grad = False

# Remplacer la dernière couche pour correspondre au nombre de classes dans le nouveau jeu de données
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: chiens et chats

model = model.to(device)

# Définir la fonction de perte et l'optimiseur
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Normaliser les images de chiens et de chats pour correspondre aux images sur lesquelles ResNet a été formé
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Charger les images de chiens et de chats
image_datasets = {x: datasets.ImageFolder('/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet/dogs-vs-cats/'+x, data_transforms[x]) for x in ['train', 'val']}

# Définir l'ensemble de données de test et le chargeur de données
test_dataset = datasets.ImageFolder('/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet/dogs-vs-cats', data_transforms['val'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

# Mettre le modèle en mode d'évaluation
model.eval()

correct = 0
total = 0

# Pas de calcul de gradient nécessaire pendant l'évaluation
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Précision du modèle sur les données de test: {}%'.format(100 * correct / total))