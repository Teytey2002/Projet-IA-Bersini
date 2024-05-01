import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models #ddd
from cnn import Net

#Script to test 1 model

# Define the path to the model and the classes
#PATH = './model/model_20240501_113312_13'
PATH = './model/best_model_fromscratch'
classes = {0: 'cat', 1: 'dog'}

# Définir la transformation
# transform = transforms.Compose([
    # transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
transform = transforms.Compose([
    transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the test dataset and the model and GPU
data_dir = '/mnt/c/Users/theod/OneDrive/Documents/ULB/Ma0/Q2/INFO-H-410_TechniquesOfArtificialIntelligence/Projet_IA/dogs-vs-cats'
dataset = torchvision.datasets.ImageFolder(data_dir+'/test', transform=transform)
testloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
net = models.resnet50(pretrained=False)  #ddd
net = Net()
# Ajuster la dernière couche du modèle pour avoir 2 sorties au lieu de 1000
#num_features = net.fc.in_features   #ddd
#net.fc = torch.nn.Linear(num_features, 2)  # 2 classes: chiens et chats #ddd
net.load_state_dict(torch.load(PATH))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)  # Move the images and labels to the GPU
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')