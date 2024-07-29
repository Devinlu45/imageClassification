import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

# Define transformations for training and testing
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 dataset
training = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
testing = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(training, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testing, batch_size=64, shuffle=False, num_workers=2)

# Loading a resnet model
model = models.resnet18(pretrained=True)

num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10) 

# Transfer the model to GPU for faster processing
device = torch.device('cuda')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Actual training
for epoch in range(10): 
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        opt.step()

print('Finished Training')

torch.save(model.state_dict(), 'cifar10_resnet18.pth')