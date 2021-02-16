import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 4
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_datasets = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform = transform)
test_datasets = torchvision.datasets.CIFAR10(root='data', train=False, download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size= batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size,shuffle= False)

classes = ('plane', 'car', 'bird','cat', 'deer', 'dog','frog','horse','ship','truck')


class ConvNet(nn.Module):
      def __init__(self):
          super(ConvNet, self).__init__()
          ## formula = (w-f +2p)/s +1 w=input, f- filter size , p=padding, stride =1
          self.conv1 = nn.Conv2d(3,6,5) # 3 input channel RGB, 6 o/p channel, 5- filters
          self.pool = nn.MaxPool2d(2,2)
          self.conv2 = nn.Conv2d(6,16,5) ## for con2d o/p pf conv1 size is 6, 16 output size, 5 filters
          self.fc1 = nn.Linear(16*5*5, 120) # 120 can be any
          self.fc2 = nn.Linear(120,84) # 84 can be anything
          self.fc3 = nn.Linear(84,10) # 84 from prebious and 10 classes
          # image size reduces by 2 while applying padding/ Hence last image will a dimesnsion 0f 16 * 5* 5
      def forward(self,x):
          x = self.pool(F.relu(self.conv1(x)))
          x= self.pool(F.relu(self.conv2(x)))
          x = x.view(-1,16*5*5)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimzer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i,(images, lables) in enumerate(train_loader):

        images = images.to(device)
        lables = lables.to(device)

        # forwaed
        output = model(images)
        loss = criterion(output, lables)

        #backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss ={loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
