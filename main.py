import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from CustomLinearLayer import CustomLinearLayer

# Hyper-parameters

num_epochs = 2
learning_rate = 0.01
input_size = 784  # 28*28
batch_size = 100
hidden_size = 500
classes = 10

# Data Preparation

train_data = torchvision.datasets.MNIST('\data', transform=transform.ToTensor(), train=True, download=True)
test_data = torchvision.datasets.MNIST('\data', transform=transform.ToTensor(), train=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Model Building

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, number_classes):
        super(MLP, self).__init__()
        self.number_classes = number_classes
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.L1 = nn.Linear(input_size, hidden_size)
        #self.L1 = CustomLinearLayer(input_size, hidden_size)
        self.Relu = nn.ReLU()
        self.L2 = nn.Linear(hidden_size, number_classes)
        #self.L2 = CustomLinearLayer(hidden_size, number_classes)

    def forward(self, x):
        out = self.L1(x)
        out = self.Relu(out)
        out = self.L2(out)

        return out


Model = MLP(input_size, hidden_size, classes)

# Loss & optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), learning_rate)

# Model Training

num_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, 28 * 28)

        # Forward Pass

        output = Model(image)
        loss = criterion(output, label)

        # Backward Pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epochs[{epoch + 1}/{num_epochs}]/Step[{i + 1}/{num_step}],Loss = {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for image,label in test_loader:

        image = image.reshape(-1, 28*28)
        output = Model(image)

        # Value_Index

        _, prediction = torch.max(output,-1)
        n_sample += label.shape[0]
        n_correct += (prediction == label).sum().item()

accuracy = 100 * n_correct / n_sample

print(f"Accuracy = {accuracy}")


#Accuracy = 93.22