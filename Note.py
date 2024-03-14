import torch
import torchvision.datasets
from torchvision import datasets
import torchvision.transforms as transform
#import matplotlib.pyplot as plt
import torch.nn as nn
#from CustomLinearLayer import CustomLinearLayer

# Device Configuration

Device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Hyper Parameters

inputs_size = 784  # 28*28
hidden_size = 500
batch_size = 100
num_classes = 10
learning_rate = 0.01
num_epochs = 2

# MNIST Dataset

train_dataset = torchvision.datasets.MNIST("\data",
                                           train=True,
                                           download=True,
                                           transform=transform.ToTensor())

test_dataset = torchvision.datasets.MNIST('\data',
                                          train=True,
                                          transform=transform.ToTensor())

# Data Loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

example = iter(test_loader)

example_image, example_target = next(example)

# Fully connected the network with one hidden layer

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.l1 =CustomLinearLayer(input_size, hidden_size)
        #self.l1 = nn.Linear(input_size, hidden_size, bias=True )
        self.relu = nn.ReLU()
        self.l2 = CustomLinearLayer(hidden_size, num_classes)
        #self.l2 = nn.Linear(hidden_size, num_classes, bias=True)


    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


model = NeuralNetwork(inputs_size, hidden_size, num_classes).to(Device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):

        image = image.reshape(-1, 28 * 28).to(Device)
        label = label.to(Device)

        # Forward pass

        outputs = model(image)
        loss = criterion(outputs, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epochs [{epoch + 1} /{num_epochs}],step [{i + 1}/{n_total_step}], loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for image, label in test_loader:
        image = image.reshape(-1 , 28*28)
        outputs = model(image)

        # Value, index

        _, predictions = torch.max(outputs,1)
        n_samples += label.shape[0]
        n_correct += (predictions == label).sum().item()

acc = 100 * n_correct / n_samples

print(f"accuracy = {acc}")


import torch
import torch.nn as nn


class CustomLinearLayer(nn.Module):
    def __init__(self, input_size, output_size,batch_size=100):
        super(CustomLinearLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size))  # 500,784
        self.bias = nn.Parameter(torch.Tensor(batch_size,input_size, output_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Perform element-wise multiplication without summing up
        # bias = self.bias
        # weight = self.weight
        # xu = x.unsqueeze(2)
        # wu = self.weight.unsqueeze(0)
        result = x.unsqueeze(2) * self.weight.unsqueeze(0)

        #bias_unsqueeze = self.bias.unsqueeze(0)
        # Add the bias matrix
        #shape_result =result.shape
        result += self.bias
        #result = torch.transpose(result,result.shape[2],result.shape[1])
        result = torch.mean(result, dim=1)
        shape = result.shape

        return result


