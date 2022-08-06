import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
#%%
# Same as linear regression! 
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 28*28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)

#%%

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# 1. Computes softmax (logistic/softmax function)
# 2. Computes cross entropy

#parameters = parameters - learning_rate * parameters_gradients
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Type of parameter object
print(model.parameters())

# Length of parameters
print(len(list(model.parameters())))

# FC 1 Parameters 
print(list(model.parameters())[0].size())

# FC 1 Bias Parameters
print(list(model.parameters())[1].size())

#%%
"""
**7step process for training models**
1-Convert inputs/labels to tensors with gradients
2-Clear gradient buffets
3-Get output given inputs
4-Get loss
5-Get gradients w.r.t. parameters
6-Update parameters using gradients
7-parameters = parameters - learning_rate * parameters_gradients
8-REPEAT

"""

# Train the model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        images = images.view(-1, 28*28).requires_grad_().to(device) #[28,28],(gradyanlar hesaplanacak)
        labels = labels.to(device)
        model.to(device)

        # Clear gradients w.r.t. parameters (sıfırlama yapılıyor)
        optimizer.zero_grad()

        # Forward pass to get output/logits (networkte ileri yayılım hesabı)
        outputs = model(images)

        #yapılan error hesabı için loss function: softmax --> cross entropy loss
        loss = criterion(outputs, labels) 

        #kaydedilen gradyanların getirilmesi
        loss.backward() 

        #gradyanlara göre parametrelerin güncellenmesi
        optimizer.step() 

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
#%%

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
