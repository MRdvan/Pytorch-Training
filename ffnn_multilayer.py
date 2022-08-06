import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


"""
Step 1: Load Dataset
Step 2: Make Dataset Iterable
Step 3: Create Model Class
Step 4: Instantiate Model Class
Step 5: Instantiate Loss Class
Step 6: Instantiate Optimizer Class
Step 7: Train Model
"""

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

#%%
"""Step 1"""

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Hyper-parameters for data
batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

"""Step 2"""
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
#%%
"""Step 3"""

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
       
        
        # Linear function 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Non-linearity 1
        self.relu1 = nn.ReLU()#no parameters to update.

        # Linear function 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()#no parameters to update.

        # Linear function 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 3
        self.relu3 = nn.ReLU()

        # Linear function 4 (readout): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim) 
        

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)
        
        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3
        out = self.fc3(out)
        # Non-linearity 3
        out = self.relu3(out)

        # Linear function 4 (readout)
        out = self.fc4(out)
        return out
        return out


"""Step 4"""
input_dim = 28*28 # 28x28
# too small of a hidden size would mean there would be insufficient model capacity to predict competently.
# In layman terms, too small of a capacity implies a smaller brain capacity so no matter 
# how many training samples you give it, it has a maximum capacity in terms of its predictive power.
hidden_dim = 100 # not too small and not too high its picked by the hardness of problem
output_dim = 10
model = FeedforwardNeuralNetModel(input_dim, hidden_dim,output_dim)

#%%
"""Step 5"""
# Cross entropy for loss function 
# 1. Computes softmax (logistic/softmax function)
# 2. Computes cross entropy
criterion = nn.CrossEntropyLoss()


"""Step 6"""
# Too small and the algorithm learns too slowly,too large and the algorithm learns too fast resulting in instabilities.
learning_rate = 0.1
#parameters = parameters - learning_rate * parameters_gradients
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
 

# Type of parameter object
print(model.parameters())

# Length of parameters
print(len(list(model.parameters())))

# FC 1 Parameters 
print(list(model.parameters())[0].size()) #W1 (100,784)

# FC 1 Bias Parameters
print(list(model.parameters())[1].size()) #B1 (100,1)

# FC 2 Parameters
print(list(model.parameters())[2].size()) #W2 (10,100)

# FC 2 Bias Parameters
print(list(model.parameters())[3].size()) #B2 (10,1)

#%%
"""
** 7 steps process for training models **

1-Convert inputs/labels to tensors with gradients
2-Clear gradient buffets
3-Get output given inputs
4-Get loss
5-Get gradients w.r.t. parameters
6-Update parameters using gradients
    parameters = parameters - learning_rate * parameters_gradients
7-REPEAT

"""

# Train the model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        images = images.view(-1, 28*28).requires_grad_().to(device) #[28,28],(gradyanlar hesaplanacak)
        labels = labels.to(device)
       

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

#Accuracy of the network on the 10000 test images: 92.15 % (one hidden layer + sigmoid)
#Accuracy of the network on the 10000 test images: 96.43 % (two hidden layer + relu)
#Accuracy of the network on the 10000 test images: 96.87 % (three hidden layer + relu) 