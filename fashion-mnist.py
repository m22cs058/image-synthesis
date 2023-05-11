import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms

transform = transforms.ToTensor()
train = datasets.FashionMNIST(root = '.', train = True, download = True, 
                       transform = transform)
train_loader = torch.utils.data.DataLoader(train, batch_size = 256)

class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    # 100 -> 32 -> 64 -> 128 -> 784
    self.dense0 = nn.Linear(100, 32)
    self.dense1 = nn.Linear(32, 64)
    self.dense2 = nn.Linear(64, 128)
    self.dense3 = nn.Linear(128, 784)
    self.dropout = nn.Dropout(0.3)

  def forward(self, X):
    X = self.dropout(F.leaky_relu(self.dense0(X), 0.2))
    X = self.dropout(F.leaky_relu(self.dense1(X), 0.2))
    X = self.dropout(F.leaky_relu(self.dense2(X), 0.2))
    X = torch.tanh(self.dense3(X))
    X = X.view(X.shape[0], 28, 28)
    return X
  

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    # 784 -> 128 -> 64 -> 32 -> 1
    self.dense0 = nn.Linear(784, 128)
    self.dense1 = nn.Linear(128, 64)
    self.dense2 = nn.Linear(64, 32)
    self.dense3 = nn.Linear(32, 1)
    self.dropout = nn.Dropout(0.3)

  def forward(self, X):
    X = X.view(X.shape[0], 28 * 28)
    X = self.dropout(F.leaky_relu(self.dense0(X), 0.2))
    X = self.dropout(F.leaky_relu(self.dense1(X), 0.2))
    X = self.dropout(F.leaky_relu(self.dense2(X), 0.2))
    X = self.dense3(X)
    return X
  
G = Generator()
D = Discriminator()

G_optimizer = optim.Adam(G.parameters(), lr = 0.002)
D_optimizer = optim.Adam(D.parameters(), lr = 0.002)

criterion = nn.BCEWithLogitsLoss()
     
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

G.to(device)
D.to(device)

D_losses = []
G_losses = []

for epoch in range(500):
    D_running_loss = 0
    G_running_loss = 0
    
    for i, (imagens_real, _) in enumerate(train_loader):
        batch_size = imagens_real.size(0)
        imagens_real = imagens_real * 2 - 1
        imagens_real = imagens_real.to(device)

        # Training the generator
        G_optimizer.zero_grad()
        noise = np.random.uniform(low=-1., high=1., size=(batch_size, 100))
        noise = torch.from_numpy(noise).float().to(device)
        input_false = G.forward(noise)
        output_false = D.forward(input_false)
        labels_false = torch.ones(batch_size).to(device)
        G_loss = criterion(output_false.view(*labels_false.shape), labels_false)
        G_loss.backward()
        G_optimizer.step()

        # Training the discriminator
        D_optimizer.zero_grad()
        outputs_real = D.forward(imagens_real)
        labels_real = (torch.ones(batch_size) * 0.9).to(device)
        D_loss_real = criterion(outputs_real.view(*labels_real.shape), labels_real)

        noise = np.random.uniform(-1., 1., (batch_size, 100))
        noise = torch.from_numpy(noise).float().to(device)
        input_false = G.forward(noise)
        output_false = D.forward(input_false)
        labels_false = torch.zeros(batch_size).to(device)
        D_loss_false = criterion(output_false.view(*labels_false.shape), labels_false)

        D_loss = D_loss_real + D_loss_false
        D_loss.backward()
        D_optimizer.step()

        D_running_loss += D_loss.item()
        G_running_loss += G_loss.item()

    D_running_loss /= len(train_loader)
    G_running_loss /= len(train_loader)
    D_losses.append(D_running_loss)
    G_losses.append(G_running_loss)
    
    print('EPOCH {:03d} D_Loss {:03.6f} - G_Loss {:03.6f}'.format(epoch + 1, D_running_loss, G_running_loss))
    
    if epoch % 5 == 0:
        fig, ax = plt.subplots(1, 5, figsize=(10, 5))
        for i in range(5):
            ax[i].imshow(input_false.cpu().detach().numpy()[i].reshape(28, 28), cmap='gray')
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
        plt.savefig('plots/fmnist/epoch_{}.png'.format(epoch))
        plt.close()

# Plotting generator and discriminator losses
plt.plot(D_losses, label='Discriminator Loss')
plt.plot(G_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses_plot_fmnist.png')
plt.show()


torch.save(G.state_dict(), 'models/G_FMNIST.pt')