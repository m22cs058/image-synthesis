# Data processing
import numpy as np
import pandas as pd

# Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils

# Visualization
import matplotlib.pyplot as plt

rooty = "processed_celeba_small/"


dataset = dset.ImageFolder(root=rooty,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

need_length = 60000
waste = 29931
main_dataset,test_dataset=torch.utils.data.random_split(dataset,(need_length,waste))

dataloader = torch.utils.data.DataLoader(main_dataset, batch_size=128,
                                         shuffle=True, num_workers=2)

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),
            self._block(features_d*2, features_d*4, 4, 2, 1),
            self._block(features_d*4, features_d*8, 4, 2, 1),
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()

            )
        
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        
        )
    
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),
            self._block(features_g*16,features_g*8, 4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(features_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
                    
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
        
        )
    
    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d )):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

gen.train()
disc.train()

fixed_noise = torch.randn(64, 100, 1, 1, device=device)
img_list = []
iters = 0

discriminator_losses = []
generator_losses = []

for epoch in range(NUM_EPOCHS):
    total_discriminator_loss = 0
    total_generator_loss = 0
    
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((128, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_discriminator = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_discriminator.backward(retain_graph=True)
        opt_disc.step()
        
        total_discriminator_loss += loss_discriminator.item()
        
        output = disc(fake).reshape(-1)
        loss_generator = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_generator.backward()
        opt_gen.step()
        
        total_generator_loss += loss_generator.item()
        
        if (iters % 500 == 0) or (epoch == NUM_EPOCHS - 1):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        
        iters += 1
    
    discriminator_losses.append(total_discriminator_loss / len(dataloader))
    generator_losses.append(total_generator_loss / len(dataloader))
    
    print("Epoch {}: Discriminator Loss: {:.4f}, Generator Loss: {:.4f}".format(epoch, discriminator_losses[-1], generator_losses[-1]))
    
    # Save the grid of generated images
    grid = img_list[-1]
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('plots/faces/generated_images_epoch_{}.png'.format(epoch))
    plt.close()
    
# Plot discriminator and generator losses
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.plot(generator_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_face.png')

torch.save(gen.state_dict(), 'models/G_FACES.pt')
