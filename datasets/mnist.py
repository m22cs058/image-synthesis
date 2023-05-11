from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
import torch

def dataloader():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
    ])

    batch_size = 32
    data_loader = torch.utils.data.DataLoader(MNIST('./data', train=True, download=True, transform=transform),
                                            batch_size=batch_size, shuffle=True)
    return dataloader