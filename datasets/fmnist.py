from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch

def dataloader():
    transform = transforms.ToTensor()
    train = FashionMNIST(root = './data', train = True, download = True, 
                        transform = transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 256)
    return train_loader