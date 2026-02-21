import torch
from torchvision import transforms
from medmnist import PneumoniaMNIST

def get_test_dataset(split='test'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = PneumoniaMNIST(
        split=split,
        transform=transform,
        download=True
    )
    
    return dataset
