import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import PneumoniaMNIST


def get_dataloaders(batch_size=32, model_type="simple"):

    if model_type == "simple":
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    elif model_type == "resnet":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    train_dataset = PneumoniaMNIST(split="train", transform=train_transform, download=True)
    val_dataset = PneumoniaMNIST(split="val", transform=test_transform, download=True)
    test_dataset = PneumoniaMNIST(split="test", transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
