from medmnist import PneumoniaMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = PneumoniaMNIST(
        split="test",
        transform=transform,
        download=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader
