from dataset import get_dataloaders
from models import SimpleCNN, get_resnet18
from train import train_model

MODEL_TYPE = "resnet"  # change to "simple" or "resnet"

if MODEL_TYPE == "simple":
    train_loader, val_loader, _ = get_dataloaders(batch_size=64, model_type="simple")
    model = SimpleCNN()
    save_path = "models/simple_cnn.pth"
    experiment_name = "simple_cnn"

elif MODEL_TYPE == "resnet":
    train_loader, val_loader, _ = get_dataloaders(batch_size=16, model_type="resnet")
    model = get_resnet18()
    save_path = "models/resnet18.pth"
    experiment_name = "resnet18"

train_model(model,
            train_loader,
            val_loader,
            epochs=15,
            lr=1e-4,
            save_path=save_path,
            experiment_name=experiment_name)
