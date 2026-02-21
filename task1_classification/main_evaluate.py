from dataset import get_dataloaders
from models import SimpleCNN, get_resnet18
from evaluate import evaluate_model

MODEL_TYPE = "resnet"  # change accordingly

if MODEL_TYPE == "simple":
    _, _, test_loader = get_dataloaders(batch_size=64, model_type="simple")
    model = SimpleCNN()
    model_path = "models/simple_cnn.pth"
    experiment_name = "simple_cnn"

elif MODEL_TYPE == "resnet":
    _, _, test_loader = get_dataloaders(batch_size=16, model_type="resnet")
    model = get_resnet18()
    model_path = "models/resnet18.pth"
    experiment_name = "resnet18"

evaluate_model(model, test_loader, model_path, experiment_name)
