import torch
import torch.nn as nn
from torchvision import transforms
from utils import get_test_dataset
from vlm_inference import generate_report, PROMPTS

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Define SimpleCNN -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ----- Load Model -----
model = SimpleCNN()
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.to(device)
model.eval()

dataset = get_test_dataset()

for i in range(10):
    img, label = dataset[i]
    
    input_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    img_pil = transforms.ToPILImage()(img)

    report = generate_report(img_pil, PROMPTS["structured"])

    print(f"\n===== Image {i} =====")
    print(f"Ground Truth : {label.item()}")
    print(f"CNN Prediction: {pred}")
    print(f"\nVLM Report:\n{report}")
