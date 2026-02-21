import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from torchvision import transforms
from PIL import Image
from utils import get_test_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

print("Loading VLM...")
processor = LlavaProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

model.eval()

# Prompt versions
PROMPTS = {
    "basic": "Describe this chest X-ray image.",
    
    "radiologist": """You are a medical radiologist.
Analyze the chest X-ray image and provide:
1. Key observations
2. Possible diagnosis
3. Clinical impression.""",
    
    "structured": """You are an expert thoracic radiologist.
Carefully analyze this chest X-ray.
Focus on:
- Lung opacity
- Consolidation
- Infiltrates
- Pleural abnormalities
Provide a concise diagnostic report."""
}

def generate_report(image, prompt):
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200
        )

    return processor.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    dataset = get_test_dataset()

    for i in range(5):
        img, label = dataset[i]
        img_pil = transforms.ToPILImage()(img)

        print(f"\n===== Image {i} | GT: {label.item()} =====")

        for key, prompt in PROMPTS.items():
            report = generate_report(img_pil, prompt)
            print(f"\n--- Prompt: {key} ---")
            print(report)
