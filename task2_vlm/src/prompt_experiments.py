from vlm_inference import generate_report, PROMPTS
from utils import get_test_dataset
from torchvision import transforms

dataset = get_test_dataset()

keywords_pneumonia = ["opacity", "infiltrate", "consolidation"]
keywords_normal = ["clear", "normal", "no abnormality"]

def keyword_score(report, keywords):
    report = report.lower()
    return sum([1 for k in keywords if k in report])

for i in range(15):
    img, label = dataset[i]
    img_pil = transforms.ToPILImage()(img)

    report = generate_report(img_pil, PROMPTS["structured"])

    if label.item() == 1:
        score = keyword_score(report, keywords_pneumonia)
    else:
        score = keyword_score(report, keywords_normal)

    print(f"\nImage {i} | GT: {label.item()} | Keyword Score: {score}")
    print(report)
