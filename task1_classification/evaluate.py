import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import os
import json


def evaluate_model(model, test_loader, model_path, experiment_name="experiment"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    failure_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

            for i in range(len(images)):
                if preds[i] != labels[i] and len(failure_images) < 10:
                    failure_images.append(
                        (images[i].cpu(), labels[i].cpu(), preds[i].cpu())
                    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print("\n===== Test Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    os.makedirs("reports", exist_ok=True)

    # Save metrics
    with open(f"reports/{experiment_name}_metrics.json", "w") as f:
        json.dump({
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc)
        }, f)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"reports/{experiment_name}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(f"reports/{experiment_name}_roc_curve.png")
    plt.close()

    # Failure Cases
    if len(failure_images) > 0:
        plt.figure(figsize=(12, 8))
        for i, (img, true, pred) in enumerate(failure_images):
            plt.subplot(2, 5, i + 1)
            img = img[0] if img.shape[0] == 1 else img.mean(dim=0)
            plt.imshow(img, cmap="gray")
            plt.title(f"T:{true.item()} P:{pred.item()}")
            plt.axis("off")

        plt.suptitle("Failure Cases")
        plt.savefig(f"reports/{experiment_name}_failure_cases.png")
        plt.close()

    print("✔ Evaluation artifacts saved in /reports/")
