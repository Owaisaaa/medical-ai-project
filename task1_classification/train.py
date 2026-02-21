import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import json


def evaluate_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model, train_loader, val_loader,
                epochs=15,
                lr=1e-4,
                save_path="models/best_model.pth",
                experiment_name="experiment"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        running_loss = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("✔ Best model saved.")

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"reports/{experiment_name}_loss_curve.png")
    plt.close()

    # Save loss values
    with open(f"reports/{experiment_name}_loss_values.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses
        }, f)

    print("✔ Loss curves saved.")
    print("Training complete.")
