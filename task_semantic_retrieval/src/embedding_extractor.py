import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from dataset_loader import get_dataloader
from config import DEVICE, BATCH_SIZE, EMBEDDING_PATH, LABEL_PATH


def extract_embeddings():

    print("Device:", DEVICE)

    # Ensure embeddings directory exists
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = model.to(DEVICE)
    model.eval()

    dataset, loader = get_dataloader(BATCH_SIZE)

    all_embeddings = []
    all_labels = []

    print("Extracting embeddings...")

    with torch.no_grad():
        for images, labels in tqdm(loader):

            # Convert grayscale (1 channel) → 3 channel
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            inputs = processor(
                images=images,
                return_tensors="pt"
            )

            pixel_values = inputs["pixel_values"].to(DEVICE)

            # Forward pass through vision encoder
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output

            # Projection layer
            image_features = model.visual_projection(pooled_output)

            # Normalize (cosine similarity)
            image_features = torch.nn.functional.normalize(
                image_features, p=2, dim=1
            )

            embeddings = image_features.cpu().numpy()

            all_embeddings.append(embeddings)
            all_labels.append(labels.numpy().reshape(-1))

    embeddings = np.vstack(all_embeddings).astype("float32")
    labels = np.concatenate(all_labels, axis=0)

    np.save(EMBEDDING_PATH, embeddings)
    np.save(LABEL_PATH, labels)

    print("\nEmbeddings saved successfully.")
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
    print("Saved at:", EMBEDDING_PATH)


if __name__ == "__main__":
    extract_embeddings()
