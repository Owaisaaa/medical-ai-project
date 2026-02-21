import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDING_PATH = os.path.join(BASE_DIR, "embeddings", "image_embeddings.npy")
LABEL_PATH = os.path.join(BASE_DIR, "embeddings", "labels.npy")
INDEX_PATH = os.path.join(BASE_DIR, "index", "faiss_index.bin")

BATCH_SIZE = 10
TOP_K = 5
