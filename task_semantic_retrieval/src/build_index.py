import faiss
import numpy as np
import os
from config import EMBEDDING_PATH, INDEX_PATH


def build_faiss_index():

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    embeddings = np.load(EMBEDDING_PATH)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    print("FAISS index saved at:", INDEX_PATH)


if __name__ == "__main__":
    build_faiss_index()
