import faiss
import numpy as np
from config import INDEX_PATH, EMBEDDING_PATH, TOP_K


def load_index():
    return faiss.read_index(INDEX_PATH)


def image_to_image(query_index, k=TOP_K):
    index = load_index()
    embeddings = np.load(EMBEDDING_PATH)

    query_vector = embeddings[query_index].reshape(1, -1)

    distances, indices = index.search(query_vector, k)

    return indices[0]
    