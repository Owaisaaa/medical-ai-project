import numpy as np
from search import image_to_image
from config import TOP_K, LABEL_PATH


def precision_at_k(query_index, k=TOP_K):
    labels = np.load(LABEL_PATH)
    retrieved = image_to_image(query_index, k)

    query_label = labels[query_index]
    retrieved_labels = labels[retrieved]

    return np.sum(retrieved_labels == query_label) / k


def evaluate():
    labels = np.load(LABEL_PATH)
    scores = []

    for i in range(len(labels)):
        scores.append(precision_at_k(i))

    print(f"Mean Precision@{TOP_K}: {np.mean(scores):.4f}")


if __name__ == "__main__":
    evaluate()
