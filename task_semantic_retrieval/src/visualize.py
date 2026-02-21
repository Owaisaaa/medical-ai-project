import matplotlib.pyplot as plt
from dataset_loader import get_dataloader
from search import image_to_image
from config import TOP_K

def visualize(query_index):

    dataset, _ = get_dataloader()
    retrieved = image_to_image(query_index, TOP_K)

    query_img, query_label = dataset[query_index]

    plt.figure(figsize=(12,3))

    plt.subplot(1, TOP_K+1, 1)
    plt.imshow(query_img.squeeze(), cmap="gray")
    plt.title(f"Query\nLabel: {query_label.item()}")
    plt.axis("off")

    for i, idx in enumerate(retrieved):
        img, label = dataset[idx]
        plt.subplot(1, TOP_K+1, i+2)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"Label: {label.item()}")
        plt.axis("off")

    plt.show()

if __name__ == "__main__":
    visualize(query_index=10)
