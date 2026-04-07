import os
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

DATA_DIR = "data/"
CACHE_FILE = "data/cifar100_cache.pt"


def load_cifar100():
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        return torch.load(CACHE_FILE, weights_only=False)

    print("Downloading/loading dataset...")
    train_data = datasets.CIFAR100(root=DATA_DIR, train=True, download=True)
    test_data = datasets.CIFAR100(root=DATA_DIR, train=False, download=True)
    torch.save({"train": train_data, "test": test_data}, CACHE_FILE)
    return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    cache = load_cifar100()
    train_data = cache["train"]

    img, label = train_data[0]
    label_name = train_data.classes[label]

    plt.imshow(img)
    plt.title(label_name)
    plt.axis("off")
    plt.show()
    print(f"Label: {label_name}")
