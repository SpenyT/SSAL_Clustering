import os
import torch
import matplotlib.pyplot as plt
import torchvision.datasets as datasets

from data.utils import calculate_save_mean_std, load_data


if __name__ == "__main__":
    calculate_save_mean_std()
    mean, std = load_data("mean"), load_data("std")
    print(f"Mean: {mean}\nStd: {std}")
