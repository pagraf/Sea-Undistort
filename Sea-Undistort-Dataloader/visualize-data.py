from torch.utils.data import DataLoader
from SeaUndistort import SeaUndistort
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config



config = load_config("configs/config-example.yaml")

train_dataset = SeaUndistort(config, split_mode="train")

train_loader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

fig, axs = plt.subplots(4, 3, figsize=(15, 20))

axs[0, 0].set_title("Image")
axs[0, 1].set_title("Label")
axs[0, 2].set_title("Mask")


for batch_idx, batch in enumerate(train_loader):

    # print("Batch: ", batch)
    inputs = batch["lq"]
    labels = batch["gt"]
    masks = batch["mask"]

    # print("Shape of input: ", inputs.shape)
    # print("Shape of label: ", labels.shape)
    # print("Shape of mask: ", masks.shape)

    inputImg = inputs[0].permute(1,2,0).numpy()
    axs[batch_idx, 0].imshow(inputImg, aspect="auto")
    axs[batch_idx, 0].axis("off")

    label = labels[0].permute(1,2,0).numpy()
    axs[batch_idx, 1].imshow(label, aspect="auto")
    axs[batch_idx, 1].axis("off")

    mask = masks[0].permute(1,2,0).numpy()
    axs[batch_idx, 2].imshow(mask, aspect="auto")
    axs[batch_idx, 2].axis("off")

    if batch_idx == 3:
        break

plt.tight_layout()
plt.savefig(f"./visualization.png") # , dpi=300)

plt.close(fig)           # or plt.close('all')