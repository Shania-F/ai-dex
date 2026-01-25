import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# https://www.kaggle.com/datasets/sevans7/pokemon-images-scraped-from-bulbapedia/data
# Bulbapedia images → transparent PNGs
# Transparency is not RGB
# PIL converts transparency → black
# Better backgrounds (like white or random noise)= better generalization (later)

def get_pokemon_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=2,
    val_split=0.1,
    seed=42,
):
    """
    Returns train and validation DataLoaders for the Pokémon dataset.
    """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # IMPORTANT: validation must NOT use augmentation
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, full_dataset


# move to data.py later
import matplotlib.pyplot as plt
import numpy as np


def show_one_image(images, labels, class_names):
    """
    images: tensor [B, 3, H, W]
    labels: tensor [B]
    """
    img = images[0].cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C,H,W → H,W,C

    # Unnormalize ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    label_idx = labels[0].item()
    label_name = class_names[label_idx]

    plt.imshow(img)
    plt.title(f"Label: {label_name} ({label_idx})")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Quick test
    data_dir = r"D:\PycharmProjects\Pokemon_Dataset_879"
    dataloader, val_loader, dataset = get_pokemon_dataloaders(data_dir=data_dir, num_workers=0)
    images, labels = next(iter(dataloader))
    print(images.shape, labels.shape)  # should be [8, 3, 224, 224] and [8]

    print("Label index:", labels[0].item())
    print("Label name:", dataset.classes[labels[0].item()])

    show_one_image(images, labels, dataset.classes)
