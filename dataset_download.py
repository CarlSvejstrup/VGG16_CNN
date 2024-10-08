import os
import torch
import PIL
from torch import nn
from torch.utils.data.dataloader import default_collate

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models
from torch.utils.data import Subset

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary

import utils
from VGG16_D import *

# Check if you have cuda available, and use if you do
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Set a random seed for everything important
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


# Set a seed with a random integer, in this case, I choose my verymost favourite sequence of numbers
seed_everything(
    sum([115, 107, 105, 98, 105, 100, 105, 32, 116, 111, 105, 108, 101, 116])
)

# Specify dataset you wanna use

def get_dataset(
    dataset_name, validation_size=0.1, transform=None, v=True, crop_size=224
):

    if transform is None:
        transform = ToTensor()

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=ToTensor()
        )
        test_set = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=ToTensor()
        )

        # Purely for our convenience - Mapping from cifar labels to human readable classes
        cifar10_classes = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }

    elif dataset_name == "mnist":
        train_set = datasets.MNIST(
            root="./data", train=True, download=True, transform=ToTensor()
        )
        test_set = datasets.MNIST(
            root="./data", train=False, download=True, transform=ToTensor()
        )

    elif dataset_name == "imagenette":
        download = not os.path.exists("./data/imagenette2")

        # Specific transform in the case we use imagenette
        imagenette_transform = transforms.Compose(
            [
                transforms.Resize(256),  # Resize to 256x256
                transforms.RandomCrop(crop_size),  # Crop the center to 224x224
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize(
                    mean=[0.4650, 0.4553, 0.4258], std=[0.2439, 0.2375, 0.2457]
                ),  # Normalize each image, numbers because of function courtesy of chatgpt
            ]
        )
        train_set = datasets.Imagenette(
            root="./data",
            split="train",
            download=download,
            size="full",
            transform=imagenette_transform,
        )
        test_set = datasets.Imagenette(
            root="./data",
            split="val",
            download=False,
            size="full",
            transform=imagenette_transform,
        )

    # If we want a validation set of a given size, take it from train set
    if validation_size is not None:
        # These will both be of the torch.utils.data.Subset type (not the Dataset type), and are basically just mappings of indices
        # This does not matter when we make Dataloaders of them, however
        if dataset_name != "imagenette":
            train_set, validation_set = torch.utils.data.random_split(
                train_set, [1 - validation_size, validation_size]
            )

        # In the case of imagenette, the 'test set' is already a pretty big validation set, so we'll use that to create the test set instead
        else:
            validation_set, test_set = torch.utils.data.random_split(
                test_set, [validation_size, 1 - validation_size]
            )

    if v:
        print(f"There are {len(train_set)} examples in the training set")
        print(f"There are {len(test_set)} examples in the test set \n")

        print(
            f"Image shape is: {train_set[0][0].shape}, label example is {train_set[0][1]}"
        )

    return train_set, validation_set, test_set
