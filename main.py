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
from dataset_download import get_dataset

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


# collate function just to cast to device, same as in week_3 exercises
def collate_fn(batch):
    return tuple(x_.to(device) for x_ in default_collate(batch))


dataset_name = "imagenette"
# dataset_name = "cifar10"
# dataset_name = "mnist"

crop_size = 224

# Load the dataset
train_set, validation_set, test_set = get_dataset(
    dataset_name, validation_size=0.1, crop_size=crop_size
)


def create_subset(dataset, fraction):
    """Creates a random subset of a dataset based on the specified fraction."""
    num_samples = int(len(dataset) * fraction)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)


def create_dataloader(subset, batch_size, shuffle, collate_fn):
    """Creates a DataLoader for the given subset."""
    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


# Define fractions for train, test, and validation sets
# fractions = {"train": 0.5, "test": 0.5, "val": 0.5}
# For full dataset
fractions = {"train": 1, "test": 1, "val": 1}

# Create subsets
train_subset = create_subset(train_set, fractions["train"])
test_subset = create_subset(test_set, fractions["test"])
val_subset = create_subset(validation_set, fractions["val"])

# Batch size for DataLoader
# Increased batch size to 32, to utilize GPU better.
batch_size = 32

# Create DataLoaders
train_dataloader = create_dataloader(
    train_subset, batch_size, shuffle=True, collate_fn=collate_fn
)
test_dataloader = create_dataloader(
    test_subset, batch_size, shuffle=False, collate_fn=collate_fn
)
val_dataloader = create_dataloader(
    val_subset, batch_size, shuffle=False, collate_fn=collate_fn
)


in_channels = next(iter(train_dataloader))[0].shape[1]
in_width_height = next(iter(train_dataloader))[0].shape[-1]
# Make a dummy model to find out the size before the first linear layer
CNN_model = VGG16D(num_classes=10, in_channels=3)

# WARNING - THIS PART MIGHT BREAK
features_fore_linear = utils.get_dim_before_first_linear(
    CNN_model.features, in_width_height, in_channels, brain=False
)

# Now make true model when we know how many features we have before the first linear layer
CNN_model = VGG16D(
    num_classes=10,
    in_channels=in_channels,
    features_fore_linear=features_fore_linear,
    dataset=test_set,
)

# Extracting pretrained weights from VGG16 to the model class
CNN_model = get_vgg_weights(CNN_model)
CNN_model.to(device)

# Training epochs
train_epochs = 5

# Train the model
train_accs, test_accs = CNN_model.train_model(
    train_dataloader, epochs=train_epochs, val_dataloader=val_dataloader
)

# Save the model
torch.save(CNN_model.state_dict(), "./data/models/full_set-10_class_vgg16.pth")
