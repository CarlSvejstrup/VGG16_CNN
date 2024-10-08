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


class VGG16D(torch.nn.Module):
    def __init__(
        self, num_classes, in_channels=3, features_fore_linear=25000, dataset=None
    ):
        super().__init__()

        conv_stride = 1  # vgg16: 1
        pool_stride = 2  # vgg16: 2
        conv_kernel = 3  # vgg16: 3
        pool_kernel = 2  # vgg16: 2
        dropout_probs = 0.5  # v
        optim_momentum = ...
        weight_decay = 0  # Adam: 0, vgg16: ?
        learning_rate = 0.001  # Adam: 0, vgg16: ?

        # Convolutional layers
        self.features = torch.nn.Sequential(
            # First 2 conv layers with maxpooling
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.ReLU(),
            # Next 2 conv layers with maxpooling
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.ReLU(),
            # Next 3 conv layers with maxpooling
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.ReLU(),
            # Next 3 conv layers with maxpooling
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.ReLU(),
            # Next 3 conv layers with maxpooling
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=conv_kernel,
                padding=1,
                stride=conv_stride,
            ),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
        )

        # Linear layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=features_fore_linear, out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout_probs),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(dropout_probs),
            nn.Linear(in_features=4096, out_features=10),
            nn.Softmax(),
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()

        # Optimizer - For now just set to Adam to test the implementation
        self.optim = torch.optim.Adam(
            list(self.features.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # self.optim = torch.optim.SGD(list(self.features.parameters()) + list(self.classifier.parameters()), lr=learning_rate, momentum=optim_momentum, weight_decay=weight_decay)

        self.dataset = dataset

    def forward(self, x):
        return self.classifier(self.features(x))

    def train_model(
        self, train_dataloader, epochs=1, val_dataloader=None, eval_every_epoch=1
    ):

        # Call .train() on self to turn on dropout
        self.train()

        # To hold accuracy during training and testing
        train_accs = []
        test_accs = []

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_acc = 0

            for inputs, targets in tqdm(train_dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = self(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()

                # Keep track of training accuracy
                epoch_acc += (torch.argmax(logits, dim=1) == targets).sum().item()
            train_accs.append(epoch_acc / len(train_dataloader.dataset))

            # If val_dataloader, evaluate after every epoch
            if val_dataloader is not None:
                # Turn off dropout for testing
                self.eval()
                acc = self.eval_model(val_dataloader)
                test_accs.append(acc)
                print(f"Epoch {epoch} validation accuracy: {acc}")
                # turn on dropout after being done
                self.train()

        return train_accs, test_accs

    def eval_model(self, test_dataloader):

        self.eval()
        total_acc = 0
        with torch.no_grad():
            for input_batch, label_batch in test_dataloader:
                input_batch, label_batch = input_batch.to(device), label_batch.to(
                    device
                )
                logits = self(input_batch)
                total_acc += (torch.argmax(logits, dim=1) == label_batch).sum().item()
        total_acc = total_acc / len(test_dataloader.dataset)
        return total_acc

    def predict(self, img_path):
        img = PIL.Image.open(img_path)
        img = self.dataset.dataset.transform(img)
        classification = torch.argmax(self(img.unsqueeze(dim=0)), dim=1)
        return img, classification

    def predict_random(self, num_predictions=16):
        """
        Plot random images from own given dataset
        """
        random_indices = np.random.choice(
            len(self.dataset) - 1, num_predictions, replace=False
        )
        classifcations = []
        labels = []
        images = []
        for idx in random_indices:
            img, label = self.dataset.__getitem__(idx)

            classifcation = torch.argmax(self(img.unsqueeze(dim=0)), dim=1)

            classifcations.append(classifcation)
            labels.append(label)
            images.append(img)

        return classifcations, labels, images


def get_vgg_weights(model):
    """
    Loads VGG16 weights for the classifier to an already existing model (1000 classes).
    Also sets training to only the classifier.
    """
    # Load the complete VGG16 model with pre-trained weights
    temp = torchvision.models.vgg16(weights="DEFAULT")

    # Get its state dict
    state_dict = temp.state_dict()

    # Modify the classifier to fit the output of your model (10 classes instead of 1000). Changing the last layer of the classifier
    state_dict["classifier.6.weight"] = torch.randn(
        10, 4096
    )  # Adjust last layer to 10 classes
    state_dict["classifier.6.bias"] = torch.randn(10)

    # Directly apply the state dict to your model without modifying the classifier
    model.load_state_dict(
        state_dict, strict=True
    )  # strict=True ensures everything must match

    # Freeze the features part of the network (only train the classifier).
    # THis is done, since we have random wright for the last classifier layer.
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the optimizer to only train the classifier
    model.optim = torch.optim.Adam(model.classifier.parameters())

    return model
