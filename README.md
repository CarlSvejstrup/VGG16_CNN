# VGG16 Implementation in PyTorch

A from-scratch implementation of the VGG16 architecture trained on the Imagenette dataset (a subset of ImageNet). This project explores the mechanics of Convolutional Neural Networks, custom weight loading, and transfer learning.

## 🧠 Architecture

The model follows the original VGG16 specification (Simonyan & Zisserman, 2014):
* **Feature Extractor:** 5 blocks of convolutional layers (3x3 kernels) followed by max-pooling.
* **Classifier:** 3 Fully Connected layers (4096 -> 4096 -> 10).
* **Modifications:** The classifier is adapted for 10 classes (Imagenette) instead of the original 1000 (ImageNet).

## 🛠 Features

* **Custom Model Definition:** The network is defined layer-by-layer in `VGG16_D.py` to demonstrate understanding of the architecture.
* **Transfer Learning:** Includes a custom utility to map pre-trained ImageNet weights to the modified architecture.
* **Data Pipeline:** Efficient loading and transforming of the Imagenette dataset.

## 🚀 Usage

1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy matplotlib tqdm
    ```

2.  **Train the Model:**
    ```bash
    python main.py
    ```

## 📊 Performance

The model achieves convergence on the Imagenette validation set within 5 epochs using Adam optimization.

---
*Created by Carl Schmidt-Svejstrup*
