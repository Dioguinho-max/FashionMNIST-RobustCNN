# My first Deep Learning project developed with PyTorch.

This project implements a Convolutional Neural Network (CNN) for image classification on the FashionMNIST dataset and explores a basic Out-of-Distribution (OOD) detection approach using confidence thresholding.

---

## Features

- Image classification using Convolutional Neural Networks
- FashionMNIST classification
- Out-of-Distribution (OOD) detection
- Data Augmentation
- Batch Normalization
- Dropout regularization
- Early Stopping
- Learning Rate Scheduler
- Confusion Matrix
- Classification Report
- AUROC evaluation
- Automatic CSV prediction generation

---

## Model Architecture

```
Input (28×28)

↓

Conv2D (32)
BatchNorm
ReLU
MaxPool

↓

Conv2D (64)
BatchNorm
ReLU
MaxPool

↓

Conv2D (128)
BatchNorm
ReLU
AdaptiveAvgPool

↓

Dropout

↓

Linear (128 → 64)

↓

ReLU

↓

Dropout

↓

Linear (64 → 10)
```

---

## Dataset

### In-distribution

- FashionMNIST
- 54,000 training images
- 6,000 validation images

### Out-of-Distribution

- MNIST
- 200 images used exclusively for OOD evaluation

Datasets are automatically downloaded through `torchvision`.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Epochs | 20 |
| Early Stopping | Patience = 5 |
| Scheduler | ReduceLROnPlateau |
| Dropout | 0.30 |

---

## Results

| Metric | Value |
|--------|------:|
| Validation Accuracy | **90.67%** |
| Trainable Parameters | **102,026** |
| Number of Classes | 10 |
| OOD Method | Softmax Confidence Threshold |
| Evaluation Metrics | AUROC and Balanced Accuracy |

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn

---

## Repository Structure

```
CNN-FashionMNIST-OOD/
│
├── model.py
├── melhor_modelo.pth
├── predicoes.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Future Work

- Energy-based OOD detection
- Mahalanobis distance
- Temperature scaling
- ResNet architectures
- Confidence calibration
- Larger benchmark datasets

---

## About

This repository contains my first Deep Learning project using PyTorch.

The objective was to understand the complete workflow of training a convolutional neural network, including data preprocessing, regularization techniques, model evaluation, and basic Out-of-Distribution detection.

Although developed as a learning project, it follows good deep learning practices such as early stopping, learning rate scheduling, Batch Normalization, dropout regularization, and comprehensive evaluation metrics.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
