# Student Information
Name: Md Ashikur Rahman  
Student ID: 210110  
Course: Artificial Intelligence and Machine learning Lab  
Course Code: CSE 3202  

# Rock–Paper–Scissors Classification with CNN (PyTorch)

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) in **PyTorch** to classify **hand gestures** for the game Rock–Paper–Scissors.

- **Standard Dataset**: Rock-Paper-Scissors dataset from **TensorFlow Datasets**.
- **Custom Data**: My own smartphone photos of my hand showing Rock, Paper, and Scissors.
- **Goal**: Train a CNN on the standard dataset and test it on real-world photos taken with my phone.

This repository is built to satisfy the course assignment requirements:
- Automatic data loading (standard dataset + custom images from this GitHub repo).
- Full training pipeline in PyTorch.
- Real-world testing on custom phone images.
- Visualizations: training curves, confusion matrix, prediction gallery, and error analysis.

---

## 2. Repository Structure

```
.
├── dataset/           # 10+ custom smartphone images (Rock, Paper, Scissors)
├── model/             # Saved model state dict (.pth)
├── 210110.ipynb       # Google Colab notebook (replace with your actual ID)
└── README.md          # This file
dataset/: Contains my phone images
model/210110.pth: The trained PyTorch model weights (state_dict).
210110.ipynb: Notebook that can be opened in Colab and run end-to-end.
```
## 3. Dataset Details

### 3.1 Standard Dataset: Rock–Paper–Scissors
Source: tensorflow_datasets (rock_paper_scissors)
Number of classes: 3
Class 0: Rock
Class 1: Paper
Class 2: Scissors
Input images are RGB, resized to 150 × 150.
Transforms applied (training and test):
```
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
```
### 3.2 Custom Smartphone Images
I took photos of my own hand making:
Rock (fist)
Paper (open hand)
Scissors (two-finger V shape)
Number of images: TODO: write how many custom images you used (>=10)
Photos were taken on a plain background (table/wall) to reduce noise.
These images are stored in the dataset/ folder and automatically loaded by the notebook.
Preprocessing for custom images:
```
transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
```
This matches the training data format (RGB, size, normalization).

## 4. Model Architecture (CNN)
The CNN is implemented in RPS_CNN(nn.Module) with 3 convolutional blocks and 2 fully connected layers:
```
class RPS_CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(RPS_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 18 * 18, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```
Input: 3 × 150 × 150 (RGB image)
Convolution + ReLU + MaxPool repeated 3×
Final feature map: 128 × 18 × 18
Fully connected layers: 128*18*18 → 512 → 3
Output: logits for the 3 classes

## 5. Training Setup
Loss Function: nn.CrossEntropyLoss
Optimizer: torch.optim.Adam with lr=1e-3
Batch size: 32
Epochs: 10
Device: GPU (cuda) when available, otherwise CPU.
Training loop includes:
```
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```
Validation is done on the provided test split from the RPS dataset.

At the end of training, the model parameters are saved as:
```
torch.save(model.state_dict(), 'model/210110.pth')
```
## 6. Results
Epoch 1/10 - Train Loss: 0.0131, Train Acc: 99.80% | Val Loss: 0.8634, Val Acc: 85.75%
Epoch 2/10 - Train Loss: 0.0012, Train Acc: 99.92% | Val Loss: 0.4732, Val Acc: 88.98%
Epoch 3/10 - Train Loss: 0.0268, Train Acc: 99.44% | Val Loss: 1.8152, Val Acc: 86.29%
Epoch 4/10 - Train Loss: 0.0078, Train Acc: 99.80% | Val Loss: 1.7973, Val Acc: 86.56%
Epoch 5/10 - Train Loss: 0.0001, Train Acc: 100.00% | Val Loss: 1.8055, Val Acc: 81.72%
Epoch 6/10 - Train Loss: 0.0003, Train Acc: 100.00% | Val Loss: 1.7874, Val Acc: 81.72%
Epoch 7/10 - Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 2.0883, Val Acc: 84.95%
Epoch 8/10 - Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 2.1415, Val Acc: 85.48%
Epoch 9/10 - Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 2.4016, Val Acc: 84.95%
Epoch 10/10 - Train Loss: 0.0000, Train Acc: 100.00% | Val Loss: 2.3995, Val Acc: 85.22%

## 6.2 Training History Plots
The notebook generates:

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/3c76b6f0-0fec-4562-8c2c-479cef64788c" />

Loss vs. Epochs (train & validation)
Accuracy vs. Epochs (train & validation)

## 7. Confusion Matrix (Standard Test Set)
The confusion matrix shows performance on the standard test set classes: Rock, Paper, Scissors.

<img width="574" height="473" alt="image" src="https://github.com/user-attachments/assets/34a79309-154c-45e9-81ad-5b3a65364cf2" />

## 8. Custom Phone Image Predictions

<img width="1438" height="1229" alt="image" src="https://github.com/user-attachments/assets/44043be0-e649-4d10-a440-9312110aa731" />

## 9. Error Analysis (Misclassified Examples)
<img width="816" height="299" alt="image" src="https://github.com/user-attachments/assets/949b7302-6abb-498e-b6d4-db077ca9333b" />

## 10. How to Run This Project
### 10.1 Run in Google Colab
Open the notebook in Colab:
https://colab.research.google.com/drive/1BMtl8_2g2RPHELZH8Q5qJbSBL2QQh40q?usp=sharing

In Colab, go to Runtime → Run all.

The notebook will:

Clone this repo:
!git clone https://github.com/PhataPoster/CNN
Load the standard Rock–Paper–Scissors dataset.
Either:
Load the saved model from model/210110.pth, or
Train a new model if the file is missing.
Generate:
Training/validation loss & accuracy plots (if training took place in this run).
Confusion matrix.
Custom prediction gallery from images in dataset/.
Error analysis plots.
No manual file uploads are required during the run.

### 10.2 Requirements
The notebook installs required packages automatically in Colab:
torch, torchvision
tensorflow, tensorflow-datasets
pandas, matplotlib, seaborn, scikit-learn

## 11. Notes and Limitations
The model is trained only on the Rock–Paper–Scissors dataset, so it’s not expected to generalize to unrelated gestures or objects.
Performance can be improved with:
More epochs
Data augmentation
A deeper CNN or transfer learning

