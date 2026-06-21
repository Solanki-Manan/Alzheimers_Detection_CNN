# 🧠 Alzheimer's Stage Classification using a Custom CNN

A PyTorch-based deep learning project that classifies brain MRI scans into four Alzheimer's disease stages using a custom-built Convolutional Neural Network. The model is trained on one dataset and validated on a **completely independent, unseen dataset** to test real-world generalization.

## 🎯 Problem Statement

Alzheimer's disease progresses through distinct stages, and early detection from MRI scans can support faster clinical decision-making. This project builds an end-to-end deep learning pipeline to classify brain MRI images into four categories:

- **NonDemented**
- **VeryMildDemented**
- **MildDemented**
- **ModerateDemented**

## 📊 Datasets

| Dataset | Role | Size | Source |
|---|---|---|---|
| Mendeley Augmented Alzheimer Dataset | Train + Test | 27,187 train / 6,797 test | [Mendeley Data](https://data.mendeley.com/datasets/ch87yswbz4/1) |
| OASIS | Independent validation (unseen) | 6,400 images | [Kaggle](https://www.kaggle.com/datasets/ninadaithal/imagesoasis) |

Using a second, independent dataset for validation (rather than just a train/test split from the same source) was a deliberate choice to check whether the model generalizes to MRI scans from a different acquisition pipeline, rather than just memorizing dataset-specific artifacts.

## 🏗️ Model Architecture

A custom CNN built from scratch in PyTorch (no pretrained backbone):

```
Input (1×64×64 grayscale MRI)
   │
Block 1: Conv(1→32)   → BatchNorm → ReLU → MaxPool   →  32×32
Block 2: Conv(32→64)  → BatchNorm → ReLU → MaxPool   →  16×16
Block 3: Conv(64→128) → BatchNorm → ReLU → MaxPool   →   8×8
Block 4: Conv(128→256)→ BatchNorm → ReLU → MaxPool   →   4×4
   │
Flatten → Dropout(0.5) → FC(4096→512) → ReLU → FC(512→4)
```

**Training setup:** Adam optimizer, StepLR scheduler, CrossEntropy loss, 10 epochs, batch size 64, data augmentation (random horizontal flip + rotation) on the training set.

## 🚀 Results

| Metric | Score |
|---|---|
| Mendeley Test Accuracy | **93.22%** |
| OASIS (unseen) Validation Accuracy | **98.69%** |

### Training Curves
<img src="training_history.png" width="600">

### Confusion Matrix — Mendeley Test Set
<img src="confusion_matrix_dataset_1.png" width="400">

### Confusion Matrix — OASIS Validation Set
<img src="confusion_matrix_dataset_2.png" width="400">

Most misclassifications occur between **adjacent severity stages** (e.g. VeryMildDemented ↔ NonDemented, MildDemented ↔ VeryMildDemented), which aligns with the clinical reality that these stages are visually similar on MRI — rather than indicating random model error.

## 📁 Repository Structure

```
.
├── Alzheimers_Detection_CNN.ipynb   # Step-by-step exploration, training, evaluation
├── train.py                          # End-to-end training & evaluation script
├── best_cnn_model.pth                # Saved model weights (best checkpoint)
├── training_history.png              # Loss & accuracy curves
├── confusion_matrix_dataset_1.png    # Mendeley test set confusion matrix
├── confusion_matrix_dataset_2.png    # OASIS validation confusion matrix
├── requirements.txt
└── README.md
```

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets and place them in the project root:
   - Mendeley dataset folder → rename to `AugmentedAlzheimerDataset`
   - OASIS dataset folder → rename to `OriginalDataset`

## 💻 Usage

**Run the full pipeline (train + evaluate + plots):**
```bash
python train.py
```

**Or explore interactively:**
```bash
jupyter notebook Alzheimers_Detection_CNN.ipynb
```

**Load the trained model for inference:**
```python
import torch
from train import AlzheimerCNN

model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load("best_cnn_model.pth", map_location="cpu"))
model.eval()
```

## 🔭 Future Improvements

- Experiment with transfer learning (ResNet/EfficientNet) as a stronger baseline comparison
- Add Grad-CAM visualizations for model interpretability
- Address class imbalance (ModerateDemented is underrepresented) with weighted loss or oversampling
- Deploy as a simple inference API/demo

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is **not** a validated clinical diagnostic tool and should not be used for real medical decision-making.

## 🧰 Tech Stack

`Python` · `PyTorch` · `torchvision` · `scikit-learn` · `NumPy` · `Matplotlib` · `Seaborn`
