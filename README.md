# Alzheimer's Detection Using Deep Learning

This repository contains a Custom Convolutional Neural Network (CNN) built with PyTorch to detect and classify Alzheimer's disease from MRI scans.

## 📊 Datasets
The model is trained and evaluated using two distinct datasets to ensure robust generalization:
1. **[Mendeley Dataset](https://data.mendeley.com/datasets/ch87yswbz4/1)** (Dataset 1): Used for training (27,187 images) and testing (6,797 images).
2. **[OASIS Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)** (Dataset 2): Used as a completely unseen validation set (6,400 images) to test real-world generalization.

## 🚀 Performance
- **Mendeley Test Accuracy:** 93.22%
- **OASIS Validation Accuracy:** 98.69%

## 📁 Repository Structure
- `Alzheimers_Detection_CNN.ipynb`: Jupyter notebook containing data exploration, step-by-step model training, and evaluation logic.
- `train.py`: Production-ready Python script for training and evaluating the model.
- `best_cnn_model.pth`: Saved PyTorch model weights.
- `training_history.png`: Plot of accuracy and loss during training.
- `confusion_matrix_dataset_1.png`: Evaluation results on the Mendeley test set.
- `confusion_matrix_dataset_2.png`: Evaluation results on the OASIS validation set.
- `Final_Report.pdf`: Detailed project report.

## 🛠️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets and place them in the root directory:
   - Name the Mendeley dataset folder: `AugmentedAlzheimerDataset`
   - Name the OASIS dataset folder: `OriginalDataset`

## 💻 How to Run

You can run the end-to-end pipeline using the Python script:
```bash
python train.py
```
Or explore the step-by-step process in the `Alzheimers_Detection_CNN.ipynb` notebook.
