📄 DINOv3 Image Embedding Classification

A CS5100 Final Project by Divyam Gupta

📌 Overview

This project investigates the use of DINOv3, a state-of-the-art self-supervised Vision Transformer (ViT-B/16), for extracting high-quality image embeddings to improve downstream image classification performance. Instead of training a full neural network from scratch, the approach leverages DINOv3’s pretrained dense visual features and trains lightweight classifiers—including logistic regression, SVMs, and MLPs—on top of these frozen embeddings.

The focus is on analyzing how the depth of a projection head (MLP) and the use of Batch Normalization affect classification stability and accuracy across diverse datasets.

📎 Project Code: https://github.com/dgibn/FAI_project.git

📊 Datasets

Experiments were conducted on three widely used image classification benchmarks:

CUB-200-2011 – Fine-grained bird species dataset (200 classes)

PACS – Domain generalization benchmark (7 classes)

CIFAR-10 – Standard multi-class dataset (10 classes)

## 🚀 Training

To train the model, you can use the `scripts/train.sh` script. The model is trained on a specific dataset

### Automated Training Script
Use the provided bash script to automate training across multiple domains defined in the script:

```bash
bash scripts/train.sh
```
*Note: Edit `train.sh` to select the dataset you wish to train on. and edit the dataloader files in ./data folder*

🏆 Best Overall Configuration
Dataset	Best Accuracy	Model Setting
CUB-200-2011	~46.1%	n = 1, No BatchNorm
PACS	~99.1%	n = 1, No BatchNorm
CIFAR-10	~95.0%	n = 3, No BatchNorm

🔭 Future Work

Add contrastive or margin-based loss for better class separation

Evaluate robustness under domain shift

Integrate attention modules into the classifier for better local feature reasoning

📚 References

DINOv3 Paper: https://arxiv.org/abs/2508.10104

CIFAR-10 Dataset

CUB-200-2011 Dataset

PACS Dataset
