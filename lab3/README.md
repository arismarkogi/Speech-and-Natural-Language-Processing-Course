# 📝 Sentiment Analysis & Transformer Fine-Tuning

> A comprehensive deep learning pipeline for sentiment analysis, featuring custom-built Transformer architectures and state-of-the-art LLM fine-tuning.

This repository contains the code for Lab 3 of the Speech and Natural Language Processing course. It explores text classification and sentiment analysis across two distinct datasets: the **Movie Reviews (MR)** dataset and the **SemEval 2017 Task 4A** (Twitter sentiment) dataset. 

The project bridges the gap between understanding core architectural mechanics and applying modern NLP frameworks, moving from building raw attention mechanisms to fine-tuning massive language models.

## ✨ Key Features

* **Custom Transformer Architecture:** Implemented Multi-Head Attention and complete Transformer blocks entirely from scratch using PyTorch to deeply understand self-attention mechanics (`attention.py`, `models.py`).
* **Pre-trained LLM Fine-Tuning:** Leveraged the Hugging Face ecosystem to load, adapt, and fine-tune state-of-the-art models (e.g., **DistilBERT**, **RoBERTa**) for downstream sequence classification (`finetune_pretrained.py`).
* **Robust Training Pipeline:** Engineered a complete ML pipeline featuring custom PyTorch `DataLoaders`, Word2Vec embedding integration, learning rate scheduling, and Early Stopping to prevent overfitting (`training.py`, `early_stopper.py`).
* **Comprehensive Evaluation:** Monitored and plotted Accuracy, F1-Score, and Recall across complex multimodal and multi-class target domains.

## 📂 Repository Structure

The project is divided into preparatory and main experimentation modules:

* `prep_lab/`: Introductory code and foundational exercises for embeddings and baseline models.
* `main_lab/`: The core implementation directory.
  * `attention.py` / `models.py`: Custom PyTorch neural network architectures (LSTMs, Transformers).
  * `finetune_pretrained.py`: End-to-end script for fine-tuning SOTA Hugging Face models.
  * `transfer_pretrained.py`: Feature extraction and transfer learning pipelines.
  * `training.py`: The main training and validation loop logic.
  * `utils/`: Data ingestion and GloVe/Word2Vec embedding processors.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required deep learning libraries (PyTorch, Transformers, Datasets, etc.).

```bash
# Install dependencies
pip install -r requirements.txt
