# Transformer Model from Scratch-Sentiment Classification

This repository contains a PyTorch implementation of a Transformer model built from scratch for sentiment classification using the Sentiment140 dataset. The model incorporates advanced components such as **Rotary Positional Encoding (RoPE)**, **Multi-Head Attention**, **RMSNorm**, and **SwiGLU FeedForward**. This project is ideal for learning about Transformer architectures, experimenting with NLP tasks, or customizing for specific use cases.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Studying the Transformer Architecture](#studying-the-transformer-architecture)
- [Customizing the Model](#customizing-the-model)
- [Resources and Requirements](#resources-and-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [License](#license)

## Overview

The model is designed to classify sentiments (positive, neutral, negative) from tweets in the Sentiment140 dataset. It includes a complete pipeline: data preprocessing, model definition, training with early stopping, and evaluation. The architecture is modular, making it easy to study, modify, or extend for other NLP tasks.

## Features

- **Custom Transformer Architecture**: Implements Multi-Head Attention, RoPE, RMSNorm, and SwiGLU.
- **Sentiment Classification**: Trained on Sentiment140 with three classes (positive, neutral, negative).
- **Training Pipeline**: Includes early stopping, learning rate scheduling (Cosine Annealing), and gradient clipping.
- **Evaluation Metrics**: Reports accuracy and weighted F1-score.
- **Modular Code**: Each component (e.g., attention, normalization) is implemented separately for easy understanding and modification.

## Getting Started

To use this project, you'll need basic knowledge of Python, PyTorch, and NLP concepts. Follow the steps below to set up, run, and study the model.

### Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: A GPU (e.g., NVIDIA CUDA-enabled) is recommended for faster training. CPU can be used but will be slower.
- **Dependencies**: Listed in `requirements.txt` (see [Installation](#installation)).
- **Dataset**: Sentiment140 is automatically downloaded via the `datasets` library.

## Studying the Transformer Architecture

To understand the Transformer model and its components, follow these steps:

1. **Read the Notebook**:
   - Open `transformers.ipynb` in Jupyter Notebook or Colab.
   - Each section includes markdown explanations and code for components like RoPE, Multi-Head Attention, RMSNorm, and SwiGLU.
   - Study the comments and markdown cells for insights into why each component is used.

2. **Key Components to Focus On**:
   - **Rotary Positional Encoding (RoPE)**: Learn how it encodes positional information efficiently (`Rotary_Positional_Encoding` class).
   - **Multi-Head Attention**: Understand how attention mechanisms capture relationships between tokens (`Multi_Head_Attention` class).
   - **RMSNorm**: Explore normalization for stable training (`RMSNorm_Add` class).
   - **SwiGLU FeedForward**: Study the non-linear transformation for better expressiveness (`FeedForward` class).

3. **Recommended Resources**:
   - **Papers**:
     - "Attention is All You Need" (original Transformer paper).
     - "RoPE: Rotary Position Embedding" for positional encoding.
   - **Tutorials**:
     - Illustrated Transformer by Jay Alammar.
     - PyTorch Tutorials for deep learning basics.
   - **Books**:
     - *Deep Learning* by Ian Goodfellow et al. (for foundational concepts).
     - *Natural Language Processing with Transformers* by Lewis Tunstall et al. (for practical NLP).

4. **Experiment with Code**:
   - Modify hyperparameters (e.g., `d_model`, `num_heads`, `num_layers`) in the `Model Preparation` section.
   - Run small experiments to see how changes affect performance.
   - Add print statements or use a debugger to inspect intermediate outputs (e.g., attention weights).

## Customizing the Model

You can adapt the model for other datasets or tasks by following these steps:

1. **Change the Dataset**:
   - Replace `sentiment140` with another dataset (e.g., IMDB, SST-2) in the `Data Preparation` section.
   - Update the `process_labels` function to match the new dataset's label structure.
   - Adjust `max_length` in `tokenize_function` based on the new dataset's text length.

2. **Modify the Architecture**:
   - Add more `TransformerBlock` layers by increasing `num_layers`.
   - Experiment with different `d_model` or `d_ff` sizes for larger or smaller models.
   - Replace SwiGLU with standard ReLU or GeLU in the `FeedForward` class.

3. **Fine-Tune for Specific Tasks**:
   - For binary classification, set `num_classes=2` in `TransformerModel`.
   - For other NLP tasks (e.g., text generation), modify the output layer and loss function.
   - Use a different tokenizer (e.g., GPT-2, RoBERTa) by updating the `tokenizer` in `Data Preparation`.

4. **Hyperparameter Tuning**:
   - Adjust `dropout`, `learning_rate`, or `batch_size` to improve performance.
   - Experiment with different schedulers (e.g., `ReduceLROnPlateau`) in the `Training` section.
   - Increase `patience` for early stopping to allow longer training.

5. **Add Features**:
   - Implement attention visualization to inspect what the model focuses on.
   - Add data augmentation (e.g., synonym replacement) to improve robustness.
   - Integrate mixed precision training for faster computation.

## Resources and Requirements

### Hardware

- **GPU**: NVIDIA GPU with at least 8GB VRAM (e.g., GTX 1080, RTX 3060) for training on the full dataset.
- **CPU**: Multi-core CPU for preprocessing and smaller experiments.
- **RAM**: 16GB or more for handling large datasets.
- **Storage**: ~500MB for the dataset and model checkpoints.

### Software

- **Operating System**: Linux, macOS, or Windows (Linux recommended for GPU compatibility).
- **Python Libraries**:
  ```plaintext
  torch>=2.0.0
  transformers>=4.30.0
  datasets>=2.10.0
  tqdm>=4.60.0
  scikit-learn>=1.0.0
  numpy>=1.20.0
  ```
- **Optional**: Jupyter Notebook for interactive exploration.

### Estimated Costs

- **Cloud GPUs**: If using cloud platforms (e.g., AWS, Google Colab Pro), expect $0.5–$2/hour for GPU instances.
- **Local Setup**: A mid-range GPU (~$500) is sufficient for most experiments.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yayaiu6/Transformer-Model-from-Scratch-Sentiment-Classification.git
   cd Transformer-Model-from-Scratch-Sentiment-Classification
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   - Open `transformers.ipynb` and run the first cell to check for CUDA availability.

## Usage

1. **Run the Notebook**:
   - Open `transformers.ipynb` in Jupyter Notebook or Google Colab.
   - Execute cells sequentially to preprocess data, train, and evaluate the model.

2. **Train the Model**:
   - The `Training` section trains the model with early stopping and saves the best checkpoint as `best_model.pt`.
   - Adjust hyperparameters in the `Model Preparation` section if needed.

3. **Evaluate the Model**:
   - The `Evaluation` section loads the best model and reports validation accuracy and F1-score.
   - Test on custom sentences by modifying `test_sentence` in the evaluation cell.

4. **Example Command** (if converted to a script):
   ```bash
   python train.py
   ```
   *(Note: You need to convert the notebook to a `.py` script for this.)*

## Training and Evaluation

- **Training Time**: ~10–12 hours for 17 epochs on an NVIDIA RTX 3060 (varies by hardware).
- **Performance**:
  - Validation Accuracy: ~80.48%
  - Validation F1-Score: ~80.47%
- **Tips for Improvement**:
  - Reduce `dropout` (e.g., from 0.3 to 0.2) for better generalization.
  - Increase `patience` (e.g., to 10) for longer training.
  - Use data augmentation or regularization to boost performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by Yahya Mahroof. For questions, open an issue or contact yahyamahroof35@gmail.com.*
