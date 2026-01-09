# TinySense

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**CloudSense** is a deep learning project designed for Wi-Fi-based Human Pose Estimation. It leverages **Selective Kernel Networks (SKNet)** to efficiently extract features and perform reconstruction and human pose estimation.

## 📂 Project Structure

The repository is organized into the following modules:

* **`Model/`**: Contains the core neural network architecture definitions.
* **`SK_network/`**: Implementation of the Selective Kernel (SK) Network components or backbone.
* **`trainer/`**: Scripts and utilities for training the model, including training loops and loss calculations.
* **`evaluation/`**: Tools for evaluating model performance and generating metrics.

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* [PyTorch](https://pytorch.org/) (Recommended)
* NumPy, Scikit-learn, etc.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Toandinh1/CloudSense.git
    cd CloudSense
    ```

2.  **Install dependencies:**
    *(Note: Please ensure you have a `requirements.txt` file or install necessary packages manually)*
    ```bash
    pip install -r requirements.txt
    # OR manually:
    pip install torch torchvision numpy
    ```

## 🛠️ Usage

### Training

To train the model, use the training script located in the `trainer` directory:

```bash
python trainer/train.py --config config.json
