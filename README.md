# Emotion-detection_openCV
Here’s a professional `README.md` for your **real-time Emotion Detection with CNN and OpenCV** project (`emotion.py`):

---

## 😄 Emotion Detection with CNN & OpenCV

This project implements a real-time emotion detection system using a Convolutional Neural Network (CNN) trained on grayscale facial images. It uses OpenCV for face detection and TensorFlow/Keras for emotion classification.

---

### 📸 Live Demo

The system uses your webcam to detect faces and classify emotions in real-time, displaying labels like `happy`, `sad`, `angry`, etc., on detected faces.

---

### 🧠 Model Overview

* **Model Type:** CNN (Convolutional Neural Network)
* **Input:** 48x48 grayscale facial images
* **Output:** Emotion classification (softmax)
* **Framework:** TensorFlow/Keras
* **Face Detection:** OpenCV Haar cascades

---
 Recommended Kaggle Dataset:
FER-2013: Facial Expression Recognition
🔗 Link: https://www.kaggle.com/datasets/msambare/fer2013
### 🗂 Dataset Format

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── happy/
│   ├── sad/
│   └── ...
├── test/
│   ├── happy/
│   ├── sad/
│   └── ...
```

Each emotion class should be in its own folder with grayscale 48x48 facial images.

---

### ⚙️ Installation

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/Emotion-detection_openCV.git
cd emotion-detection
```

2. **Install dependencies**

```bash
pip install tensorflow opencv-python numpy
```

> You may also need:

```bash
pip install keras
```

---

### 🚀 How to Run

1. **Train the model**

```bash
python emotion.py
```

This will:

* Load and preprocess images
* Train a CNN model
* Save the model as `emotion_model.h5`
* Launch webcam for live prediction

2. **Use webcam for emotion detection**
   The webcam will open automatically after training, or load `emotion_model.h5` if it already exists.

Press `q` to exit webcam mode.

---

### 📁 Files

| File               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `emotion.py`       | Main script for training and real-time detection |
| `emotion_model.h5` | Trained CNN model (saved after training)         |
| `dataset/`         | Folder containing training and testing images    |

---

### 📊 Model Architecture

* Conv2D → ReLU → MaxPooling
* Conv2D → ReLU → MaxPooling
* Flatten → Dense → Dropout → Softmax

---

### 🔍 Emotion Labels

Automatically inferred from folder names in `dataset/train/`. For example:

* `happy`, `angry`, `surprised`, `sad`, etc.

---

### 📌 Notes

* Ensure your dataset is balanced across emotion classes.
* Webcam access must be granted for live prediction.
* Model is trained in grayscale for efficiency.

---

### 📄 License

MIT License — free to use, modify, and share.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Last Commit](https://img.shields.io/github/last-commit/Soham-Banerjee-web/Emotion-detection_openCV)
![Issues](https://img.shields.io/github/issues/Soham-Banerjee-web/Emotion-detection_openCV)

