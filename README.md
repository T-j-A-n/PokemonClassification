# PokemonClassification
This project implements an image classification model that can identify **up to 249 different Pokémon characters** using deep learning and transfer learning with **ResNet101**. It is built with TensorFlow and Keras, and trained on a dataset of Pokémon images organized into folders by class.

---

## 📂 Dataset Structure

The dataset is organized as follows:\n
data/\n
├── train/\n
│ ├── Bulbasaur/\n
│ │ ├── img1.jpg\n
│ │ ├── img2.jpg\n
│ ├── Pikachu/\n
│ │ ├── img1.jpg\n
│ │ ├── img2.jpg\n
│ └── ... (149 Pokémon class folders, each with 30–100 images)\n
├── test/\n
│ ├── 0001.jpg\n
│ ├── 0002.jpg\n
│ └── ... (2195 total test images)\n

- Images are resized to **256x256 pixels**.
- Labels for training are automatically extracted from folder names.
- The test set contains unlabeled images for model evaluation/inference.

---

## 🧰 Tech Stack

- **Python 3.8+**
- **TensorFlow 2.x / Keras**
- **NumPy, Pandas**
- **ResNet101 (pre-trained on ImageNet)**

---

## 🏗️ Model Architecture

- **Backbone**: ResNet101 (`include_top=False`, ImageNet weights)
- **Unfrozen Layers**: Last 30% of the base model for fine-tuning
- **Classifier Head**:
  - GlobalAveragePooling2D
  - BatchNormalization
  - Dense(2048) with ReLU and L2 Regularization
  - Dropout(0.6)
  - Dense(249) with softmax activation

**Training Settings**:
- **Loss**: SparseCategoricalCrossentropy
- **Optimizer**: Adam with CosineDecay learning rate scheduler
- **EarlyStopping**: Patience = 5, monitor = val_loss

---

## 🧪 Training Instructions

### 🔧 Install Dependencies

```bash
git clone https://github.com/your-username/pokemon-classifier.git
cd pokemon-classifier
pip install -r requirements.txt

📊 Results
Training Accuracy: (Add your result here)

Validation Accuracy: (Add your result here)

Test Accuracy: (Optional, if you manually label test set)
