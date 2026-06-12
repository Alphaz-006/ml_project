# CIFAR-10 Image Classification with ResNet50 Transfer Learning

## Overview
An image classification model that categorizes images into 10 classes (airplane, car, bird, cat, etc.) using transfer learning with a pretrained ResNet50 backbone.

## Dataset
- **CIFAR-10**: 60,000 32x32 color images across 10 classes (50,000 train / 10,000 test)
- Loaded directly via `tensorflow.keras.datasets`

## Approach
1. Resized images from 32x32 to 96x96 to fit ResNet50's expected input range
2. Applied ResNet50-specific preprocessing (`preprocess_input`)
3. **Phase 1**: Froze the ResNet50 base and trained a custom classification head
4. **Phase 2**: Unfroze the last 20 layers of ResNet50 and fine-tuned with a lower learning rate

## Model Architecture
- ResNet50 (pretrained on ImageNet, frozen initially)
- GlobalAveragePooling2D
- Dense (128, ReLU) + Dropout (0.3)
- Dense (10, softmax)

## Results
- **Accuracy after frozen training**: [XX.X]%
- **Accuracy after fine-tuning**: [XX.X]%
- Per-class confusion matrix included in notebook

## Tech Stack
`Python` `TensorFlow/Keras` `ResNet50` `Matplotlib`

## How to Run
1. Open `cifar10_transfer_learning.ipynb` in Google Colab (GPU runtime recommended)
2. Run all cells sequentially — dataset downloads automatically
3. Training takes ~15-20 minutes on a T4 GPU

## Future Improvements
- Try other backbones (EfficientNet, MobileNetV3) for accuracy/speed tradeoffs
- Apply data augmentation (random crop, flip, color jitter)
- Deploy as a simple web app for live image classification
