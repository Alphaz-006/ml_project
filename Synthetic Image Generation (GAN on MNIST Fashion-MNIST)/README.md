# Synthetic Image Generation using DCGAN

## Overview
A Deep Convolutional Generative Adversarial Network (DCGAN) trained to generate realistic synthetic images of fashion items, learning the underlying data distribution of the Fashion-MNIST dataset.

## Dataset
- **Fashion-MNIST**: 60,000 grayscale 28x28 images across 10 clothing categories
- Loaded directly via `tensorflow.keras.datasets`

## Approach
1. Normalized images to [-1, 1] range to match generator's `tanh` output
2. Built a Generator that maps random noise vectors (100-dim) to 28x28 images via transposed convolutions
3. Built a Discriminator (CNN) that classifies images as real or generated
4. Trained both networks adversarially using binary cross-entropy loss

## Model Architecture
**Generator**: Dense → Reshape → 3x Conv2DTranspose (with BatchNorm + LeakyReLU)
**Discriminator**: 2x Conv2D (with LeakyReLU + Dropout) → Dense

## Results
- Trained for [50] epochs
- Generated samples show recognizable clothing silhouettes by epoch [30-40]
- Sample generated images and loss curves included in notebook

## Tech Stack
`Python` `TensorFlow/Keras` `Matplotlib` `NumPy`

## How to Run
1. Open `gan_image_generation.ipynb` in Google Colab (GPU runtime recommended)
2. Run all cells sequentially — dataset downloads automatically
3. View generated samples printed every 10 epochs

## Future Improvements
- Extend to a conditional GAN (cGAN) to generate specific classes on demand
- Experiment with a diffusion model (DDPM) for higher-quality outputs
- Train on a custom domain dataset (e.g., product images)
