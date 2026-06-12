# ml_project
# Speech Emotion Recognition using MFCC + CNN-LSTM

## Overview
A deep learning model that classifies human emotions (happy, sad, angry, calm, neutral, fearful, disgust, surprised) from raw speech audio using MFCC features and a hybrid CNN-LSTM architecture.

## Dataset
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,440 audio samples across 8 emotion classes from 24 actors
- Source: [Kaggle - RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## Approach
1. Loaded and preprocessed `.wav` audio files using `librosa`
2. Extracted 40-dimensional MFCC (Mel-Frequency Cepstral Coefficients) features per frame, padded/truncated to a fixed length
3. Built a CNN-LSTM model: Conv1D layers extract local spectral patterns, LSTM captures temporal dependencies across frames
4. Trained with categorical cross-entropy loss and Adam optimizer

## Model Architecture
- 2x Conv1D + BatchNorm + MaxPooling blocks
- LSTM layer (128 units)
- Dropout (0.4) for regularization
- Dense output layer with softmax (8 classes)

## Results
- **Test Accuracy**: [XX.X]%
- Confusion matrix and per-class precision/recall included in notebook

## Tech Stack
`Python` `TensorFlow/Keras` `Librosa` `NumPy` `Scikit-learn`

## How to Run
1. Open `speech_emotion_recognition.ipynb` in Google Colab
2. Upload `kaggle.json` for dataset download (or manually upload RAVDESS dataset)
3. Run all cells sequentially

## Future Improvements
- Fine-tune a pretrained Wav2Vec2 model for comparison
- Data augmentation (pitch shift, noise injection) to improve generalization
- Real-time inference from microphone input
