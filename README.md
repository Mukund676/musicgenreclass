# 🎵 Music Genre Classification App

This project is a machine learning-based application that classifies music into genres based on audio data. It extracts relevant features from an input music file and uses a neural network to predict the genre.

## What It Does

The Music Genre Classification App takes an audio file (in formats like MP3, WAV, etc.) as input and predicts its genre from a set of pre-trained music genres. Some of the genres that the app can classify include:

- Rock
- Jazz
- Classical
- Hip-hop
- Pop
- Metal
- Country
- Blues
- Reggae
- Disco

## How It Works

The app works by following these steps:

1. **Audio Feature Extraction**: 
   - The input music file is processed to extract key audio features. The features include:
     - **MFCCs (Mel Frequency Cepstral Coefficients)**: These coefficients represent the power spectrum of the sound signal and capture timbral information.
     - **Chroma Features**: These are used to represent the tonal content of the audio based on pitch classes.
     - **Spectral Contrast**: Captures differences in amplitude between peaks and valleys in a sound spectrum.
   - These features are extracted using the **Librosa** library.

2. **Feature Scaling & Data Preparation**:
   - Once the audio features are extracted, they are normalized and scaled to ensure consistent input to the neural network. The data is then split into training, validation, and testing sets.

3. **Model Training (Convolutional Neural Network)**:
   - The core of the app is a **Convolutional Neural Network (CNN)** model that is trained to classify audio features into music genres. The model learns to identify patterns and correlations in the feature data that correspond to each genre.
   - The architecture includes:
     - **Convolutional Layers**: Detect features from the input data.
     - **MaxPooling Layers**: Downsample feature maps to reduce dimensionality and computation.
     - **Fully Connected Layers**: Interpret the features and classify them into a genre.
     - **Softmax Layer**: Outputs the probability distribution over the possible genres.

4. **Genre Prediction**:
   - After training, the model can take new, unseen audio files and classify them into one of the predefined genres.
   - The app outputs the predicted genre along with a confidence score.

5. **Evaluation**:
   - The model is evaluated based on accuracy, precision, and recall using validation data. Cross-validation is performed to prevent overfitting and improve generalization.

## Key Concepts

- **MFCCs**: Represent the short-term power spectrum of sound, often used for identifying characteristics in speech and music.
- **Chroma Features**: Describe the intensity of the twelve different pitch classes (like notes in the musical scale) and help distinguish harmonic content.
- **Convolutional Neural Network (CNN)**: A type of deep learning model that excels at detecting spatial hierarchies in data, making it suitable for image and audio processing.

## Future Scope

The app can be extended with additional features, such as:
- Support for real-time genre classification.
- Expansion to include more genres or sub-genres.
- A user-friendly interface for uploading audio and viewing predictions.
- Integration of streaming data for live classification.

---

This provides a clear overview of what the app does and how it performs the genre classification process using machine learning.
