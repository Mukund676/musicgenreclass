import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
import librosa

model = load_model(r".\models\music_genre_cnn.h5")

def predict_genre(file_path):
    """
    This function predicts genre of audio file
    
    Parameters: 
    - file_name (str): path to audio file

    Returns:
    - str: predicted genre
    """
    
    #extract features
    features = extract_features(file_path)
    
    #normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])
    
    #reshape data for CNN
    features_scaled = np.expand_dims(features_scaled, axis=2)
    
    #predict genre
    prediction = model.predict(features_scaled)
    predicted_genre = np.argmax(prediction)
    
    return predicted_genre

#run it on half the audio files
#print whether it is correct based on if the predicted genre is in the filename

#go through each file in data/Data/genres_original\[insert genre here] and run the predict_genre function on it
import os

#path to the audio files
path = r"..\data\Data\genres_original"

#list of genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

#dictionary to store the genre and the number of correct predictions

correct_predictions = 0
total_predictions = 0

for genre in genres:
    genre_path = os.path.join(path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        prediction = predict_genre(file_path)
        if genre in file:
            correct_predictions += 1
        total_predictions += 1

print(f"Accuracy: {correct_predictions/total_predictions}")