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


#run the predict_genre function on all the files in the blues folder
#path to the audio files

#extract the genre from the path
genre = path.split("\\")[-1]
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
#run the predict_genre function on all the files in the folder

#take random 10 files from each genre
import random

for genre in genres:
    path = f".\\data\\Data\\genres_original\\{genre}"
    files = os.listdir(path)
    random_files = random.sample(files, 10)
    for file in random_files:
        file_path = os.path.join(path, file)
        predicted_genre = predict_genre(file_path)
        if genre == genres[predicted_genre]:
            print(f"Correct: {file_path}", genres[predicted_genre])
        else:
            print(f"Incorrect: {file_path}", genres[predicted_genre])
