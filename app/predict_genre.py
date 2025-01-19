import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
import librosa

class GenrePredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.scaler = StandardScaler()
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def predict(self, file_path):
        """
        This function predicts genre of audio file
        
        Parameters: 
        - file_path (str): path to audio file

        Returns:
        - dict: prediction results including genre and confidence
        """
        # Extract features
        features = extract_features(file_path)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform([features])
        
        # Reshape data for CNN
        features_scaled = np.expand_dims(features_scaled, axis=2)
        
        # Predict genre
        predictions = self.model.predict(features_scaled)
        predicted_idx = np.argmax(predictions)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'genre': self.genres[idx],
                'confidence': float(predictions[0][idx] * 100)  # Convert to percentage
            }
            for idx in top_3_idx
        ]
        
        return {
            'predicted_genre': self.genres[predicted_idx],
            'confidence': float(predictions[0][predicted_idx] * 100),  # Convert to percentage
            'top_3_predictions': top_3_predictions
        }