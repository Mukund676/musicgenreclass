import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

# Load data
data = pd.read_csv(r"data\Data\features_3_sec.csv")

#select relevant features
features = data.drop(columns=['filename', 'length', 'label'], axis=1)
labels = data['label']

#Encode labels as categorical data
labels = pd.Categorical(labels).codes
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Reshape data for CNN
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

#Define CNN model

#Add Convolutional layer

#add more layers for better learning

#flatten the output

#add fully connected layers

#output layer (number of neurons equals number of genres)

#compile model

#train model

#save the trained model