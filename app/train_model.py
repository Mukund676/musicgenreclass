import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"..\data\Data\features_3_sec.csv")

# Print unique labels to verify
print("Unique labels:", data['label'].unique())

# Select relevant features
features = data.drop(columns=['filename', 'length', 'label'], axis=1)
labels = data['label']

# Encode labels as categorical data
labels = pd.Categorical(labels).codes
labels = to_categorical(labels)

# Learning rate schedule for warm-up
def lr_schedule(epoch):
    initial_lr = 0.0001  # Lower initial learning rate
    if epoch < 5:
        return initial_lr
    elif epoch < 10:
        return initial_lr * 5  # Gradual increase
    else:
        return 0.001  # Normal learning rate

def create_enhanced_model(input_shape, num_classes):
    model = Sequential([
        # Initial convolution with more filters
        Conv1D(128, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),  # Reduced dropout
        
        # Increased filters in middle layers
        Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(512, kernel_size=3, padding='same', activation='relu'),  # Doubled filters
        BatchNormalization(),
        GlobalMaxPooling1D(),  # Changed to GlobalMaxPooling1D
        Dropout(0.2),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  # Increased units
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower initial learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def cross_validate_model(X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f'Fold {fold + 1}/{n_splits}')
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
        X_val_scaled = np.expand_dims(X_val_scaled, axis=2)
        
        model = create_enhanced_model(input_shape=(X_train_scaled.shape[1], 1), num_classes=y.shape[1])
        
        # Enhanced callbacks
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,  # Increased patience
            min_lr=0.00001,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        )
        
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=70,  # Increased epochs
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=[reduce_lr, early_stopping, lr_scheduler],
            verbose=1
        )
        
        score = model.evaluate(X_val_scaled, y_val, verbose=0)
        scores.append(score[1])
        
    return np.mean(scores), np.std(scores)

# Print input shape for debugging
print("\nInput shape before model creation:", features.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# Print shape after preprocessing
print("Training data shape after preprocessing:", X_train_scaled.shape)

# Create and train model
model = create_enhanced_model(input_shape=(X_train_scaled.shape[1], 1), num_classes=labels.shape[1])

# Print model summary
model.summary()

# Define callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train model
history = model.fit(
    X_train_scaled, 
    y_train,
    epochs=70,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[reduce_lr, early_stopping, lr_scheduler]
)

# Save model
model.save(r'models\enhanced_music_genre_cnn.h5')

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Loss: {test_loss:.2f}")

# Predictions
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
y_true = np.argmax(y_test, axis=1)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('enhanced_training_history.png')
plt.show()
plt.close()

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data['label'].unique(), 
            yticklabels=data['label'].unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('enhanced_confusion_matrix.png')
plt.show()
plt.close()

# Cross-validation
print("\nPerforming cross-validation...")
mean_accuracy, std_accuracy = cross_validate_model(features.values, labels)
print(f"\nCross-validation results:")
print(f"Mean accuracy: {mean_accuracy:.2f} (+/- {std_accuracy:.2f})")