import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization, SpatialDropout1D, Add, Input
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

def lr_schedule(epoch):
    initial_lr = 0.00005  # Lower initial learning rate
    if epoch < 8:
        return initial_lr * (1.5 ** epoch)  # More gradual warm-up
    elif epoch < 16:
        return 0.0008  # Lower maximum learning rate
    else:
        # Add decay phase
        return 0.0008 * (0.95 ** (epoch - 15))

def create_enhanced_model(input_shape, num_classes):
    model = Sequential([
        # Initial convolution with spatial dropout
        Conv1D(128, kernel_size=3, padding='same', activation='relu', 
               kernel_regularizer=l2(0.01), input_shape=input_shape),
        BatchNormalization(),
        SpatialDropout1D(0.3),  # Increased dropout
        MaxPooling1D(pool_size=2),
        
        # Middle layers with increased regularization
        Conv1D(256, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        SpatialDropout1D(0.3),
        MaxPooling1D(pool_size=2),
        
        Conv1D(512, kernel_size=5, padding='same', activation='relu',
               kernel_regularizer=l2(0.015)),  # Increased regularization
        BatchNormalization(),
        SpatialDropout1D(0.4),
        GlobalMaxPooling1D(),
        
        # Dense layers with stronger regularization
        Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.5),  # Increased dropout
        Dense(256, activation='relu', kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Lower initial learning rate
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
            factor=0.1,
            patience=5,
            min_lr=0.000001,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=70,
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
    factor=0.1,
    patience=5,
    min_lr=0.000001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
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