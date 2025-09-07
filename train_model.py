# Import necessary libraries
import os  # For file handling
import csv  # For reading CSV files
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.preprocessing import LabelEncoder  # Encode string labels to numbers
from tensorflow.keras.models import Sequential  # Sequential model for LSTM
from tensorflow.keras.layers import LSTM, Dense, Dropout  # LSTM and Dense layers
from tensorflow.keras.utils import to_categorical  # Convert labels to one-hot

# === SETTINGS ===
DATA_DIR = "sign_language_data"  # Folder containing CSV files of hand landmarks
TIMESTEPS = 1  # Each sample is still 1 frame (no sequences yet)
FEATURES = 126  # 2 hands * 21 landmarks * 3 coordinates (x, y, z)

# === LOAD DATA ===
X = []  # Input features
y = []  # Labels

# Loop through CSV files and read data
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")  # Get label from file name
        with open(os.path.join(DATA_DIR, file)) as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header row
            for row in reader:
                coords = list(map(float, row[1:]))  # Convert coordinates to float

                # Pad missing hand coordinates if only 1 hand was captured
                if len(coords) < FEATURES:
                    coords += [0.0] * (FEATURES - len(coords))

                X.append(coords)
                y.append(label)

# Convert lists to NumPy arrays for model processing
X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)
print("Labels:", set(y))

# === ENCODE LABELS ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Convert string labels to integers
y_onehot = to_categorical(y_encoded)  # Convert integers to one-hot vectors

# Save label classes for later use (e.g., when predicting)
np.save("label_classes.npy", encoder.classes_)

# Reshape X for LSTM: [samples, timesteps, features]
X = X.reshape((X.shape[0], TIMESTEPS, FEATURES))

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=0.2,  # 20% test data
    random_state=42,
    stratify=y_onehot  # Keep label distribution consistent
)

# === BUILD MODEL ===
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(TIMESTEPS, FEATURES)),  # LSTM layer
    Dropout(0.3),  # Prevent overfitting
    Dense(64, activation="relu"),  # Fully connected layer
    Dense(y_onehot.shape[1], activation="softmax")  # Output layer (softmax for multi-class)
])

# Compile the model
model.compile(
    optimizer="adam",  # Adaptive optimizer
    loss="categorical_crossentropy",  # Multi-class classification loss
    metrics=["accuracy"]
)

# Show model summary
model.summary()

# === TRAIN MODEL ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),  # Validate on test set
    epochs=30,  # Number of training epochs
    batch_size=32  # Number of samples per batch
)

# === SAVE MODEL ===
model.save("sign_model.h5")  # Save trained model for later use
print("✅ Model saved as sign_model.h5")
