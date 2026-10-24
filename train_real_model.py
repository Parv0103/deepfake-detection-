# train_real_model.py - Complete working training script
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Constants
SEQ_LEN = 25
FEAT_DIM = 1408
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 30
MAX_IMAGES = 200  # Use 200 real + 200 fake for training

print("="*60)
print("DEEPFAKE DETECTION - REAL TRAINING")
print("="*60)

# 1. Load backbone
print("\n[1/4] Loading EfficientNetB2 backbone...")
backbone = EfficientNetB2(weights="imagenet", include_top=False, pooling="avg")
backbone.trainable = False  # Freeze backbone
print("✓ Backbone loaded")

def extract_features(img_path):
    """Extract features from a single image"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = preprocess_input(img)
    
    features = backbone.predict(np.expand_dims(img, 0), verbose=0)
    return features[0]

# 2. Load and process images
print("\n[2/4] Loading and processing images...")

# Get image paths
real_folder = "real"
fake_folder = "fake"

real_images = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES]
fake_images = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES]

print(f"   Real images found: {len(real_images)}")
print(f"   Fake images found: {len(fake_images)}")

# Extract features
X = []
y = []

print("\n   Extracting features from REAL images...")
for i, img_path in enumerate(real_images):
    feat = extract_features(img_path)
    if feat is not None:
        # Create sequence with variations
        seq = []
        for j in range(SEQ_LEN):
            # Add small random noise for variety
            noise = np.random.normal(0, 0.01, FEAT_DIM)
            seq.append(feat + noise)
        X.append(np.array(seq))
        y.append(0)
    if (i + 1) % 50 == 0:
        print(f"      Processed {i+1}/{len(real_images)}")

print("   Extracting features from FAKE images...")
for i, img_path in enumerate(fake_images):
    feat = extract_features(img_path)
    if feat is not None:
        seq = []
        for j in range(SEQ_LEN):
            noise = np.random.normal(0, 0.01, FEAT_DIM)
            seq.append(feat + noise)
        X.append(np.array(seq))
        y.append(1)
    if (i + 1) % 50 == 0:
        print(f"      Processed {i+1}/{len(fake_images)}")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n   Dataset shape: {X.shape}")
print(f"   REAL samples: {np.sum(y == 0)}")
print(f"   FAKE samples: {np.sum(y == 1)}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")

# 3. Build LSTM model
print("\n[3/4] Building LSTM model...")

model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(SEQ_LEN, FEAT_DIM)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
]

# 4. Train the model
print("\n[4/4] Training model (this will take 5-15 minutes)...")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save model
model.save('latest_EfficientNetB2.h5')
print("\n✓ Model saved as 'latest_EfficientNetB2.h5'")

# Show results
print("\n" + "="*60)
print("TRAINING RESULTS")
print("="*60)
print(f"Final Training Accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Test on a few samples
print("\n[TEST] Sample predictions:")
test_preds = model.predict(X_val[:10])
for i, (pred, actual) in enumerate(zip(test_preds.flatten(), y_val[:10])):
    pred_label = "FAKE" if pred >= 0.5 else "REAL"
    actual_label = "FAKE" if actual == 1 else "REAL"
    confidence = pred if pred >= 0.5 else 1-pred
    print(f"   {i+1}: Predicted={pred_label} ({confidence:.1%}), Actual={actual_label}")

print("\n✅ Training complete! Now run:")
print("   python evaluate.py --dataset . --model latest_EfficientNetB2.h5")
print("   streamlit run app.py")