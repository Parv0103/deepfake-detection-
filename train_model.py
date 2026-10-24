# train_model.py - Complete working version
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import random

# Constants
SEQ_LEN = 25
FEAT_DIM = 1408
IMG_SIZE = 224
BATCH_SIZE = 4  # Smaller batch size for lower memory
EPOCHS = 20
MAX_IMAGES_PER_CLASS = 500  # Use 500 real + 500 fake for training

print("="*60)
print("DEEPFAKE DETECTION - MODEL TRAINING")
print("="*60)

# 1. Load backbone
print("\n[1/5] Loading pre-trained backbone (EfficientNetB2)...")
backbone = EfficientNetB2(weights="imagenet", include_top=False, pooling="avg")
backbone.trainable = False  # Freeze backbone initially
print("✓ Backbone loaded")

def extract_features(img_path):
    """Extract features from a single image"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    # Resize and preprocess
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    
    # EfficientNet preprocessing
    from tensorflow.keras.applications.efficientnet import preprocess_input
    img = preprocess_input(img)
    
    # Extract features
    features = backbone.predict(np.expand_dims(img, 0), verbose=0)
    return features[0]

print("\n[2/5] Loading dataset...")
real_folder = "real"
fake_folder = "fake"

# Get image paths
real_images = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES_PER_CLASS]
fake_images = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES_PER_CLASS]

print(f"   Real images: {len(real_images)}")
print(f"   Fake images: {len(fake_images)}")

print("\n[3/5] Extracting features (this may take a while)...")
X = []
y = []

# Process real images (label=0)
print("   Processing REAL images...")
for i, img_path in enumerate(real_images):
    feat = extract_features(img_path)
    if feat is not None:
        # Create sequence of 25 by adding small variations
        sequence = []
        for j in range(SEQ_LEN):
            # Add small random noise for variation
            noise = np.random.normal(0, 0.005, FEAT_DIM)
            sequence.append(feat + noise)
        X.append(np.array(sequence))
        y.append(0)
    
    if (i + 1) % 50 == 0:
        print(f"      Processed {i+1}/{len(real_images)} real images")

# Process fake images (label=1)
print("   Processing FAKE images...")
for i, img_path in enumerate(fake_images):
    feat = extract_features(img_path)
    if feat is not None:
        # Create sequence of 25 by adding small variations
        sequence = []
        for j in range(SEQ_LEN):
            noise = np.random.normal(0, 0.005, FEAT_DIM)
            sequence.append(feat + noise)
        X.append(np.array(sequence))
        y.append(1)
    
    if (i + 1) % 50 == 0:
        print(f"      Processed {i+1}/{len(fake_images)} fake images")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"   REAL samples: {np.sum(y == 0)}")
print(f"   FAKE samples: {np.sum(y == 1)}")

# Split data
print("\n[4/5] Splitting data and building model...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")

# Build LSTM model (smaller for faster training)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, FEAT_DIM)),
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
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

print("\n[5/5] Training model...")
print("   (This may take 10-30 minutes depending on your system)")
print("-" * 60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save('latest_EfficientNetB2.h5')
print("\n✓ Model saved as 'latest_EfficientNetB2.h5'")

# Print results
print("\n" + "="*60)
print("TRAINING RESULTS")
print("="*60)
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Training Accuracy:   {final_acc:.4f} ({final_acc*100:.1f}%)")
print(f"Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
print(f"Training Loss:       {final_loss:.4f}")
print(f"Validation Loss:     {final_val_loss:.4f}")

# Test predictions on validation set
print("\n[TEST] Sample predictions on validation set:")
val_preds = model.predict(X_val[:10])
for i, (pred, actual) in enumerate(zip(val_preds.flatten(), y_val[:10])):
    pred_class = "FAKE" if pred >= 0.5 else "REAL"
    actual_class = "FAKE" if actual == 1 else "REAL"
    print(f"   Sample {i+1}: Pred={pred:.4f} ({pred_class}) | Actual={actual_class}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nNext steps:")
print("1. Run: python evaluate.py --dataset . --model latest_EfficientNetB2.h5")
print("2. Run: streamlit run app.py")