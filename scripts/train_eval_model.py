# scripts/train_eval_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
from prepare_data import X_train, y_train, X_val, y_val
from tqdm.keras import TqdmCallback  # adds nice progress bar

# Detect GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU detected and memory growth enabled.")
else:
    print("⚠️ No GPU detected. Training will use CPU.")

# Build a small CNN
model = models.Sequential([
    layers.Input(shape=(8,8,12)),
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # output: evaluation in pawns
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=64,
    verbose=0,            # hide default output
    callbacks=[TqdmCallback(verbose=1)]
)

# Save model 
model.save("data/processed/eval_model.keras")
print("✅ Model trained and saved to data/processed/eval_model.keras")
