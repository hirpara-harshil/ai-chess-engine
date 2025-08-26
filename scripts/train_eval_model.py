# scripts/train_eval_model.py  (minimal changes)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from prepare_data import X_train, y_train, X_val, y_val
from tqdm.keras import TqdmCallback

# clip extreme SF scores (in pawns) to reduce outliers
y_train_clip = np.clip(y_train, -10.0, 10.0)  # +/- 1000cp
y_val_clip   = np.clip(y_val,   -10.0, 10.0)

model = models.Sequential([
    layers.Input(shape=(8,8,12)),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)   # pawns
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.Huber(delta=0.5),
              metrics=['mae'])

model.fit(
    X_train, y_train_clip,
    validation_data=(X_val, y_val_clip),
    epochs=15,
    batch_size=128,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)]
)

model.save("data/processed/eval_model.keras")
print("✅ Saved model.")
