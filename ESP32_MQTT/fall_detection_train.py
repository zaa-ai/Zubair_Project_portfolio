#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 100  # samples per window
CSV_PATH     = 'fall_dataset.csv'
MODEL_OUT    = 'fall_model.h5'

def load_data(csv_path):
    data, labels = [], []
    with open(csv_path, 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            labels.append(1 if vals[0] == 'F' else 0)
            feats = list(map(float, vals[1:]))
            data.append(np.array(feats).reshape(WINDOW_SIZE, 6))
    X = np.stack(data)
    y = tf.keras.utils.to_categorical(labels, 2)
    return train_test_split(X, y, test_size=0.2,
                            random_state=42, stratify=y)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(WINDOW_SIZE,6)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    X_train, X_val, y_train, y_val = load_data(CSV_PATH)
    model = build_model()
    model.fit(X_train, y_train,
              epochs=30,
              validation_data=(X_val, y_val))
    model.save(MODEL_OUT)
    print(f"Saved fall model to {MODEL_OUT}")

if __name__ == '__main__':
    main()
