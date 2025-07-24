#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_SIZE = 100
CSV_PATH     = 'terrain.csv'
MODEL_OUT    = 'terrain_cnn.h5'

def load_and_split(csv_path):
    raw = np.loadtxt(csv_path, delimiter=',')
    labels = raw[:,0].astype(int)
    X = raw[:,1:].reshape(-1, WINDOW_SIZE, 6)
    y = tf.keras.utils.to_categorical(labels, 4)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=0.5,
        stratify=y_tmp.argmax(1),
        random_state=42
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(WINDOW_SIZE,6)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation='softmax'),
    ])
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    return model

def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_and_split(CSV_PATH)
    model = build_model()
    model.fit(X_tr, y_tr,
              epochs=30,
              batch_size=32,
              validation_data=(X_val, y_val))

    y_pred = model.predict(X_te).argmax(axis=1)
    y_true = y_te.argmax(axis=1)

    print("=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=['Smooth','Rough','Slope','Collision']
    ))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred))

    model.save(MODEL_OUT)
    print(f"Saved terrain model to {MODEL_OUT}")

if __name__ == '__main__':
    main()
