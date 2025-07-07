import os
import sys
from typing import List
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'digit_classifier.h5')

def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(epochs: int = 5) -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('L')  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def predict(images: List[str]) -> List[int]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    data = np.stack([load_image(p) for p in images])
    preds = model.predict(data)
    return np.argmax(preds, axis=1).tolist()

def main():
    if len(sys.argv) < 2:
        print("Usage: python digit_classifier.py train|IMAGE_PATH [IMAGE_PATH ...]")
        return
    if sys.argv[1] == 'train':
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        train_model(epochs)
    else:
        images = sys.argv[1:]
        results = predict(images)
        for path, pred in zip(images, results):
            print(f"{path}: {pred}")

if __name__ == '__main__':
    main()
