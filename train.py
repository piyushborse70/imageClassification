import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


def train_model(data_path: str, model_path: str):
    print("Starting model training...")
    # Load image data
    X, y = [], []
    labels = ["dog", "horse", "cat", "elephant", "sheep"]
    for label in labels:
        folder = os.path.join(data_path, label)
        for img_file in glob.glob(os.path.join(folder, "*.jpeg")):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (64, 64))
            X.append(img.flatten())
            y.append(label)
    X, y = np.array(X), np.array(y)

    print("Data loaded successfully")

    if len(X) == 0:
        raise ValueError("No images found; please check dataset path and images.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Data split successfully")

    # Create pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", probability=True)),
        ]
    )

    print("Pipeline created successfully")

    # Train model
    pipeline.fit(X_train, y_train)

    print("Model trained successfully")

    # Test model
    y_pred = pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    print("Model tested successfully")

    # Save model
    joblib.dump(pipeline, model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    train_model("dataset_images", "model/svm_model.pkl")
