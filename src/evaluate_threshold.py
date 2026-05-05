import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

MODEL_PATH = "/home/beyza/image-forgery-detection/models/xception_model.keras"
BASE_DIR = "/home/beyza/image-forgery-detection/dataset/processed_ela"

IMG_SIZE = (299, 299)
BATCH_SIZE = 16

val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

val_probs = model.predict(val_data).ravel()
val_true = val_data.classes

best_threshold = 0.5
best_acc = 0

print("\nThreshold sonuçları:")
for threshold in np.arange(0.30, 0.71, 0.01):
    val_pred = (val_probs >= threshold).astype(int)
    acc = accuracy_score(val_true, val_pred)

    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

    print(f"Threshold: {threshold:.2f} | Val Accuracy: {acc:.4f}")

print("\nEn iyi threshold:", round(best_threshold, 2))
print("En iyi validation accuracy:", round(best_acc, 4))

test_probs = model.predict(test_data).ravel()
test_true = test_data.classes
test_pred = (test_probs >= best_threshold).astype(int)

print("\nTEST SONUÇLARI")
print("Accuracy :", accuracy_score(test_true, test_pred))
print("Precision:", precision_score(test_true, test_pred))
print("Recall   :", recall_score(test_true, test_pred))
print("F1 Score :", f1_score(test_true, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(test_true, test_pred))

print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=["authentic", "tampered"]))
