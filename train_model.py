import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === Parametreler === #
DATA_DIR = "dataset"
IMG_SIZE = 28

# === Etiket haritası === #
CLASSES = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(CLASSES)}
reverse_label_map = {v: k for k, v in label_map.items()}

# === Görsellerin yüklenmesi === #
images = []
labels = []

for label in CLASSES:
    folder_path = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Uyarı: {img_path} yüklenemedi.")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = 255 - img  # Arka planı beyaz, karakterleri siyah yap
        images.append(img)
        labels.append(label_map[label])

# === Normalize ve reshape === #
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = np.array(labels)

# === Eğitim ve test ayrımı === #
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# === Veri artırma (augmentation) === #
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# === Model (Gelişmiş CNN) === #
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

# === Derleme === #
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Eğitim === #
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=30,
    validation_data=(X_test, y_test)
)

# === Model Kaydetme === #
model.save("char_model.h5")
print("✅ Model başarıyla kaydedildi: char_model.h5")

# === Confusion Matrix === #
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=False, cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Normalize Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.show()
