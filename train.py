import tensorflow as tf


# --- 1. Ayarlar ---
dataset_path = "dataset/"   # Görüntülerin klasör yapısı: dataset/sınıf_ismi/dosya.png
img_height = 28
img_width = 28
batch_size = 32

# --- 2. Veri Setini Yükleme ---
# image_dataset_from_directory, klasör isimlerini otomatik sınıf etiketi olarak kullanır.
# validation_split ile veriyi %80 eğitim, %20 doğrulama olarak ayırıyoruz.
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,      # %20 test (doğrulama) ayır
    subset="training",         # eğitim seti seç
    seed=123,                  # bölme işleminin sabit olması için seed ver
    image_size=(img_height, img_width),  # Görüntüyü model girişine göre yeniden boyutlandır
    batch_size=batch_size,
    color_mode="grayscale"     # Tek kanallı gri tonlama (1 kanal)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",       # doğrulama (test) seti
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="grayscale"
)

# Sınıf isimlerini yazdırıyoruz (örnek: ['0', '1', ..., '+', '-', '*', '/', '=', 'x'])
class_names = train_ds.class_names
print("Sınıflar:", class_names)

# --- 3. Veri Normalizasyonu ---
# Tensorflow dataset uint8 (0-255) tipinde veriyor, biz 0-1 aralığına dönüştürüyoruz.
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Performansı artırmak için veri setini önceden hazırlayalım
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 4. Model Mimarisi ---
num_classes = len(class_names)  # Klasör sayısı kadar çıktı nöronu

def create_model(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # Ek katman
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Aşırı öğrenmeyi engeller
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model(num_classes)

# --- 5. Modeli Eğitme ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()


# --- 6. Modeli Kaydetme ---
model.save("char_model.h5")
print("Model başarıyla kaydedildi: char_model.h5")
