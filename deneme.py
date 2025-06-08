import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Modeli yükle
model = tf.keras.models.load_model("char_model1.keras")
print("Model input shape:", model.input_shape)

# MNIST test verisini yükle
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize et
x_test = x_test / 255.0

# Gerekirse yeniden şekillendir
if model.input_shape[-1] == 1:
    x_test_input = x_test.reshape(-1, 28, 28, 1)
else:
    x_test_input = x_test

# Tahmin yap
predictions = model.predict(x_test_input)
predicted_labels = np.argmax(predictions, axis=1)

# 8 olarak tahmin edilen ilk 50 örnek
eight_indices = np.where(predicted_labels == 8)[0][:50]

# Görselleri çiz ve tahminleri yaz
plt.figure(figsize=(15, 8))
for i, idx in enumerate(eight_indices):
    plt.subplot(5, 10, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.axis('off')
    plt.title(str(predicted_labels[idx]), fontsize=8)
plt.tight_layout()
plt.show()
