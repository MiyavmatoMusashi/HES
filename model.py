import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("char_model.h5")

class MyCustomSymbolModel:
    def __init__(self):
        self.model = None
        self.labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '+', 11: '-', 12: 'x', 13: ',', 14: '=', 15: '÷'}


    def load_model(self):
        self.model = model

    def predict(self, img):
        preds = self.model.predict(img)
        print("Model çıktısı (preds):", preds)
        print("Preds shape:", preds.shape)
        print("Labels:", self.labels)
        if preds.shape[1] != len(self.labels):
            raise ValueError(f"Model output size {preds.shape[1]} ile label sayısı {len(self.labels)} uyuşmuyor.")
        idx = np.argmax(preds)
        print("Tahmin edilen indeks:", idx)
        return self.labels[idx]
