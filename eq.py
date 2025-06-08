import cv2
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from model import MyCustomSymbolModel  # Kendi modelini buraya ekle

class sekiller:
    def __init__(self, image_path):
        self.image_path = image_path
        self.model = MyCustomSymbolModel()
        self.model.load_model()

    def resize_and_pad(self, img, width, height):
        h, w = img.shape
        scale = min(width/w, height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img = cv2.resize(img, (new_w, new_h))
        pad_w, pad_h = width - new_w, height - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded

    def process_image(self):
        original = cv2.imread(self.image_path)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Sembolleri soldan sağa doğru sırala
        sorted_contours = [c for _, c in sorted(zip(bounding_boxes, contours), key=lambda b: b[0][0])]

        symbols = []

        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < 50:  # Çok küçük alanları ele
                continue
            roi = binary[y:y+h, x:x+w]
            resized = self.resize_and_pad(roi, 28, 28)
            reshaped = resized.reshape(1, 28, 28, 1).astype("float32") / 255.0

            predicted_label = self.model.predict(reshaped)
            symbols.append(str(predicted_label))

            # Görsel için kutu ve etiket çiz
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Detected Symbols")
        plt.axis("off")
        plt.show()

        return symbols

def solve_equation(symbols):
    equation_str = "".join(symbols)  # Listeyi stringe çevir
    if '=' in equation_str:
        left_side, right_side = equation_str.split('=')
    else:
        left_side, right_side = equation_str, '0'

    try:
        left_expr = sp.sympify(left_side)
        right_expr = sp.sympify(right_side)
    except (sp.SympifyError, SyntaxError):
        return "Denklem ifadesi çözülemedi, sembolleri kontrol edin."

    equation = sp.Eq(left_expr, right_expr)
    variables = equation.free_symbols

    if variables:
        solutions = sp.solve(equation, list(variables))
        return solutions
    else:
        # Değişken yoksa doğruluk kontrolü yap
        return sp.simplify(left_expr - right_expr) == 0

if __name__ == "__main__":
    image_path = "denklem_resmi.jpg"  # İşlemek istediğin denklem resminin yolu
    sekil = sekiller(image_path)
    symbols = sekil.process_image()
    print("Tespit edilen semboller:", symbols)

    sonuc = solve_equation(symbols)
    print("Denklem çözüm sonucu:", sonuc)
