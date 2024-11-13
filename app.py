from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2

app = Flask(__name__)
model = load_model('my_trained_model.h5')  # Load mô hình đã huấn luyện

# Hàm tiền xử lý ảnh trước khi đưa vào mô hình
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    equalized_img = cv2.equalizeHist(img)
    equalized_img_color = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2RGB)
    img_resize = cv2.resize(equalized_img_color, (224, 224))
    img_array = np.array(img_resize) / 255.0  # Chuẩn hóa ảnh
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)
            predicted_labels = np.argmax(prediction, axis=1)
            confidence = np.max(prediction) * 100
            
            # Giả sử đầu ra là một lớp duy nhất với xác suất dự đoán
            result = "Pneumonia" if predicted_labels==1 else "Normal"
            return render_template("index.html", prediction=result, confidence=confidence, img_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
