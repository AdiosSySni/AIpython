from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

# Загрузка эталонных изображений
def load_reference_images():
    reference_images = []
    for filename in os.listdir("./static/reference_images/"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join("reference_images", filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            reference_images.append((filename, img))
    return reference_images

# Сравнение изображений
def compare_images(query_image, reference_images):
    query_hist = cv2.calcHist([query_image], [0], None, [256], [0, 256])
    similar_images = []
    for ref_filename, ref_image in reference_images:
        ref_hist = cv2.calcHist([ref_image], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(query_hist, ref_hist, cv2.HISTCMP_CORREL)
        similar_images.append((similarity, f'/static/reference_images/{ref_filename}'))
    similar_images.sort(reverse=True)
    return similar_images[:10]

# Инициализация приложения Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'reference_images'

# Загружаем эталонные изображения при старте приложения
REFERENCE_IMAGES = load_reference_images()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Преобразуем загруженное изображение в формат OpenCV
        query_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Находим 10 наиболее похожих изображений
        similar_images = compare_images(query_image, REFERENCE_IMAGES)
        
        # Возвращаем результат в формате JSON
        return jsonify({
            'similar_images': [url for _, url in similar_images],
        }), 200
    else:
        return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)