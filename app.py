import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # Correct import for secure_filename
from keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load your pre-trained model
model = load_model('models/best_model.keras')

# Create uploads directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    return img, img_resized

def segment_brain(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brain_contour = max(contours, key=cv2.contourArea) if contours else None
    return brain_contour

def segment_tumor(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]  # Adjust area threshold
    return contours

def estimate_tumor_size(image_path):
    original_img, img_resized = load_and_preprocess_image(image_path)

    # Expand dims to match model input shape (1, 128, 128, 3)
    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)

    # Estimate brain area
    brain_contour = segment_brain(original_img)
    total_brain_area = cv2.contourArea(brain_contour) if brain_contour is not None else 0

    if predicted_class == 1:
        contours = segment_tumor(original_img)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            tumor_size = area * (0.5 ** 2)  # Adjust based on scaling factor
            # Calculate percentage of tumor area with respect to total brain area
            tumor_percentage = (tumor_size / total_brain_area) * 100 if total_brain_area > 0 else 0
            return tumor_size, total_brain_area, tumor_percentage
        else:
            return None, None, None  # No tumor found
    else:
        return None, None, None  # No tumor detected

@app.route('/')
def index():
    return render_template('index.html', tumor_size=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', tumor_size=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', tumor_size=None)

    # Save the file
    image_path = os.path.join('uploads', secure_filename(file.filename))
    file.save(image_path)

    # Call your estimation function
    tumor_size, total_brain_area, tumor_percentage = estimate_tumor_size(image_path)

    return render_template('index.html', 
                           tumor_size=tumor_size, 
                           total_brain_area=total_brain_area,
                           tumor_percentage=tumor_percentage)

if __name__ == '__main__':
    app.run(debug=True)
