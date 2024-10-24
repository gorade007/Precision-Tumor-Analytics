import numpy as np
import cv2
from keras.models import load_model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized.astype('float32') / 255.0
    return img, img_resized

def segment_tumor(original_img):
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    return contours

def estimate_tumor_size(image_path):
    original_img, img_resized = load_and_preprocess_image(image_path)
    model = load_model('models/best_model.keras')

    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction)

    total_brain_area_mm2 = 250000

    if predicted_class == 1:
        contours = segment_tumor(original_img)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            tumor_size = area * (0.5 ** 2)  # Adjust according to your scale
            tumor_percentage = (tumor_size / total_brain_area_mm2) * 100
            return tumor_size, tumor_percentage
        else:
            return None, None
    else:
        return 0, 0  # No tumor detected

# Example usage
image_path = r'C:\Users\ATHERVA\Desktop\TOMOR\BrainTumorDetection\data\tumor\Y7.jpg'
try:
    tumor_size, tumor_percentage = estimate_tumor_size(image_path)
    if tumor_size is not None:
        print(f'Estimated Tumor Area: {tumor_size:.2f} mm²')
        print(f'Percentage of Tumor Area: {tumor_percentage:.2f}%')
    else:
        print('No tumor detected or unable to estimate size.')
except ValueError as e:
    print(e)
