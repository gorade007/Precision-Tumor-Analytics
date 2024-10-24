import numpy as np
import cv2
from keras.models import load_model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found or unable to read: {image_path}")
    img_resized = cv2.resize(img, (128, 128)) / 255.0
    return img, img_resized

def segment_brain(original_img):
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use adaptive thresholding for better results
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours for the brain area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour corresponds to the brain
    brain_contour = max(contours, key=cv2.contourArea) if contours else None
    return brain_contour

def segment_tumor(original_img):
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use adaptive thresholding for better results in varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to enhance segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to remove small noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]  # Adjust area threshold as needed

    return contours

def estimate_tumor_size(image_path):
    original_img, img_resized = load_and_preprocess_image(image_path)
    model = load_model('models/best_model.keras')

    # Expand dims to match model input shape (1, 128, 128, 3)
    img_input = np.expand_dims(img_resized, axis=0)

    # Check shape before predicting
    print(f"Input shape for prediction: {img_input.shape}")

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
            print(f'Estimated Tumor Area: {tumor_size:.2f} mm²')

            # Calculate percentage of tumor area with respect to total brain area
            tumor_percentage = (tumor_size / total_brain_area) * 100 if total_brain_area > 0 else 0
            print(f'Total Brain Area: {total_brain_area:.2f} mm²')
            print(f'Percentage of Tumor Area: {tumor_percentage:.2f}%')
        else:
            print('No tumor contours found, unable to estimate tumor size.')
    else:
        print('No tumor detected.')

# Example usage
image_path = r'C:\Users\ATHERVA\Desktop\TOMOR\BrainTumorDetection\data\tumor\Y1.jpg'
estimate_tumor_size(image_path)
