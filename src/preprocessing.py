import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2

def load_data():
    # Define paths to data directories
    tumor_dir = 'data/tumor'
    no_tumor_dir = 'data/no_tumor'
    
    X = []
    y = []
    
    # Load tumor images
    for filename in os.listdir(tumor_dir):
        img_path = os.path.join(tumor_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(1)  # Tumor class
    
    # Load no tumor images
    for filename in os.listdir(no_tumor_dir):
        img_path = os.path.join(no_tumor_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            X.append(img)
            y.append(0)  # No tumor class
    
    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
