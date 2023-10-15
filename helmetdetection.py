import numpy as np
from PIL import Image
from tensorflow import keras

# Load a pre-trained helmet detection model
model = keras.models.load_model('helmet_detection_model.h5')

# Load and preprocess an image for inference
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))  # Adjust the size to match the model's input size
    image = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)

# Define classes for helmet and non-helmet
classes = {0: 'No Helmet', 1: 'Helmet'}

# Perform inference on an image
def predict_helmet(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    class_id = np.argmax(prediction)
    class_label = classes[class_id]
    confidence = prediction[0, class_id]
    return class_label, confidence

# Path to the image you want to test
image_path = 'test_image.jpg'

# Make a prediction
result, confidence = predict_helmet(image_path)
print(f'Predicted Class: {result}')
print(f'Confidence: {confidence * 100:.2f}%')