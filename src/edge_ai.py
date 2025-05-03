import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_model():
    return tf.keras.models.load_model("ai_model.h5")

def make_prediction(model, img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    return class_index

# Run prediction on an Edge device
if __name__ == "__main__":
    model = load_model()
    img_path = "test_image.jpg"  # Replace with actual image file
    prediction = make_prediction(model, img_path)
    print(f"Predicted class index: {prediction}")
