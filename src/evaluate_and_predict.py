import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# Load model function
def load_trained_model():
    model = load_model("ai_model.h5")  # Change to .keras if saved in that format
    return model

# Evaluate the model on the test data
def evaluate_model(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = tf.image.resize(x_test, (64, 64))  # Resize images to 64x64
    x_test = x_test / 255.0  # Normalize

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

# Make predictions on new data
def make_predictions(model):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = tf.image.resize(x_test, (64, 64))  # Resize images to 64x64
    x_test = x_test / 255.0  # Normalize

    predictions = model.predict(x_test[:10])  # Predict on the first 10 test images
    print("Predictions (first 10 images):")
    print(predictions)

    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted class labels (first 10 images):")
    print(predicted_classes)

# Track emissions during the model's training or evaluation
def track_emissions():
    tracker = EmissionsTracker()
    tracker.start()

    model = load_trained_model()  # Use this model for evaluation or prediction
    evaluate_model(model)

    tracker.stop()  # Stop tracking
    emissions_data = tracker.final_emissions_data  # Use final_emissions_data to get the emissions data

    # Convert emissions data to a pandas DataFrame
    emissions_df = pd.DataFrame([emissions_data])

    # Save emissions data to CSV
    emissions_df.to_csv('emissions.csv', index=False)

    # Optionally, print emissions data to console for debugging or logging
    print(emissions_df)

# Predict a custom image
def predict_custom_image(img_path):
    model = load_trained_model()

    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class[0]  # Return the predicted class

# Main function to run everything
def main():
    track_emissions()

    model = load_trained_model()
    evaluate_model(model)
    make_predictions(model)

    # Test custom image prediction
    custom_image_path = "src/images/Dog.png"  # Update with your image path
    predict_custom_image(custom_image_path)

if __name__ == "__main__":
    main()
