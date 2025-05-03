import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load model function
def load_trained_model():
    # Load the saved model (either ai_model.h5 or ai_model.keras)
    model = load_model("ai_model.h5")  # Change to .keras if saved in that format
    return model

# Evaluate the model on the test data
def evaluate_model(model):
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Resize the images to 64x64 (same as training)
    x_test = tf.image.resize(x_test, (64, 64))

    # Normalize the images (same as training phase)
    x_test = x_test / 255.0

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

# Make predictions on new data
def make_predictions(model):
    # Load CIFAR-10 test data (just an example, you can use your own data)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Resize and normalize the images (same as training phase)
    x_test = tf.image.resize(x_test, (64, 64))
    x_test = x_test / 255.0

    # Predict the classes of the test images
    predictions = model.predict(x_test[:10])  # Predict on the first 10 test images

    # Display predictions
    print("Predictions (first 10 images):")
    print(predictions)

    # Show predicted class labels (based on the highest predicted probability)
    predicted_classes = np.argmax(predictions, axis=1)
    print("Predicted class labels (first 10 images):")
    print(predicted_classes)

# Main function to run everything
def main():
    # Load and evaluate the model
    model = load_trained_model()

    # Evaluate the model
    evaluate_model(model)

    # Make predictions on new data
    make_predictions(model)

if __name__ == "__main__":
    main()
