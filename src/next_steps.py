import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
from codecarbon import EmissionsTracker

# Load model function
def load_trained_model():
    # Load the saved model (either ai_model.h5 or ai_model.keras)
    model = load_model("ai_model.h5")  # Change to .keras if you used that format
    return model

# Evaluate the model on the test data
def evaluate_model(model):
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
    
    # Normalize the images (same as training phase)
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

# Track emissions during the model's training or evaluation
def track_emissions():
    # Start tracking emissions
    tracker = EmissionsTracker()
    tracker.start()

    # You can train or evaluate the model during this period.
    model = load_trained_model()  # If training, train here. If evaluating, use evaluate_model(model)

    # After training or evaluation, stop emissions tracking
    tracker.stop()
    tracker.save()  # Save the emissions data to a CSV file

# Main function to run everything
def main():
    # Track emissions during evaluation or prediction
    track_emissions()

    # Load and evaluate the model
    model = load_trained_model()

    # Evaluate the model
    evaluate_model(model)

    # Make predictions on new data
    make_predictions(model)

if __name__ == "__main__":
    main()
