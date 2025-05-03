import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
import os

# Load model function
def load_trained_model():
    model_path = 'C:/Users/chris/OneDrive/Documents/GitHub/CarbonEdgeAI/src/ai_model.h5'  # Path to your model
    model = load_model(model_path)
    return model

# Preprocess the image before prediction
def preprocess_image(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))  # Resize the image to match the model input
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize the image (optional, depending on your model)
    
    return img_array

# Function to handle data augmentation and make predictions
def predict_custom_image(image_path):
    model = load_trained_model()  # Load the trained model
    
    # Create an image data generator with augmentation (only during prediction for this example)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,          # Normalize the image
        rotation_range=20,      # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift the image width
        height_shift_range=0.2, # Randomly shift the image height
        shear_range=0.2,        # Shear transformation
        zoom_range=0.2,         # Zoom in/out
        horizontal_flip=True,   # Flip images horizontally
        fill_mode='nearest'     # Fill missing pixels
    )

    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Augment the image and make predictions
    augmented_images = datagen.flow(img_array)

    # Get the prediction on the first augmented image
    prediction = model.predict(augmented_images[0])  # Predict on the first augmented image
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability
    
    return predicted_class[0]  # Return the predicted class as an integer

# Track emissions during the model's prediction or evaluation
def track_emissions():
    tracker = EmissionsTracker()
    tracker.start()

    model = load_trained_model()  # Use this model for evaluation or prediction
    
    # Example for testing predictions with emissions tracking
    custom_image_path = "src/images/Dog.png"  # Update with your image path
    predict_custom_image(custom_image_path)  # Perform prediction and track emissions

    tracker.stop()  # Stop tracking
    emissions_data = tracker.final_emissions_data  # Use final_emissions_data to get the emissions data

    # Convert emissions data to a pandas DataFrame
    emissions_df = pd.DataFrame([emissions_data])

    # Save emissions data to CSV
    emissions_df.to_csv('emissions.csv', index=False)

    # Optionally, print emissions data to console for debugging or logging
    print(emissions_df)

# Main function to run everything
def main():
    track_emissions()

if __name__ == "__main__":
    main()
