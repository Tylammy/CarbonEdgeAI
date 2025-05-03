import tensorflow as tf
from tensorflow.keras import layers, models
from codecarbon import EmissionsTracker

# Create a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # Assuming 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset (CIFAR-10 for this example)
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Resize the images to 64x64
    x_train = tf.image.resize(x_train, (64, 64))
    x_test = tf.image.resize(x_test, (64, 64))
    
    # Normalize the images
    x_train, x_test = x_train / 255.0, x_test / 255.0 
    
    return x_train, y_train, x_test, y_test

# Train the model
def train_model():
    x_train, y_train, x_test, y_test = load_data()

    model = create_model()
    model.summary()

    # Track emissions during training
    tracker = EmissionsTracker()
    tracker.start()

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Stop emissions tracking and automatically save the data
    tracker.stop()  # This saves the emissions data

    # Save model
    model.save("ai_model.h5")

# Run training
if __name__ == "__main__":
    train_model()
