import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
num_classes = 10

text_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def load_data():
    # Load the data from a directory
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to categorical format
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def create_cnn_model():
    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# Create an ImageDataGenerator object
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the data
(x_train, y_train), (x_test, y_test) = load_data()

# Create a generator for training data
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# Generate a batch of augmented images
images, labels = train_generator.next()

# Plot the generated images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.title(text_labels[np.argmax(labels[i])])
    plt.axis('off')
plt.show()

# Create a CNN model
model = create_cnn_model()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit the model using the generator
model.fit_generator(
    train_generator, steps_per_epoch=len(x_train) // 32, epochs=3)

# Make predictions on the test set
predictions = model.predict(x_test)

# Calculate the accuracy of the predictions
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
print('Test accuracy:', accuracy)

# Save the model
model.save('model.h5')
#model = keras.models.load_model('model.h5')

# Get the class labels from the predicted probabilities
predicted_labels = np.argmax(predictions, axis=1)

# Get the class labels from the ground truth
true_labels = np.argmax(y_test, axis=1)




# Plot the test images with their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[i])
    plt.title(text_labels[predicted_labels[i]])
    plt.axis('off')
plt.show()

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Create a figure
plt.figure(figsize=(8, 8))

# Plot the confusion matrix as an image
plt.imshow(cm, cmap='Blues')

# Add title and axis labels
plt.title('Confusion matrix')
plt.ylabel('True labels')
plt.xlabel('Predicted labels')

# Add tick marks and class labels
tick_marks = np.arange(num_classes)
class_labels = ['class_{}'.format(i) for i in range(num_classes)]
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Add numerical values to the plot
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha='center', va='center')

# Show the plot
plt.show()
