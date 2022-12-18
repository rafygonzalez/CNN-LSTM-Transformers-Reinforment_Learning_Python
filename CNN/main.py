import os
from random import shuffle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import LearningRateScheduler

num_classes = 10
num_classes_2 = 3
text_labels = [
    "motorbike",
    "dog",
    "cat"
]
#text_labels = [
#    "airplane",
#    "automobile",
#    "bird",
#    "cat",
#    "deer",
#    "dog",
#    "frog",
#    "horse",
#    "ship",
#    "truck"
#]



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

def load_images_and_labels(data_dir):

    label_dict = {
        "motorbike": 0,
        "dog": 1,
        "cat": 2
    }
    # Get the list of all subdirectories in the data directory
    subdir_list = os.listdir(data_dir)
    # Initialize empty lists to store the images and labels
    images = []
    labels = []
    # Iterate over the subdirectories
    for subdir in subdir_list:
        # Get the list of all files in the subdirectory
        if subdir != ".DS_Store":
            file_list = os.listdir(os.path.join(data_dir, subdir))
            # Iterate over the files
            for file in file_list:
                # Load the image
                print(file)
                if file != ".DS_Store":
                    image = load_img(os.path.join(data_dir, subdir, file), target_size=(32, 32))
                    # Convert the image to a NumPy array
                    image = img_to_array(image)
                    # Get the label from the subdirectory name
                    label = subdir
                    # Add the image and label to the lists
                    images.append(image)
                    if label not in labels:
                        label = label_dict[subdir]
                        labels.append(label)
    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    # Return the images and labels
    return images, labels

def local_load_data():
    # Load the data from a local directory
    x_train, y_train = load_images_and_labels("train/data")
    x_test, y_test = load_images_and_labels("test/data")

    # Preprocess the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Convert labels to categorical format
    y_train = tf.keras.utils.to_categorical(y_train, num_classes_2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes_2)

    return (x_train, y_train), (x_test, y_test)



def create_cnn_model():
    # Create the model

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(units=64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
 


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
    model.add(tf.keras.layers.Dense(num_classes_2))
    model.add(tf.keras.layers.Activation('softmax'))



    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


# Create an ImageDataGenerator object
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load the data
#(x_train, y_train), (x_test, y_test) = load_data()
(x_train, y_train), (x_test, y_test) = local_load_data()
# Create a generator for training data
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# Generate a batch of augmented images
images, labels = train_generator.next()

# Plot the generated images
plt.figure(figsize=(10, 10))
for i in range(32):
    plt.subplot(6, 6, i+1)
    plt.imshow(images[i])
    plt.title(text_labels[np.argmax(labels[i])])
    plt.axis('off')
plt.show()

# Create a CNN model
model = create_cnn_model()

def lr_schedule(epoch):

  if epoch < 32:
    return 0.001
  elif epoch < 64:
    return 0.0005
  elif epoch < 128:
    return 0.0002
  else:
    return 0.0001

# Create a LearningRateScheduler callback with the lr_schedule function
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile and fit the model, passing in the LearningRateScheduler callback
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=132, callbacks=[lr_scheduler])

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
for i in range(32):
    plt.subplot(6, 6, i+1)
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
tick_marks = np.arange(num_classes_2)
class_labels = ['class_{}'.format(i) for i in range(num_classes_2)]
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Add numerical values to the plot
for i in range(num_classes_2):
    for j in range(num_classes_2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

# Show the plot
plt.show()
