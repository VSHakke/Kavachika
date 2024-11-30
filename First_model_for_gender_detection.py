# Gender_detection : Import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
# filename pattern

# Set initial parameters
epochs = 100
lr = 1e-3  # Learning rate
batch_size = 64
img_dims = (96, 96, 3)  # Dimensions of the input images (width, height, depth)

data = []
labels = []

# Load image files from the dataset
# E:/PLACEMENT/Project Section/Kavachika/
image_files = [f for f in glob.glob(r'E:\PLACEMENT\Project Section\Kavachika\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)  # Shuffle the dataset for randomness

# Convert images to arrays and label the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))  # Resize image to 96x96
    image = img_to_array(image)  # Convert image to array
    data.append(image)

    # Label the image as '1' for "woman" and '0' for "man"
    label = img.split(os.path.sep)[-2]  # Extract the label from the directory name
    label = 1 if label == "woman" else 0
    labels.append([label])

# Pre-processing: normalize image data and convert labels to NumPy arrays
data = np.array(data, dtype="float") / 255.0  # Normalize pixel values to [0, 1]
labels = np.array(labels)

# Split the dataset into training and testing sets (80% training, 20% testing)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding (binary classification: 2 classes)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# Augment the dataset to create more diverse images
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

# Define the CNN model architecture
def build(width, height, depth, classes):
    model = Sequential()  # Initialize the model
    inputShape = (height, width, depth)  # Define input shape
    chanDim = -1  # Define channel dimension (last axis for "channels_last")
# Channels-first format ("channels_first"): The tensor shape would be (channels, height, width). This means the channel dimension comes first.
# Channels-last format ("channels_last"): The tensor shape would be (height, width, channels). This means the channel dimension comes last.
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # First CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # Second CONV => RELU => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Third CONV => RELU => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer (binary classification with sigmoid activation)
    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build the model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Compile the model using Adam optimizer (with no decay)
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model using the updated `fit()` method
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Save the model to disk
model.save('gender_detection.h5')

# Plot the training/validation loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save the plot to disk
plt.savefig('plot.png')
plt.show()
