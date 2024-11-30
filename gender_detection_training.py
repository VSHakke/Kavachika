from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
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

# Set initial parameters
epochs = 20  # Increased epoch for better training
lr = 1e-4  # Lower learning rate to avoid overfitting
batch_size = 32  # Smaller batch size
img_dims = (96, 96, 3)  # Dimensions of the input images (width, height, depth)

data = []
labels = []

# Load image files from the dataset
image_files = [f for f in glob.glob(r'E:\PLACEMENT\Project Section\Kavachika\gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

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
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build the model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Compile the model
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Save the model to disk
model.save('gender_detection.h5')

# Plot the training and validation loss/accuracy
plt.style.use("ggplot")
plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc", color="green")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", color="orange")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss", color="red")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", color="blue")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Print accuracy values
print("Final Training Accuracy:", H.history['accuracy'][-1])
print("Final Validation Accuracy:", H.history['val_accuracy'][-1])
