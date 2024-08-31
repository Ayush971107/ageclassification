# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from zipfile import ZipFile
# import time
# from datetime import datetime
# import itertools
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
# from tensorflow.keras import utils
# from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
# # Setting random seeds to reduce the amount of randomness in the neural net weights and results.
# # The results may still not be exactly reproducible.
# np.random.seed(42)
# tf.random.set_seed(42)

# train_aug_df = pd.read_csv('/teamspace/studios/this_studio/train_aug.csv')
# test_df = pd.read_csv('/teamspace/studios/this_studio/test.csv')

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# # Converting the filenames and target class labels into lists for augmented train and test datasets.
# train_aug_df = train_aug_df[train_aug_df['Filepath'].apply(os.path.exists)]
# train_aug_filenames_list = list(train_aug_df['Filepath'])
# train_aug_labels_list = list(train_aug_df['Label'])

# test_filenames_list = list(test_df['Filepath'])
# test_labels_list = list(test_df['Label'])
     

# # Creating tensorflow constants of filenames and labels for augmented train and test datasets from the lists defined above.

# train_aug_filenames_tensor = tf.constant(train_aug_filenames_list)
# train_aug_labels_tensor = tf.constant(train_aug_labels_list)

# test_filenames_tensor = tf.constant(test_filenames_list)
# test_labels_tensor = tf.constant(test_labels_list)


# # Defining a function to read the image, decode the image from given tensor and one-hot encode the image label class.
# # Changing the channels para in tf.io.decode_jpeg from 3 to 1 changes the output images from RGB coloured to grayscale.

# num_classes = 6

# def _parse_function(filename, label):
    
#     image_string = tf.io.read_file(filename)
#     image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
#     # image_resized = tf.image.resize(image_decoded, [200, 200])
#     label = tf.one_hot(label, num_classes)

#     return image_decoded, label


# train_aug_dataset = tf.data.Dataset.from_tensor_slices((train_aug_filenames_tensor, train_aug_labels_tensor))
# train_aug_dataset = train_aug_dataset.map(_parse_function)
# train_aug_dataset = train_aug_dataset.batch(512)    
# test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
# test_dataset = test_dataset.map(_parse_function)
# test_dataset = test_dataset.batch(512)  

# final_cnn = Sequential()

# # Input layer with 32 filters, followed by an AveragePooling2D layer.
# final_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)))    
# final_cnn.add(AveragePooling2D(pool_size=(2,2)))
# # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
# final_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
# final_cnn.add(AveragePooling2D(pool_size=(2,2)))
# final_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
# final_cnn.add(AveragePooling2D(pool_size=(2,2)))
# final_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
# final_cnn.add(AveragePooling2D(pool_size=(2,2)))
# # GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
# final_cnn.add(GlobalAveragePooling2D())
# final_cnn.add(Dense(132, activation='relu'))
# final_cnn.add(Dense(6, activation='softmax'))
# final_cnn.summary()


# final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     

# # Creating a TensorBoard callback object and saving it at the desired location.

# tensorboard = TensorBoard(log_dir=f"predage/final_cnn")
     


# # Creating a ModelCheckpoint callback object to save the model according to the value of val_accuracy.

# checkpoint = ModelCheckpoint(filepath=f"predage/final_cnn_model_checkpoint.keras",
#                              monitor='val_accuracy',
#                              save_best_only=True,
#                              save_weights_only=False,
#                              verbose=1
#                             )
     

# # Fitting the above created CNN model.

# final_cnn_history = final_cnn.fit(train_aug_dataset,
#                                   batch_size=512,
#                                   validation_data=test_dataset,
#                                   epochs=60,
#                                   callbacks=[tensorboard, checkpoint],
#                                   shuffle=False    # shuffle=False to reduce randomness and increase reproducibility
#                                  )



import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# Setting random seeds to reduce randomness in the neural net weights and results.
np.random.seed(42)
tf.random.set_seed(42)

# Load the datasets
train_aug_df = pd.read_csv('/teamspace/studios/this_studio/train_aug.csv')
test_df = pd.read_csv('/teamspace/studios/this_studio/test.csv')

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Filter out rows with nonexistent file paths
train_aug_df = train_aug_df[train_aug_df['Filepath'].apply(lambda x: os.path.exists(x.strip()))]

# Convert file paths and labels to lists
train_aug_filenames_list = list(train_aug_df['Filepath'])
train_aug_labels_list = list(train_aug_df['Label'])

test_filenames_list = list(test_df['Filepath'])
test_labels_list = list(test_df['Label'])

# Convert lists to TensorFlow constants
train_aug_filenames_tensor = tf.constant(train_aug_filenames_list)
train_aug_labels_tensor = tf.constant(train_aug_labels_list)

test_filenames_tensor = tf.constant(test_filenames_list)
test_labels_tensor = tf.constant(test_labels_list)

# Define the number of classes
num_classes = 6

# Define a function to parse the image and label
def _parse_function(filename, label):
    image_string = tf.io.read_file(filename)
    try:
        image_decoded = tf.io.decode_jpeg(image_string, channels=1)  # Use channels=1 for grayscale
    except tf.errors.NotFoundError:
        print(f"Warning: File not found {filename}")
        # Return a zero-filled image of the expected shape
        image_decoded = tf.zeros([200, 200, 1], dtype=tf.uint8)
    except tf.errors.InvalidArgumentError:
        print(f"Warning: Could not decode image {filename}")
        # Return a zero-filled image of the expected shape
        image_decoded = tf.zeros([200, 200, 1], dtype=tf.uint8)

    label = tf.one_hot(label, num_classes)
    return image_decoded, label

# Create TensorFlow datasets
train_aug_dataset = tf.data.Dataset.from_tensor_slices((train_aug_filenames_tensor, train_aug_labels_tensor))
train_aug_dataset = train_aug_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
train_aug_dataset = train_aug_dataset.batch(512).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
test_dataset = test_dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(512).prefetch(tf.data.AUTOTUNE)

# Define the CNN model
final_cnn = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(200, 200, 1)),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=64, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=128, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    Conv2D(filters=256, kernel_size=3, activation='relu'),
    AveragePooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(132, activation='relu'),
    Dense(num_classes, activation='softmax')
])

final_cnn.summary()

# Compile the model
final_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Setup TensorBoard and ModelCheckpoint callbacks
tensorboard = TensorBoard(log_dir="predage/final_cnn")
checkpoint = ModelCheckpoint(filepath="predage/final_cnn_model_checkpoint.keras",
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=False,
                             verbose=1)

# Train the model
final_cnn_history = final_cnn.fit(train_aug_dataset,
                                  validation_data=test_dataset,
                                  epochs=60,
                                  callbacks=[tensorboard, checkpoint],
                                  shuffle=False)
