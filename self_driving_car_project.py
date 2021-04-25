

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import fnmatch
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.metrics import mean_squared_error  
from tensorflow.keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.io import imread_collection,pop



"""Listing directories of data"""

train_dir = 'D:/data/training_data/training_data/'
test_dir = 'D:/data/test_data/test_data/'
labels_dir = 'D:/data/training_norm.csv'

file_list = os.listdir(train_dir)
df_labels = pd.read_csv(labels_dir)

image_paths = []
pattern = "*.png"
for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(train_dir,filename))

"""Getting labels """

def get_label(file_path):
    # Extract labels from csv using filename
    file_id = str(file_path).split('/')[-1].split('.')[0]
    img_id, angle, speed = df_labels[df_labels['image_id'] == int(file_id)].to_numpy().squeeze()
    if int(file_id) == int(img_id):
        return angle, speed
    raise Exception("Mismatch ID between image and csv file")

X_train, X_valid = train_test_split( image_paths, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

def DriveCNN(input_shape, learning_rate):
    # Input layers
    inputs = Input(shape=input_shape, name='img_input')

    x = Conv2D(filters=3, kernel_size=(3,3), activation='elu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(filters=9, kernel_size=(9,9), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(filters=6, kernel_size=(6,6), activation='elu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    
    x = Flatten()(x)
    
    x = Dense(units=100, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=50, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=30, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=10, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=2)(x)

    # Model compilation
    model = Model(inputs=inputs, outputs=x, name="DriveCNN")
    optimizer = Adam(learning_rate)
    model.compile(
        loss=mean_squared_error,
        optimizer=optimizer)

    return model

learning_rate = 0.05
epochs = 7
img_height = 240
img_width = 320

input_shape = (img_height, img_width, 3)
model = DriveCNN(input_shape, learning_rate)

# Image data generator

def image_data_generator(image_paths, batch_size):
    while True:
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            labels = get_label(image_path)
            image = process_path(image_path)
            batch_images.append(image)
            batch_labels.append(labels)
            
        yield( np.asarray(batch_images), np.asarray(batch_labels))

def process_path(file_path):
    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = image/255
    tf.reshape(image,(1,240,320,3))
    return image

with tf.device("/GPU:0"):
    history = model.fit(image_data_generator( X_train, batch_size=100),
                              steps_per_epoch=50,
                              epochs=epochs,
                              validation_data = image_data_generator( X_valid, batch_size=100),
                              validation_steps=25,
                              verbose=1,
                              shuffle=1)

"""Saving the model and making predictions"""

model_output_dir='D:/data'
model.save(os.path.join(model_output_dir,'car_model_3.h5'))


'''
# Creating test data predictions
file_list_test = os.listdir(test_dir)
test_paths = []
pattern = "*.png"
for filename in file_list_test:
    if fnmatch.fnmatch(filename, pattern):
        test_paths.append(os.path.join(test_dir,filename))

test_images = [] 
for path in test_paths:
  img_feature = process_path(path)
  test_images.append(img_feature)

results = model.predict(np.array(test_images))
'''