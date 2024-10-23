# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import fnmatch
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras import Model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Listing directories of data

train_dir = 'D:/data/training_data/training_data/'
test_dir = 'D:/data/test_data/test_data/'
labels_dir = 'D:/data/training_norm.csv'
model_output_dir='D:/data/Models'

file_list = os.listdir(train_dir)
df_labels = pd.read_csv(labels_dir)

image_paths = []
pattern = "*.png"
for filename in file_list:
    if fnmatch.fnmatch(filename, pattern):
        image_paths.append(os.path.join(train_dir,filename))

# Getting labels

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

    x = Conv2D(filters=32, kernel_size=(3,3), activation='elu')(inputs)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), activation='elu')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    
    x = Dense(units=500, activation='elu')(x)
    x = Dense(units=100, activation='elu')(x)
    x = Dense(units=50, activation='elu')(x)
    x = Dense(units=10, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=2)(x)
    
    # Model compilation
    model = Model(inputs=inputs, outputs=x, name="DriveCNN")
    optimizer = Adam(learning_rate)
    model.compile(
        loss=MeanSquaredError,
        optimizer=optimizer)
    
    return model

learning_rate = 0.001
epochs = 30
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

# Fitting model 
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir,'model_call_bk.h5'), verbose=1, save_best_only=True)

with tf.device("/GPU:0"):
    history = model.fit(image_data_generator( X_train, batch_size=100),
                              steps_per_epoch=100,
                              epochs=epochs,
                              validation_data = image_data_generator( X_valid, batch_size=100),
                              validation_steps=20,
                              verbose=1,
                              shuffle=1,
                              callbacks=[checkpoint_callback])


# Plotting loss
plt.plot(history.history['loss'],color='blue')
plt.plot(history.history['val_loss'],color='red')
plt.legend(["training loss", "validation loss"])
plt.show()

# Saving the model
model.save(os.path.join(model_output_dir,'car_model_67.h5'))
