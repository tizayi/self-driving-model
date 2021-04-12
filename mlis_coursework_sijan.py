# %% [code]
import os

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import mean_squared_error  
from tensorflow.keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Image directories
train_dir = 'D:/data/training_data/training_data/*'
test_dir = 'D:/data/test_data/test_data/*'
labels_dir = 'D:/data/training_norm.csv'

# Labels CSV
df_labels = pd.read_csv(labels_dir)

# List image files and split for validation
dataset_list = tf.data.Dataset.list_files(train_dir, shuffle=True)
dataset_size = tf.data.experimental.cardinality(dataset_list).numpy()
val_size = int(dataset_size * 0.2)
train_list = dataset_list.skip(val_size)
validation_list = dataset_list.take(val_size)

def get_label(file_path):
    # Extract labels from csv using filename
    file_id = str(file_path).split('/')[-1].split('.')[0]
    img_id, angle, speed = df_labels[df_labels['image_id'] == int(file_id)].to_numpy().squeeze()
    if int(file_id) == int(img_id):
        return angle, speed
    raise Exception("Mismatch ID between image and csv file")

def decode_img(img, file_path):
    # Convert the compressed string to a 3D uint8 tensor and apply normalisation
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    img = tf.image.decode_png(img, channels=3)
    try:
        img = normalization_layer(img)
    except:
        raise Exception("Check this file: " + file_path)
    return img

def process_path(file_path):
    # load the raw data from the file as a string
    labels = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img , file_path)
    return img, labels

# Generate train dataset
train_features = []
train_labels = []
counter = 0
total = tf.data.experimental.cardinality(train_list).numpy()

try:
    for path in train_list.take(total):
        counter += 1
        file_path = path.numpy()
        img_features, img_labels = process_path(file_path)
        train_features.append(img_features)
        train_labels.append(img_labels)
        print("Train dataset: " + str(file_path) + " ----- " + str(counter) + "/" + str(total))
except Exception:
    print("InvalidArgumentError: Corrupted image file")
    pass
features_dataset = tf.data.Dataset.from_tensor_slices(train_features)
labels_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
train_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

# Generate validation dataset
validation_features = []
validation_labels = []
counter = 0
total = tf.data.experimental.cardinality(validation_list).numpy()
try:
    for path in validation_list.take(total):
        counter += 1
        file_path = path.numpy()
        img_features, img_labels = process_path(file_path)
        validation_features.append(img_features)
        validation_labels.append(img_labels)
        print("Validation dataset:", str(file_path) + " ----- " + str(counter) + "/" + str(total))
except:
    print("InvalidArgumentError: Corrupted image file")
    pass
features_dataset = tf.data.Dataset.from_tensor_slices(validation_features)
labels_dataset = tf.data.Dataset.from_tensor_slices(validation_labels)
validation_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

def configure_for_performance(dataset):
    # Caching, prefetching, shuffling and batching to configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=tf.data.experimental.cardinality(dataset).numpy())
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Dataset params
img_height = 240
img_width = 320
batch_size = 32

input_shape = (img_height, img_width, 3)

#train_dataset = configure_for_performance(train_dataset)
#validation_dataset = configure_for_performance(validation_dataset)

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
    
    x = Dense(units=20, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=50, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=30, activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=10, activation='elu')(x)
    x = BatchNormalization()(x)

    output_angle = Dense(units=1, name='output_angle')(x)
    output_speed = Dense(units=1, name='output_speed')(x)

    # Model compilation
    model = Model(inputs=inputs, outputs=[output_angle, output_speed], name="DriveCNN")
    optimizer = Adam(learning_rate)
    model.compile(
        loss=mean_squared_error,
        optimizer=optimizer,
        metrics=mean_squared_error)

    return model

learning_rate = 0.01
epochs = 10

model = DriveCNN(input_shape, learning_rate)
print(model.summary())

with tf.device("/GPU:0"):
    model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=1, shuffle=True, use_multiprocessing=True)