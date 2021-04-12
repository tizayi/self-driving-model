import numpy as np
import os
import PIL
import PIL.Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Image directories
train_dir = 'D:/data/training_data/training_data/'
test_dir = 'D:/data/test_data/test_data/'
labels_dir = 'D:/data/training_norm.csv'

# Labels CSV
df_labels = pd.read_csv(labels_dir)

img_height = 240
img_width = 320
batch_size = 32

image_root = Path(train_dir)
list_ds = tf.data.Dataset.list_files(str(image_root/'*.png'))
for f in list_ds:
  image = tf.io.read_file(f)
  image = tf.io.decode_png(image)
  