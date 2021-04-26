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
import matplotlib.pyplot as plt

# Processing images 
def process_path(file_path):
  # load the raw data from the file as a string
  image = tf.io.read_file(file_path)
  image = tf.image.decode_png(image, channels=3)
  image = image/255
  tf.reshape(image,(1,240,320,3))
  return image

# Listing directories
test_dir =  'D:/data/test_data/test_data/'
model_output_dir='D:/data/Models'
result_path = 'D:/data/Results'

# Loading the model in
model = tf.keras.models.load_model(os.path.join(model_output_dir,'model_best.h5'))

# Loading the test data in
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

model.summary()
results = model.predict(np.array(test_images))

def results_norm(results):
  results = np.where(results < 1,results,1)
  results = np.where(results > 0,results,0)
  results[:,1] = np.round(results[:,1])
  return results

print(results)

#results=results_norm(results)

# Save to csv
idx=[s.strip('.png') for s in file_list_test]
image_id = [int(i) for i in idx]
data = {'image_id':image_id,'angle': results[:,0],'speed':results[:,1]}

df = pd.DataFrame(data=data, index=image_id)
sortdf = df.sort_index()
print(sortdf)

sortdf.to_csv(os.path.join(result_path,'results_best.csv'),index=False)



