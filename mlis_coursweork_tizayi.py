import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Image directories
train_dir = 'D:/data/training_data/training_data/*'
test_dir = 'D:/data/test_data/test_data/*'
labels_dir = 'D:/data/training_norm.csv'

# Labels CSV
df_labels = pd.read_csv(labels_dir)
print(df_labels)