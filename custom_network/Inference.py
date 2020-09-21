import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import time
import pathlib
from tensorflow.keras import models, layers, datasets, optimizers
from tensorflow.keras import  Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, MaxPooling2D, Dropout, Flatten, Dense





#PREDICTION OVER NEW IMAGES
our_model = tf.keras.models.load_model('new_model.h5')
class_names = ['covid','non-covid']

dataset_covid = "test/covid"
data_dir_covid = pathlib.Path(dataset_covid)

dataset_nocovid = "test/non"
data_dir_nocovid = pathlib.Path(dataset_nocovid)

# create a list of elems
positives = list(data_dir_covid.glob('*'))
negatives = list(data_dir_nocovid.glob('*'))

predictions_list = []

# 
for i in negatives:
  # read the image
  img = keras.preprocessing.image.load_img(
    str(i), target_size=(300, 300)
  )
  # trasform into array
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  # apply your inference here
  predictions = our_model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  #class_predict = tf.nn.sigmoid_cross_entropy_with_logits(labels=1, logits=np.max(predictions[0]))
  #print(np.max(class_predict))
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )

  #for i in range(32):
  #  print('\n', np.argmax(predictions[i]))

  # save results
