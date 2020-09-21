# -*- coding: utf-8 -*-
"""train2_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1__dKCTcMLVEkM_SGQeB1N1iV_g--ryyk
"""

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
from datetime import datetime
now = datetime.now()

from google.colab import drive
drive.mount('/content/drive')

data_dir = "/content/drive/My Drive/vision/dataset"
os.chdir(data_dir)
#data_dir = pathlib.Path(data_dir)

batch_size = 64

AUTOTUNE = tf.data.experimental.AUTOTUNE


img_height = 256
img_width = 256
print(data_dir)

# Defaults image size to (256, 256)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = ['covid','non-covid']
print(class_names)

#CONFIFURA PRESTAZIONI

#Assicuriamoci di utilizzare il precaricamento bufferizzato in modo da poter produrre dati dal disco senza che l'I / O si blocchi.
#Questi sono due metodi importanti da utilizzare durante il caricamento dei dati.

#.cache() mantiene le immagini in memoria dopo che sono state caricate dal disco durante la prima epoca
#.prefetch() sovrappone alla preelaborazione dei dati e all'esecuzione del modello durante l'addestramento

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

###############################
#      DATA AUGMENTATION      #
###############################

# Possiamo aggiungere altro

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
    #layers.experimental.preprocessing.CenterCrop(25,25),
    #layers.experimental.preprocessing.RandomContrast(samples, height, width, channels) 
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomTranslation(0.1,0.1),
    layers.experimental.preprocessing.RandomZoom(0.2),
    #layers.GaussianNoise(stddev=1)   #standard deviation of noise distribution
  ]
)

###############################
#          CNN MODEL          #
###############################
#there are no real theoretical rules for which to set the number of filters: 
#they should be tuned like the hyperparameters;
#once you found a good enough functioning, keep them.

model = tf.keras.Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),

  layers.Conv2D(16, 5, padding='valid', activation='relu'),  
  layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(momentum=0.8),
  
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(momentum=0.8),

  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(momentum=0.8),

  layers.Conv2D(128, 3, padding='same', activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.BatchNormalization(momentum=0.8),

  layers.Dropout(0.5),
  layers.Flatten(),

  #change activation
  layers.Dense(256, activation='relu'),  #first fully connected layers
  
  layers.Dense(2) # 2 is actually our number of classes. #last fully con layer
  
  #nb: If BN is applied to this last layer softmax will produce slightly different values and thus predictions have low accuracy


])

#LR DECAY with STOCASTIC GRAD DESCENT
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
opt = keras.optimizers.SGD(learning_rate=lr_schedule)


#opt = optimizers.Adam(learning_rate=0.0001) #da provare poi cosi

model.compile(
  optimizer=opt,
  loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),	 #set TRUE may be more numerically stable.
  #loss= tf.keras.losses.BinaryCrossentropy(),
  metrics=['accuracy'])

#checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = 'check', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
print(model.summary())

print(len(list(dataset)))

epochs = 100

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=checkpointer,
  batch_size=batch_size,
  shuffle = 1040
)

model.save('model.h5')



###############################
#       PLOT SECTION          #
###############################
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
#plt.savefig()