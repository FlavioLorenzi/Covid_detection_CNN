#training cnn locale


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
#now = datetime.now()

#from google.colab import drive
#drive.mount('/content/drive')

#data_dir = "/content/drive/My Drive/dataset_2"
data_dir = "dataset"
#data_dir = pathlib.Path(data_dir)

batch_size = 64
AUTOTUNE = tf.data.experimental.AUTOTUNE

# TROVARE UNA FUNZIONE CHE PIU CHE RITAGLIARE LE IMG, LE RIDUCE DI UN CERTO TOT
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

#plot the first nine samples
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
  plt.show()


#print labels


for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# image_batch è un tensore della forma (32, 180, 180, 3) . Questo è un lotto di 32 immagini di forma 180x180x3 (l'ultima dimensione si riferisce ai canali di colore RGB).
# label_batch è un tensore della forma (32,) , queste sono etichette corrispondenti alle 32 immagini.


#CONFIFURA PRESTAZIONI

#Assicuriamoci di utilizzare il precaricamento bufferizzato in modo da poter produrre dati dal disco senza che l'I / O si blocchi.
#Questi sono due metodi importanti da utilizzare durante il caricamento dei dati.

#.cache() mantiene le immagini in memoria dopo che sono state caricate dal disco durante la prima epoca
#.prefetch() sovrappone alla preelaborazione dei dati e all'esecuzione del modello durante l'addestramento

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

## PICCOLA AGGIUNTA PER EVITARE ERRORI DI DIMENSIONE
#train_ds = train_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
#val_ds = val_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

###############################
#      DATA AUGMENTATION      #
###############################

# Possiamo aggiungere altro

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)), #flip
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
  layers.Conv2D(16, 5, padding='same', activation='relu'),  
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

  layers.Dense(256, activation='relu'),
  
  layers.Dense(2) # 2 is actually our number of classes
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
  loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),	 #set TRUE may be more numerically stable
  #loss= tf.keras.losses.BinaryCrossentropy(),
  metrics=['accuracy'])



#checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = 'check', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')

epochs = 80

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  #callbacks=checkpointer,
  batch_size=batch_size,
  shuffle = 1000
)

model.save('new_model.h5')

###############################################
############### PLOT SECTION ##################
###############################################
#plot accuracy e loss


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

plt.savefig('Results.jpg')
plt.show()




#model.predict_classes(val_ds,batch_size=batch_size)
#model.predict(val_ds)
test_loss, test_acc = model.evaluate(val_ds);
print('\nTest accuracy: ', test_acc)
print('\nTest loss..: ', test_loss)

predictions = model.predict(val_ds)
for i in range(32):
  print('\n', np.argmax(predictions[i]))

#print(np.argmax(predictions[0]))
for image,label in val_ds.take(1):
  print(label)

our_model = tf.keras.models.load_model('new_model.h5')

prediction_list = []
labels_list = []
i = 0

#how_many_times = 
for image_batch, labels_batch in val_ds:
  #print(image_batch[0].shape)
  #print(labels_batch.shape)
  #print(labels_batch[0]) #Torna la label

  for k in range(batch_size):
    img = image_batch[k]
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = our_model.predict(img_array)
    #print(predictions)
    score = tf.nn.softmax(predictions)
    #print(score)
    predict = class_names[np.argmax(score)]
    #print(predict)
    
    prediction_list.append(predict)
    labels_list.append(labels_batch[k])
  i = i+1

print(prediction_list)
print(labels_list)

tf.math.confusion_matrix(
    labels = labels_list,
    predictions = prediction_list
)

#NB invece della seguente sezione, sarebbe piu utile salvare un modello trainato, e fare inference sul test set direttamente ! ! !



#PREDICTION OVER NEW IMAGES


#AGGIUNGERE UN FOR IN CUI ITERA SU OGNI IMMAGINE DEL TEST SET COVID
# sep 14, 3:41

# ----------------------------------------------------------------------

# ITERATE IN THE TEST SET TO DO INFERENCE
dataset_covid = "test/covid"
data_dir_covid = pathlib.Path(dataset_covid)

dataset_nocovid = "test/non-covid"
data_dir_nocovid = pathlib.Path(dataset_nocovid)

# create a list of elems
positives = list(data_dir_covid.glob('*'))
negatives = list(data_dir_nocovid.glob('*'))

predictions_list = []

# 


for i in positives:
  # read the image
  img = keras.preprocessing.image.load_img(str(i), target_size=(300, 300))
  # trasform into array
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  # apply your inference here
  predictions = our_model.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  #class_predict = tf.nn.sigmoid_cross_entropy_with_logits(labels=1, logits=np.max(predictions[0]))
  #print(np.max(class_predict))
  print("dovrebbero uscire tutti 1 o tutti 0:")
  print("")
  print("")
  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
		)

  #for i in range(32):
  #  print('\n', np.argmax(predictions[i]))

  # save results
# ----------------------------------------------------------------------


#RIPETERE LO STESSO PER IL TEST SET NON COVID

# . . .

## Confusion matrix come su pytorch"""

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

# fix class_names
class_names = ['Non-COVID','COVID-2019']

# fix labels (ground truth) / prediction (inference result)
labels = [0,  1, 1, 1,  0, 0, 0, 0,  1, 1]
predictions = [0,  1, 1, 1,  1, 1, 1, 1,  0, 0]

cnf_matrix = confusion_matrix(predictions, labels)
np.set_printoptions(precision=2)   

# Plot normalized confusion matrix
df_cm = pd.DataFrame(cnf_matrix, index = [i for i in class_names],
                                 columns = [i for i in class_names])


ax = sn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True, cbar=False, fmt='g', xticklabels= ['Non-COVID','COVID-2019'], yticklabels= ['Non-COVID','COVID-2019'])
ax.set_title("Confusion matrix")

# save as image
# plt.savefig('./confusion_matrix_'+str(now)+'.png') #dpi = 200


