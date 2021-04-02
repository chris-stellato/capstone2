import numpy as np
#np.random.seed(1337)  # set for reproducibility

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam #Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import itertools
import os
# import shutil
# import random
# import glob
import matplotlib.pyplot as plt

from src.plotting import plot_confusion_matrix

from tensorflow.keras.preprocessing import image

from src.plotting import show_single


######consolidate imports



def train_model_vgg16(main_path, num_epochs):
    
    train_path = f'{main_path}/images_tvt_split/train'
    valid_path = f'{main_path}/images_tvt_split/valid'
    test_path = f'{main_path}/images_tvt_split/test'

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = train_path, target_size=(224,224), classes=['mask', 'no_mask'], batch_size=10) 
    
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = valid_path, target_size=(224,224), classes=['mask', 'no_mask'], batch_size=10) 
    
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory = test_path, target_size=(224,224), classes=['mask', 'no_mask'], batch_size=10, shuffle=False) 
    
    vgg16_model = tf.keras.applications.vgg16.VGG16()
    
    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    for layer in model.layers[11:]:
        layer.trainable = False
    
    model.add(Dense(units=2, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(x=train_batches, validation_data=valid_batches, epochs=num_epochs, verbose=2)

    return model, test_batches
    #tuple, store both as needed for later


    
    
    
def evaluate_and_display(model, test_batches):
    
    test_imgs, test_labels = next(test_batches)
    showImages(test_imgs, test_labels)
    
    la_list = model.evaluate(test_batches, verbose=True)
    loss = la_list[0]
    accuracy = la_list[1]

    predictions = model.predict(x=test_batches, verbose=0)
    cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
    cm_plot_labels = ['mask', 'no_mask']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
    print (f'\n LOSS: {loss} \n ACCURACY: {accuracy}')
    
    return la_list


def predict_single(model, img_path):
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
    input_prediction = model.predict(img_preprocessed)
    input_prediction = np.round(input_prediction[0])[0]
    
    show_single(img_path, input_prediction)

    if input_prediction == 1:
        return 1
    return 0
    


    
def predict_batch_temp(model, batch_folder):
    
    prediction_list = []
    
    for img_path in os.listdir(batch_folder):
        if '.png'in img_path or '.jpg' in img_path or '.jpeg' in img_path:
            img = image.load_img(f'{batch_folder}/{img_path}', target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
            input_prediction = model.predict(img_preprocessed)
            input_prediction = np.round(input_prediction[0])[0]
            prediction_list.append(input_prediction)
            show_single(f'{batch_folder}/{img_path}', input_prediction)

        
    return prediction_list
    
    
    
    
    
