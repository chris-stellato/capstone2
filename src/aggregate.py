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

from src.plotting import *
from src.model import *
from src.organize_images import *
from src.pipeline import *
from src.plotting import *

## consolidate imports

def train_evaluate_vgg16(main_file_path, num_epochs):
    model, test_batches = train_model_vgg16(main_file_path, num_epochs)
    evaluate_and_display(model, test_batches)