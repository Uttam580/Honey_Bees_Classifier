import pandas as pd
import numpy as np
import sys
import os
import random 
from pathlib import Path

import imageio
import skimage
import skimage.io
import skimage.transform

# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics


#tensorflow
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy


#setting random seed
from numpy.random import seed
seed(101)
tensorflow.random.set_seed(101)
data_path = os.path.join('c:' + os.sep, 'Users', 'uttam', 'Desktop', 'DS', 'honey_bees_classifier','Dataset')
print(data_path)

#setting Global varible
# Global variables
img_folder=os.path.join(data_path+os.sep+'bee_imgs\\')
img_width=100
img_height=100
img_channels=3

df=pd.read_csv(os.path.join(data_path+os.sep+'bee_data.csv'), 
                index_col=False,  
                parse_dates={'datetime':[1,2]},
                dtype={'subspecies':'category', 'health':'category','caste':'category'})

#print(df.head)

def read_img(file):
    img = skimage.io.imread(img_folder + file)
    img = skimage.transform.resize(img, (img_width, img_height), mode='reflect')
    return img[:,:,:img_channels]

def split_balance(df, field_name):
    """ 
    Split to train, test and validation. 
    Then balance train by given field name.
    Draw plots before and after balancing
    
    @param df: Total Bees dataset to balance and split
    @param field_name: Field to balance by
    @return:  balanced train bees, validation bees, test bees
    """
    # Split to train and test before balancing
    train_bees, test_bees = train_test_split(df, random_state=24)

    # Split train to train and validation datasets
    # Validation for use during learning
    train_bees, val_bees = train_test_split(train_bees, test_size=0.1, random_state=24)

    #Balance by subspecies to train_bees_bal_ss dataset
    # Number of samples in each category
    ncat_bal = int(len(train_bees)/train_bees[field_name].cat.categories.size)
    train_bees_bal = train_bees.groupby(field_name, as_index=False).apply(lambda g:  g.sample(ncat_bal, replace=True)).reset_index(drop=True)
    return(train_bees_bal, val_bees, test_bees)

#loading images and one hot encoding 
def prepare2train(train_bees, val_bees, test_bees, field_name):
    """
    Load images for features, drop other columns
    One hot encode for label, drop other columns
    @return: image generator, train images, validation images, test images, train labels, validation labels, test labels
    """
    # Bees already splitted to train, validation and test
    # Load and transform images to have equal width/height/channels. 
    # read_img function is defined in the beginning to use in both health and subspecies. 
    # Use np.stack to get NumPy array for CNN input

    # Train data
    train_X = np.stack(train_bees['file'].apply(read_img))
    #train_y = to_categorical(train_bees[field_name].values)
    train_y  = pd.get_dummies(train_bees[field_name], drop_first=False)

    # Validation during training data to calc val_loss metric
    val_X = np.stack(val_bees['file'].apply(read_img))
    #val_y = to_categorical(val_bees[field_name].values)
    val_y = pd.get_dummies(val_bees[field_name], drop_first=False)

    # Test data
    test_X = np.stack(test_bees['file'].apply(read_img))
    #test_y = to_categorical(test_bees[field_name].values)
    test_y = pd.get_dummies(test_bees[field_name], drop_first=False)

    # Data augmentation - a little bit rotate, zoom and shift input images.
    generator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)
    generator.fit(train_X)
    return (generator, train_X, val_X, test_X, train_y, val_y, test_y)

print('spliiting data for Bee Subspecies classification')
 #spliiting data for Bee Subspecies classification   
train_bees_bal, val_bees, test_bees=split_balance(df, 'subspecies')
# Will use balanced dataset as main
train_bees = train_bees_bal
# Call image preparation and one hot encoding
generator, train_X, val_X, test_X, train_y, val_y, test_y = prepare2train(train_bees, val_bees, test_bees, 'subspecies')
print('image preprocessing has done')

# We'll stop training if no improvement after some epochs
earlystopper1 = EarlyStopping(monitor='loss', patience=10, verbose=1)
# Build CNN model
model1=Sequential()
model1.add(Conv2D(6, kernel_size=3, input_shape=(img_width, img_height,3), activation='relu', padding='same'))
model1.add(MaxPool2D(2))
model1.add(Conv2D(12, kernel_size=3, activation='relu', padding='same'))
model1.add(Flatten())
model1.add(Dense(train_y.columns.size, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('model has compiled')
model1.summary()

print('model trainning has started')
# Train
history = model1.fit_generator(generator.flow(train_X,train_y, batch_size=32)
                        ,epochs=30
                        ,validation_data=[val_X, val_y]
                        ,steps_per_epoch=50
                        ,callbacks=[earlystopper1])

    
train_acc= history.history['accuracy']
val_acc= history.history['val_accuracy']
train_loss= history.history['loss']
val_loss= history.history['val_loss']


print(f'training accuracy : {train_acc}')

print(f'val_accuracy : {val_acc}')

print(f'training loss : {train_loss}')

print(f'val loss : {val_loss}')

print("Classification report")
# Accuracy by subspecies
test_pred = model1.predict(test_X)
test_pred = np.argmax(test_pred, axis=1)
test_truth = np.argmax(test_y.values, axis=1)
print(metrics.classification_report(test_truth, test_pred, target_names=test_y.columns))

# saving the model
MODEL_NAME = 'Subspecies_classifier.h5'
model1.save(f'./models/{MODEL_NAME}')
print(f'{MODEL_NAME} saved successfully')
