import numpy as np
import pandas as pd

from PIL import Image

from sklearn import train_test_split

from keras import applications, optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model, save_model
from keras.preprocessing.image import ImageDataGenerator

from settings import *

# build the VGG16 network
model = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width, img_height, 3),
)

# set the first 10 layers to non-trainable
for layer in model.layers[:10]:
    layer.trainable = False

# build custom model
reg = Flatten()(model.output)
reg = Dense(1024, activation='relu')(reg)
reg = Dropout(0.5)(reg)
reg = Dense(3, activation='linear')(reg)

# combine into tuned model
tuned_model = Model(input=model.input, output=reg)
tuned_model.compile(
    loss='mse',
    metrics=['mse'],
    optimizer=optimizers.Adadelta(lr=0.1),
)

# prepare train data augmentation configuration
rescale = 1. / 255
test_datagen = ImageDataGenerator(rescale=rescale)
train_datagen = ImageDataGenerator(
    rescale=rescale,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
)

# read training y
labels = pd.read_csv('labels.csv').set_index('id')
y = labels.values

# read training X
train_palm_fname = [
    'data/ %d.jpg' % fname for fname in labels.index.tolist()
]
X = np.zeros((len(train_palm_fname), img_width, img_height, 3))
for idx, fname in enumerate(train_palm_fname):
    X[idx, :, :, :] = np.array(
        Image.open(fname).resize((img_width, img_height)).getdata(),
        np.uint8,
    ).reshape(img_width, img_height, 3)

# make train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# make flow
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# fine-tune
tuned_model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    samples_per_epoch=nb_train_samples,
    nb_val_samples=nb_validation_samples,
)

save_model(tuned_model, 'palm_model.mdl')
