import os
import glob

import numpy as np
import pandas as pd

from PIL import Image

from keras import applications, optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model, save_model
from keras.preprocessing.image import ImageDataGenerator

# constant
epochs = 1
batch_size = 16
nb_train_samples = 2000
nb_validation_samples = 800
img_width, img_height = 512, 512

# read labels
labels = pd.read_csv('labels.csv').set_index('id')
all_palm_img_fname = [
    'data/ %d.jpg' % fname for fname in labels.index.tolist()
]
y = labels.values
data = np.zeros((len(all_palm_img_fname), img_width, img_height, 3))

for idx, fname in enumerate(all_palm_img_fname):
    data[idx, :, :, :] = np.array(
        Image.open(fname).resize((img_width, img_height)).getdata(),
        np.uint8,
    ).reshape(img_width, img_height, 3)

# build the VGG16 network
model = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_width, img_height, 3),
)
# import ipdb; ipdb.set_trace()
print('Model loaded.')

reg = Flatten()(model.output)
reg = Dense(1024, activation='relu')(reg)
reg = Dropout(0.5)(reg)
reg = Dense(3, activation='linear')(reg)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:10]:
    layer.trainable = False

tuned_model = Model(input=model.input, output=reg)
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
tuned_model.compile(
    loss='mse',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['mse'],
)

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')
train_generator = train_datagen.flow(data, y)

# validation_generator = test_datagen.flow(validate_data, validate_y)

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# fine-tune the tuned_model
tuned_model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    # validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
)

save_model(tuned_model, 'palm_model.mdl')
