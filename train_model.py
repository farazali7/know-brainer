import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from data_preprocessing import DataGenerator, image_augmentations

SEED = 7
np.random.seed(SEED)

VAL_SIZE = 0.2
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128

EPOCHS = 5
LEARNING_RATE = 0.0001
FINE_TUNING_LEARNING_RATE = 1e-5

HEIGHT = 256
WIDTH = 256
CHANNELS = 3
SHAPE = (WIDTH, HEIGHT, CHANNELS)

BASE_PATH = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'
DCM_IMAGE_PATH = BASE_PATH + 'stage_2_train/'

file_dir = os.path.dirname(__file__)
base_dir = os.path.join(file_dir, 'intracranial_hemorrhage_dataset/')
train_df_path = base_dir + 'final_train_df.csv'
test_df_path = base_dir + 'final_test_df.csv'

train_df = pd.read_csv(train_df_path).set_index('file_name')
test_df = pd.read_csv(test_df_path).set_index('file_name')

# Create training-validation splits
splits = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
file_names = train_df.index
labels = train_df.values
split = next(splits.split(file_names, labels))
train_idx = split[0]
valid_idx = split[1]

# Instantiate data generators
train_data_gen = DataGenerator(DCM_IMAGE_PATH, train_df.iloc[train_idx],
                               train_df.iloc[train_idx],
                               image_augmentations(),
                               TRAIN_BATCH_SIZE,
                               (WIDTH, HEIGHT), augment=True)

val_data_gen = DataGenerator(DCM_IMAGE_PATH, train_df.iloc[valid_idx],
                             train_df.iloc[valid_idx],
                             image_augmentations(),
                             VALID_BATCH_SIZE,
                             (WIDTH, HEIGHT), augment=False)

# Build the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=SHAPE, pooling='avg')
x = base_model.output
x = Dropout(0.125)(x)
output_layer = Dense(6, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss=CategoricalCrossentropy(),
              metrics=['acc', keras.metrics.AUC()])
model.summary()

model_file_path = 'know_brainer_model.h5'
checkpoint = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Training the model
print('TRAINING THE MODEL')
model.fit_generator(generator=train_data_gen,
                    validation_data=val_data_gen,
                    epochs=1,
                    callbacks=callbacks_list,
                    verbose=1)

# Fine-tuning (if necessary)
print('FINE-TUNING THE MODEL')
for layer in model.layers[:-1]:
    layer.trainable = True

model.load_weights('know_brainer_model.h5')
model.compile(optimizer=Adam(learning_rate=FINE_TUNING_LEARNING_RATE),
              loss=CategoricalCrossentropy(),
              metrics=['acc', keras.metrics.AUC()])

model.fit_generator(generator=train_data_gen,
                    validation_data=val_data_gen,
                    epochs=EPOCHS,
                    steps_per_epoch=len(train_data_gen)/6,
                    callbacks=callbacks_list,
                    verbose=1)
