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
import joblib

SEED = 7
np.random.seed(SEED)

VAL_SIZE = 0.2
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 128

LEARNING_RATE = 0.0001

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
model_weights = base_dir + 'know_brainer_model.h5'

train_df = pd.read_csv(train_df_path).set_index('file_name')
test_df = pd.read_csv(test_df_path).set_index('file_name')


def get_scores(data_gen, pred_model, file_name='scores.pkl'):
    scores = pred_model.evaluate_generator(data_gen, verbose=1)
    joblib.dump(scores, file_name)
    print(f"Accuracy: {scores[1]*100} and Loss: {scores[0]}")


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

test_data_gen = DataGenerator(DCM_IMAGE_PATH, test_df, test_df,
                              image_augmentations(),
                              1,
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
model.load_weights(model_weights)

get_scores(train_data_gen, model, file_name='train_scores.pkl')
get_scores(val_data_gen, model, file_name='val_scores.pkl')
get_scores(test_data_gen, model, file_name='test_scores.pkl')