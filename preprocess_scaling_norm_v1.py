import numpy as np
import pandas as pd
import os
import tensorflow as tf
import pydicom
from data_preprocessing import convert_to_hounsfield_units


file_dir = os.path.dirname(__file__)
base_dir = os.path.join(file_dir, 'intracranial_hemorrhage_dataset/')

train_df_path = base_dir + 'final_train_df.csv'
test_df_path = base_dir + 'final_test_df.csv'

sample_dcm_path = base_dir + 'sample_dicoms/'
sample_dcms = [sample_dcm_path + fname for fname in os.listdir(sample_dcm_path)]

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

# --- DIFFERENT SCALING + NORMALIZATION APPROACH (UNUSED) --- #
# Histogram Rescaling
# Create and save dataframes for pixel sampling
def generate_pix_samples(metadata, is_train):
    new_df = metadata[['SOPInstanceUID', 'BitsStored', 'PixelRepresentation']]
    new_df['file_name'] = new_df['SOPInstanceUID'] + '.dcm'
    new_df.drop(columns=['SOPInstanceUID'], inplace=True)
    new_df.set_index('file_name', inplace=True)
    if is_train:
        new_df.drop('ID_6431af929.dcm', inplace=True)  # Corrupted file
        df = train_df[:]
    else:
        df = test_df[:]
    combined = new_df.join(df.set_index('file_name'), 'file_name')
    combined.drop_duplicates(inplace=True)
    combined['Row_Total'] = combined.sum(numeric_only=True, axis=1) \
                            - (combined['BitsStored'] + combined['PixelRepresentation'])
    combined = combined[combined['Row_Total'] < 3]
    type = 'train' if is_train else 'test'
    file_name = type + '_samples_df.csv'
    combined.to_csv(file_name)


# Get samples of each subtype to determine bins for normalization
def get_hist_bins_tensor(input_path, pix_samples):
    combined_file_paths = [input_path + dcm for dcm in pix_samples['file_name']]
    return tf.stack([convert_to_hounsfield_units(pydicom.dcmread(dcm)) for dcm in combined_file_paths])


# Split range of pixel values into groups with equivalent number of pixels in each group
def get_freq_hist_bins(samples_tensor, n_bins=100):
    imsd = np.sort(samples_tensor.numpy().flatten())
    t = np.array([0.001])
    t = np.append(t, np.arange(n_bins)/n_bins+(1/2/n_bins))
    t = np.append(t, 0.999)
    t = (len(imsd)*t+0.5).astype(np.int)
    return np.unique(imsd[t])


# Use bins to normalize a DICOM image with uniform pixel distribution
def get_hist_scaled(dcm, bins):
    orig_shape = dcm.shape
    ys = np.linspace(0., 1., len(bins))
    x = dcm.flatten()
    x = np.interp(x, bins, ys)
    return tf.clip_by_value(tf.reshape(tf.convert_to_tensor(x), orig_shape), 0., 1.)
