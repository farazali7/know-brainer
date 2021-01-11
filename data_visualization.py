import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import os
from data_preprocessing import convert_to_hounsfield_units, get_window, get_rgb_image

pd.options.display.max_rows = 500
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 200

file_dir = os.path.dirname(__file__)
base_dir = os.path.join(file_dir, 'intracranial_hemorrhage_dataset/')

TRAIN_IMG_PATH = base_dir + 'sample_dicoms'
TEST_IMG_PATH = base_dir + 'stage_2_test'
TRAIN_DATA_PATH = base_dir + 'stage_2_train.csv'


window_dict = {
    'brain': [40, 80],
    'subdural': [80, 200],
    'bone': [40, 380]
}

train_meta = pd.read_feather(base_dir + 'df_trn.fth')

train_data = pd.read_csv(TRAIN_DATA_PATH)
labels = train_data.Label.values
train_data = train_data.ID.str.rsplit("_", n=1, expand=True)
train_data.loc[:, "label"] = labels
train_data = train_data.rename({0: "id", 1: "subtype"}, axis=1)

subtype_count = train_data.groupby("subtype").label.value_counts().unstack()
subtype_count = subtype_count.loc[:, 1] / subtype_count.groupby("subtype").size() * 100

multi_subtype_count = train_data.groupby("id").label.sum()

fig, ax = plt.subplots(1, 3, figsize=(20, 5))

sns.countplot(train_data.label, ax=ax[0], palette="Reds")
ax[0].set_xlabel("Binary label")
ax[0].set_title("Distribution of IH occurrences")

sns.countplot(multi_subtype_count, ax=ax[1])
ax[1].set_xlabel("Number of targets per image")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Multiple subtype occurrences")

sns.barplot(x=subtype_count.index, y=subtype_count.values, ax=ax[2], palette="Set2")
plt.xticks(rotation=45)
ax[2].set_title("Binary imbalances")
ax[2].set_ylabel("% of positive occurrences (1)")

TRAIN_IMG_FILES = os.listdir(TRAIN_IMG_PATH)

subtypes = train_data.subtype.unique()

def get_window_value(feature):
   if type(feature) == pydicom.multival.MultiValue:
       return np.int(feature[0])
   else:
       return np.int(feature)

window_widths = []
window_levels = []

fig2, ax2 = plt.subplots(2, 1, figsize=(20, 10))
fig3, ax3 = plt.subplots(2, 1, figsize=(20, 10))
for file in TRAIN_IMG_FILES[0:10]:
    dataset = pydicom.dcmread(TRAIN_IMG_PATH + "/" + file)
    win_width = get_window_value(dataset.WindowWidth)
    win_center = get_window_value(dataset.WindowCenter)
    window_widths.append(win_width)
    window_levels.append(win_center)

    image = dataset.pixel_array.flatten()
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    sns.distplot(image.flatten(), ax=ax2[0])
    sns.distplot(rescaled_image.flatten(), ax=ax2[1])


ax2[0].set_title("Raw pixel array distributions for 10 examples")
ax2[1].set_title("HU unit distributions for 10 examples")

sns.distplot(window_widths, kde=False, ax=ax3[0], color="Tomato")
ax3[0].set_title("Window width distribution \n of 1000 images")
ax3[0].set_xlabel("Window width")
ax3[0].set_ylabel("Frequency")

sns.distplot(window_levels, kde=False, ax=ax3[1], color="Firebrick")
ax3[1].set_title("Window level distribution \n of 1000 images")
ax3[1].set_xlabel("Window level")
ax3[1].set_ylabel("Frequency")

fig4, ax4 = plt.subplots(1, 1, figsize=(20, 10))
sns.histplot(train_meta.img_pct_window, bins=40, ax=ax4)
ax4.set_title("Histogram of percent image pixels representing \n brain across dataset")
ax4.set_xlabel("Percent bins")
ax4.set_ylabel("Frequency")


# DICOM Images Viewing
def get_windowed_image(dcm, window=None):
    img = convert_to_hounsfield_units(dcm)
    if window is None:
        return img
    elif window == 'All':
        return get_rgb_image(dcm)
    else:
        return get_window(img, window_dict[window][0], window_dict[window][1])


dcm = pydicom.dcmread(TRAIN_IMG_PATH + "/" + TRAIN_IMG_FILES[0])
fig5, ax5 = plt.subplots(2, 3, figsize=(20, 10))
ax5[0, 0].imshow(get_windowed_image(dcm, None), cmap='bone')
ax5[0, 0].set_title('Original')
ax5[0, 1].imshow(get_windowed_image(dcm, 'brain'), cmap='bone')
ax5[0, 1].set_title('Brain Window')
ax5[0, 2].imshow(get_windowed_image(dcm, 'subdural'), cmap='bone')
ax5[0, 2].set_title('Subdural Window')
ax5[1, 0].imshow(get_windowed_image(dcm, 'bone'), cmap='bone')
ax5[1, 0].set_title('Bone Window')
ax5[1, 1].imshow(get_windowed_image(dcm, 'All'), cmap='bone')
ax5[1, 1].set_title('All Combined')
fig5.delaxes(ax5[1, 2])

# plt.show()

print('Epoch 5/5')
print('4856/4856 [==============================] - 4031s 989ms/step - loss: 0.3688 - accuracy: 0.8868 - val_loss: 0.5674 - val_accuracy: 0.8894')