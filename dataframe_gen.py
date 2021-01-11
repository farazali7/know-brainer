import pandas as pd
import os

file_dir = os.path.dirname(__file__)
base_dir = os.path.join(file_dir, 'intracranial_hemorrhage_dataset/')
full_data_path = base_dir + 'stage_2_train.csv'
initial_train_path = base_dir + 'stage_1_train.csv'
initial_test_path = base_dir + 'stage_1_sample_submission.csv'

# Generate full dataset dataframe
full_dataset_df = pd.read_csv(full_data_path)
full_dataset_df['file_name'] = full_dataset_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
full_dataset_df['subtype'] = full_dataset_df['ID'].apply(lambda x: x.split('_')[-1])
full_dataset_df.drop_duplicates(['Label', 'file_name', 'subtype'], inplace=True)
full_dataset_df = pd.pivot_table(full_dataset_df.drop(columns='ID'), index='file_name', columns='subtype', values='Label')
full_dataset_df.drop('ID_6431af929.dcm', inplace=True)  # Corrupted pixel data

# Split out train and test dataframes
for set in [initial_train_path, initial_test_path]:
    initial_df = pd.read_csv(set)
    initial_df['file_name'] = initial_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
    initial_df.drop_duplicates(['ID', 'Label'], inplace=True)
    type = 'train' if 'train' in set else 'test'
    final_df = full_dataset_df[full_dataset_df.index.isin(initial_df['file_name'])]
    if 'train' in set:
        type = 'train'
        epidural_df = final_df[final_df.epidural == 1]
        final_df = pd.concat([final_df, epidural_df])
    else:
        type = 'test'

    file_name = 'final_' + type + '_df.csv'
    final_df.to_csv(file_name)

