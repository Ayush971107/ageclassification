from pathlib import Path
import pandas as pd
import numpy as np
import os 
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import gaussian, sobel
from skimage.feature import canny
from sklearn.model_selection import train_test_split


# /teamspace/studios/this_studio/20-50
image_dir = Path("/teamspace/studios/this_studio/20-50")

# print(image_dir)

# filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
# ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(int)
# # Dataframe with all our image and their labels
# raw_df = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)


filepaths = []
ages = []

# Iterate over each subdirectory in the train and test folders
for folder in ['train', 'test']:
    folder_path = image_dir / folder
    for age_folder in folder_path.iterdir():
        if age_folder.is_dir():  # Ensure it's a directory
            # Extract the age group from the folder name
            age_group = age_folder.name
            # Iterate over all .jpg files in the current age group folder
            for img_file in age_folder.glob('*.jpg'):
                # Append file path and age to the lists
                filepaths.append(str(img_file))
                ages.append(int(age_group))

# Create a DataFrame from the collected data
raw_df = pd.DataFrame({'Filepath': filepaths, 'Age': ages})

print("Length: ",len(raw_df))




def age_label(age):
    if 20 <= age <= 25:
        return 0
    elif 26 <= age <= 27:
        return 1
    elif 28 <= age <= 31:
        return 2
    elif 32 <= age <= 36:
        return 3
    elif 37 <= age <= 45:
        return 4
    elif 46 <= age <= 50:
        return 5
    else:
        return None

    # Apply the function to the 'Age' column to create a new 'Label' column
raw_df['Label'] = raw_df['Age'].apply(age_label)

print(raw_df.head(20))
X = raw_df[['Filepath', 'Age']]
y = raw_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save the DataFrames to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)






# Making Canny Data

# def features_grid(img):
#     features = np.array([], dtype='float32')
#     for y in range(0, img.shape[0], 10):
#         for x in range(0, img.shape[1], 10):
#             section_img = img[y:y+10, x:x+10]
#             section_mean = np.mean(section_img)
#             section_std = np.std(section_img)
#             features = np.append(features, [section_mean, section_std])
#     return features

# def extract_canny_edges(filepaths):
#     all_imgs = []
#     for idx, img_path in enumerate(filepaths):
#         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             print(f"Image not found: {img_path}")
#             continue
#         # Resize the image to 200x200 pixels
#         img = cv2.resize(img, (200, 200))
#         img = canny(img, sigma=0.9)
#         img_features = features_grid(img)
#         age = int(os.path.basename(os.path.dirname(img_path)))
#         img_features = np.append(img_features, age)
#         all_imgs.append(img_features)
#     all_imgs = np.array(all_imgs)
#     return all_imgs

# # Extract filepaths and ages for train and test sets
# train_filepaths = X_train['Filepath'].values
# test_filepaths = X_test['Filepath'].values

# # Extract Canny edge features
# train_features = extract_canny_edges(train_filepaths)
# test_features = extract_canny_edges(test_filepaths)

# # Check if features have the expected shape (801)
# assert train_features.shape[1] == 801, f"Expected train features to have shape (n, 801), got {train_features.shape}"
# assert test_features.shape[1] == 801, f"Expected test features to have shape (n, 801), got {test_features.shape}"

# # Save the features to .npy files
# np.save("canny_train.npy", train_features)
# np.save("canny_test.npy", test_features)