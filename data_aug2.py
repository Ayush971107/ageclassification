import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from scipy import ndimage
import os

# Define the directory for the augmented images
augmented_images_dir = Path("/teamspace/studios/this_studio/train_aug")
augmented_images_dir.mkdir(parents=True, exist_ok=True)

# Read the CSV file into a DataFrame
train_df = pd.read_csv('train.csv')

# Initialize an empty DataFrame to hold augmented image data
train_aug_df = pd.DataFrame(columns=train_df.columns)

# List to keep track of corrupted or unreadable files
corrupted_files = []

# Iterate over all images in the DataFrame
for i in range(train_df.shape[0]):

    # Extract information from the DataFrame
    img_path = train_df.loc[i, 'Filepath']
    img_age = train_df.loc[i, 'Age']
    img_label = train_df.loc[i, 'Label']

    # Read the image using OpenCV
    img = cv2.imread(img_path)

    # Check if the image was read correctly
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        corrupted_files.append(img_path)
        continue  # Skip to the next image if this one is corrupted

    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply data augmentation transformations
    try:
        img_rot_pos40 = ndimage.rotate(img, 40, reshape=False)
        img_rot_pos20 = ndimage.rotate(img, 20, reshape=False)
        img_rot_neg20 = ndimage.rotate(img, -20, reshape=False)
        img_rot_neg40 = ndimage.rotate(img, -40, reshape=False)
        img_fliplr = np.fliplr(img)
        img_fliplr_rot_pos40 = ndimage.rotate(img_fliplr, 40, reshape=False)
        img_fliplr_rot_pos20 = ndimage.rotate(img_fliplr, 20, reshape=False)
        img_fliplr_rot_neg20 = ndimage.rotate(img_fliplr, -20, reshape=False)
        img_fliplr_rot_neg40 = ndimage.rotate(img_fliplr, -40, reshape=False)
    except Exception as e:
        print(f"Error during augmentation of image {img_path}: {str(e)}")
        corrupted_files.append(img_path)
        continue

    # Get the base filename without the extension
    img_name_wo_ext = Path(img_path).stem

    # Save the original and augmented images to the new directory
    original_augmented_images = [
        (img, f"{img_name_wo_ext}.jpg"),
        (img_rot_pos40, f"{img_name_wo_ext}_rot_pos40.jpg"),
        (img_rot_pos20, f"{img_name_wo_ext}_rot_pos20.jpg"),
        (img_rot_neg20, f"{img_name_wo_ext}_rot_neg20.jpg"),
        (img_rot_neg40, f"{img_name_wo_ext}_rot_neg40.jpg"),
        (img_fliplr, f"{img_name_wo_ext}_fliplr.jpg"),
        (img_fliplr_rot_pos40, f"{img_name_wo_ext}_fliplr_rot_pos40.jpg"),
        (img_fliplr_rot_pos20, f"{img_name_wo_ext}_fliplr_rot_pos20.jpg"),
        (img_fliplr_rot_neg20, f"{img_name_wo_ext}_fliplr_rot_neg20.jpg"),
        (img_fliplr_rot_neg40, f"{img_name_wo_ext}_fliplr_rot_neg40.jpg"),
    ]

    # Create new entries for the augmented images
    for aug_img, aug_img_name in original_augmented_images:
        aug_img_path = augmented_images_dir / aug_img_name
        cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        # Append the augmented image details to the DataFrame
        temp_df = pd.DataFrame({
            'Filepath': [str(aug_img_path)],
            'Age': [img_age],
            'Label': [img_label]
        })
        train_aug_df = pd.concat([train_aug_df, temp_df], axis=0, ignore_index=True)

    # Print progress for every 500 images processed
    if (i + 1) % 500 == 0:
        print(f"Images augmented: {i + 1} of {train_df.shape[0]}")

print("\nDone augmenting all training dataset images and saved them into train_augmented.")

# Save the augmented DataFrame to a CSV file
train_aug_df.to_csv('train_aug.csv', index=False)

# Log corrupted or unreadable files
if corrupted_files:
    print("\nThe following files were unreadable or corrupted:")
    for file in corrupted_files:
        print(file)
