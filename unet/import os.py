import os
from PIL import Image
import numpy as np

img_dir = 'segmentation_full_body_mads_dataset_1192_img/images'
mask_dir = 'segmentation_full_body_mads_dataset_1192_img/masks'
output_img_dir = 'segmentation_full_body_mads_dataset_1192_img/images_preprocessed'
output_mask_dir = 'segmentation_full_body_mads_dataset_1192_img/masks_preprocessed'

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

target_size = (256, 256)  # Resize to this size

for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)  # Assuming masks have the same name as images

    if os.path.exists(mask_path):
        # Resize image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img.save(os.path.join(output_img_dir, img_file))

        # Resize mask
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        mask = mask.resize(target_size)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.uint8)  # Binary mask (0 or 1)
        mask = Image.fromarray(mask * 255)  # Convert back to image format
        mask.save(os.path.join(output_mask_dir, img_file))