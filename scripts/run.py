import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/train --gt_image_size 128 --step 64 --num_workers 16")

# os.system("python ./prepare_dataset.py --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/ESPCN/train --gt_image_size 70 --step 35 --num_workers 16")

# Split train and valid
# os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/ImageNet/train --valid_images_dir ../data/ImageNet/valid --valid_samples_ratio 0.1")
