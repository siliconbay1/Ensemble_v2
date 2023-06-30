# Ensemble_v2

## Environment
```text
Here, python pytorch version are found
install_pytorch.bat


pip install -r requirements.txt
```

## Download dataset
- [Google Driver](https://drive.google.com/file/d/1f2rrFlF9OEXIqPTQnyyGxctFYwBUmzkm/view?usp=share_link)

## Generate train data
```text
image files should exist as strucutured

├─data
     ├─ImageNet
       ├─original
          ├─1.jpg
          ├─2.jpg          


go to scripts folder and use run.py
~#python run_128.py

os.system("python ./prepare_dataset.py --images_dir ../data/ImageNet/original --output_dir ../data/ImageNet/train --gt_image_size 96 --step 64 --num_workers 16")

gt_image_size above should be greater than gt_image_size in config.py 

train_gt_images_dir in config.py stands for train data directory

also change the path for test
for Set5, search google and download it
test_gt_images_dir = f"../Ensemble-PyTorch-master/data/Set5/GTmod12"
test_lr_images_dir = f"../Ensemble-PyTorch-master/data/Set5/LRbicx{upscale_factor}"

```

## Train neural network
```text
~#python train.py

see config.py to configure batch size and ground true image size etc
```
