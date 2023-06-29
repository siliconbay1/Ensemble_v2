# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = False
# Model architecture name
d_arch_name = "_Discriminator"
g_arch_name = "SRResNet"
unet_arch_name = "UNet"
# Model arch config
in_channels = 2
out_channels = 1
# channels = 64
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "+new_-feature-softmax"
samples_dir = f"samples/{exp_name}"
results_dir = f"results/{exp_name}"

if mode == "train":
    # Dataset address
    # train_gt_images_dir = f"../SRGAN-PyTorch-main/data/ImageNet/train"
    train_gt_images_dir = f"../Ensemble-PyTorch-master/data/ImageNet/train"
    # train_gt_images_dir = f"../SRGAN-PyTorch-main/data/ImageNet/SRGAN/train"

    test_gt_images_dir = f"../Ensemble-PyTorch-master/data/Set5/GTmod12"
    test_lr_images_dir = f"../Ensemble-PyTorch-master/data/Set5/LRbicx{upscale_factor}"

    # gt_image_size = int(17 * upscale_factor)
    gt_image_size = int(24 * upscale_factor)
    # gt_image_size = int(12 * upscale_factor)
    # gt_image_size = int(6 * upscale_factor)

    batch_size = 16
    num_workers = 4

    #load the pretrained model to resume training
    resume_d_model_weights_path = f"" #./results/ensemble_x4/d_best.pth.tar"
    resume_g_model_weights_path = f"" #./results/ensemble_x4/g_best.pth.tar"
    
    pretrained_espcn_weights_path = f"./pretrained/ESPCN_x4-T91-64bf5ee4.pth.tar"
    pretrained_fsrcnn_weights_path = f"./pretrained/fsrcnn_x4-T91-97a30bfb.pth.tar"

    # Total num epochs
    epochs = 3000

    # loss function weights
    loss_weights = 1.0
    
    # Loss function weight
    pixel_weight = 1.0
    content_weight = 1.0
    adversarial_weight = 0.001
    
    lambda_gp = 10
    
    # Feature extraction layer parameter configuration
    feature_model_extractor_node = "features.35"
    feature_model_normalize_mean = [0.485, 0.456, 0.406]
    feature_model_normalize_std = [0.229, 0.224, 0.225]

    # # Optimizer parameter
    # model_lr = 1e-2
    # model_momentum = 0.9
    # model_weight_decay = 1e-4
    # model_nesterov = False
    
    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # # Dynamically adjust the learning rate policy [100,000 | 200,000]
    # lr_scheduler_step_size = epochs // 2
    # lr_scheduler_gamma = 0.1
    
    # Dynamically adjust the learning rate policy [100,000 | 200,000]
    lr_scheduler_step_size = epochs // 2
    lr_scheduler_gamma = 0.1


    # EMA parameter
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.8)]
    lr_scheduler_gamma = 0.1

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = "./data/Set5/GTmod12"

    model_weights_path = "./results/pretrained_models/ESPCN_x4-T91-64bf5ee4.pth.tar"
