# Copyright 2023 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import os
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model

from dataset import CUDAPrefetcher, BaseImageDataset, PairedImageDataset
from imgproc import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from test import test
from utils import build_iqa_model, load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, \
    Summary, AverageMeter, ProgressMeter

print(torch.__version__)

def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/SRGAN_X4.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Initialize the image clarity evaluation index
    best_psnr = 0.0
    best_ssim = 0.0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the basic functions needed to start training
    train_data_prefetcher, paired_test_data_prefetcher = load_dataset(config, device)
    espcn_model, fsrcnn_model, vdsr_model, unet_model, ema_model = build_model(config, device)
    pixel_criterion, content_criterion, adversarial_criterion = define_loss(config, device)
    unet_optimizer = define_optimizer(unet_model, config)
    unet_scheduler = define_scheduler(unet_optimizer, config)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_ESPCN_MODEL"]:
        espcn_model = load_pretrained_state_dict(espcn_model,
                                             config["MODEL"]["ESPCN"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_ESPCN_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_ESPCN_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained espcn model weights not found.")
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_FSRCNN_MODEL"]:
        fsrcnn_model = load_pretrained_state_dict(fsrcnn_model,
                                             config["MODEL"]["FSRCNN"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_FSRCNN_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_FSRCNN_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained fsrcnn model weights not found.")
        
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_VDSR_MODEL"]:
        vdsr_model = load_pretrained_state_dict(vdsr_model,
                                             config["MODEL"]["VDSR"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_VDSR_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_VDSR_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained vdsr model weights not found.")

    # Initialize the image clarity evaluation method
    psnr_model, ssim_model, lpips_model = build_iqa_model(
        config["SCALE"],
        config["TEST"]["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))
    
    best_psnr = [0, 0, 0, 0]
    best_ssim = [0, 0, 0, 0]

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(unet_model,
              ema_model,
              [espcn_model, fsrcnn_model, vdsr_model],
              train_data_prefetcher,
              pixel_criterion,
              content_criterion,
              adversarial_criterion,
              unet_optimizer,
              epoch,
              scaler,
              writer,
              device,
              config)

        # Update LR
        unet_scheduler.step()

        for i in range(len(paired_test_data_prefetcher)):
            psnr, ssim, lpips = test(unet_model,
                            [espcn_model, fsrcnn_model, vdsr_model],
                            paired_test_data_prefetcher[i],
                            psnr_model,
                            ssim_model,
                            lpips_model,
                            device,
                            config)
            print("\n")
            
            # Write the evaluation indicators of each round of Epoch to the log
            writer.add_scalar(f"Test/PSNR{i}", psnr, epoch + 1)
            writer.add_scalar(f"Test/SSIM{i}", ssim, epoch + 1)
            writer.add_scalar(f"Test/LPIPS{i}", lpips, epoch + 1)

            # Automatically save model weights
            is_best = psnr > best_psnr[i] and ssim > best_ssim[i]
            is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
            best_psnr[i]=max(psnr, best_psnr[i])
            best_ssim[i]=max(ssim, best_ssim[i])
            save_checkpoint({"epoch": epoch + 1,
                            "psnr": psnr,
                            "ssim": ssim,
                            "state_dict": unet_model.state_dict(),
                            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
                            "optimizer": unet_optimizer.state_dict()},
                            f"epoch_{epoch + 1}.pth.tar",
                            samples_dir,
                            results_dir,
                            f"epoch_{epoch + 1}_testset{i}_best.pth.tar",
                            f"epoch_{epoch + 1}_testset{i}_last.pth.tar",
                            is_best,
                            is_last)

def load_dataset(
        config: Any,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:

    degenerated_train_datasets = BaseImageDataset(
        config["TRAIN"]["DATASET"]["TRAIN_GT_IMAGES_DIR"],
        config["TRAIN"]["DATASET"]["GT_IMAGE_SIZE"],
        config["SCALE"],
    )

    # generate dataset iterator
    degenerated_train_dataloader = DataLoader(degenerated_train_datasets,
                                              batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                              shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                              num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                              pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                              drop_last=True,
                                              persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])

    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(degenerated_train_dataloader, device)
    
    paired_test_data_prefetcher = []

    for i in range(4):
        paired_test_datasets = PairedImageDataset(config["TEST"][f"DATASET{i}"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                                config["TEST"][f"DATASET{i}"]["PAIRED_TEST_LR_IMAGES_DIR"])
        
        paired_test_dataloader = DataLoader(paired_test_datasets,
                                            batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                            shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                            num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                            pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                            drop_last=False,
                                            persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])

        # Replace the data set iterator with CUDA to speed up
        paired_test_data_prefetcher.append(CUDAPrefetcher(paired_test_dataloader, device))

    return train_data_prefetcher, paired_test_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module, nn.Module, nn.Module, nn.Module or Any]:
    espcn_model = model.__dict__["ESPCN"](in_channels = 1, out_channels = 1, channels = 64, upscale_factor = 4)
    fsrcnn_model = model.__dict__["FSRCNN"](upscale_factor=4)
    vdsr_model = model.__dict__["VDSR"]()
    unet_model = model.__dict__["UNet"](config["MODEL"]["UNET"]["IN_CHANNELS"], config["MODEL"]["UNET"]["OUT_CHANNELS"])

    espcn_model = espcn_model.to(device)
    fsrcnn_model = fsrcnn_model.to(device)
    vdsr_model = vdsr_model.to(device)
    unet_model = unet_model.to(device)
    
    espcn_model.eval()
    fsrcnn_model.eval()
    vdsr_model.eval()

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_model = AveragedModel(unet_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_model = None

    # compile model
    if config["MODEL"]["UNET"]["COMPILED"]:
        unet_model = torch.compile(unet_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_model is not None:
        ema_model = torch.compile(ema_model)

    return espcn_model, fsrcnn_model, vdsr_model, unet_model, ema_model


def define_loss(config: Any, device: torch.device) -> [nn.MSELoss, model.ContentLoss, nn.BCEWithLogitsLoss]:
    if config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["NAME"] == "MSELoss":
        pixel_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['PIXEL_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NAME"] == "ContentLoss":
        content_criterion = model.ContentLoss(
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NET_CFG_NAME"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["BATCH_NORM"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["NUM_CLASSES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["MODEL_WEIGHTS_PATH"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NODES"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_MEAN"],
            config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["FEATURE_NORMALIZE_STD"],
        )
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['CONTENT_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["NAME"] == "vanilla":
        adversarial_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['ADVERSARIAL_LOSS']['NAME']} is not implemented.")

    pixel_criterion = pixel_criterion.to(device)
    content_criterion = content_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)

    return pixel_criterion, content_criterion, adversarial_criterion


def define_optimizer(unet: nn.Module, config: Any) -> [optim.Adam]:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        unet_optimizer = optim.Adam(unet.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return unet_optimizer


def define_scheduler(unet_optimizer: optim.Adam, config: Any) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "MultiStepLR":
        unet_scheduler = lr_scheduler.MultiStepLR(unet_optimizer,
                                               config["TRAIN"]["LR_SCHEDULER"]["MILESTONES"],
                                               config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])  
    else:
        unet_scheduler = lr_scheduler.StepLR(unet_optimizer,
                                             step_size=config["TRAIN"]["HYP"]["EPOCHS"]//2,
                                             gamma=config["TRAIN"]["LR_SCHEDULER"]["GAMMA"]) 

    return unet_scheduler


def train(
        unet_model: nn.Module,
        ema_model: nn.Module,
        ensemble_model: list[nn.Module],
        train_data_prefetcher: CUDAPrefetcher,
        pixel_criterion: nn.L1Loss,
        content_criterion: model.ContentLoss,
        adversarial_criterion: nn.BCEWithLogitsLoss,
        unet_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    pixel_losses = AverageMeter("Pixel Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, pixel_losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    espcn_model, fsrcnn_model, vdsr_model = ensemble_model
    
    # Set the model to training mode
    unet_model.train()

    # Define loss function weights
    pixel_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["PIXEL_LOSS"]["WEIGHT"]).to(device)
    feature_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["CONTENT_LOSS"]["WEIGHT"]).to(device)
    adversarial_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["ADVERSARIAL_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    # Used for discriminator binary classification output, the input sample comes from the data set (real sample) is marked as 1, and the input sample comes from the generator (generated sample) is marked as 0
    batch_size = batch_data["gt"].shape[0]
    

    while batch_data is not None:
    # for i in range(101): # it exists for memory capability test
        # Load batches of data
        gt = batch_data["gt"].to(device, non_blocking=True)
        lr = batch_data["lr"].to(device, non_blocking=True)
        lr_up = batch_data["lr_up"].to(device, non_blocking=True)
        
        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # Initialize the generator model gradient
        unet_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        with amp.autocast():
            # sr_1 = espcn_model(lr)
            sr_2 = fsrcnn_model(lr)
            sr_3 = vdsr_model(lr_up)
            
            residual_1 = sr_2 - lr_up[:,0:1,:,:]
            residual_2 = sr_3 - lr_up[:,0:1,:,:]

            # map = torch.cat((sr_1.detach().clone(), sr_2.detach().clone(), sr_3.detach().clone()), 1)
            map = torch.cat((sr_2.detach().clone(), sr_3.detach().clone()), 1)
            
            # hypo = F.softmax(unet(map), dim=1)
            hypo = unet_model(map)
            # sr = hypo[:,0:1,:,:] * sr_1 + hypo[:,1:2,:,:] * sr_2 + hypo[:,2:3,:,:] * sr_3
            # sr = hypo[:,0:1,:,:] * sr_1 + hypo[:,1:2,:,:] * sr_3
            sr = lr_up[:,0:1,:,:] + hypo[:,0:1,:,:] * residual_1 + hypo[:,1:2,:,:] * residual_2

            pixel_loss = torch.sum(torch.mul(pixel_weight, pixel_criterion(torch.clamp(sr, min=0, max=1), gt)))
            # hypo_loss = torch.sum(torch.mul(pixel_weight, pixel_criterion((1 - hypo[:,0:1,:,:]), hypo[:,1:2,:,:])))
            # total_loss = pixel_loss + hypo_loss
            # Compute generator total loss
            # g_loss = pixel_loss + feature_loss + adversarial_loss
        # Backpropagation generator loss on generated samples
        scaler.scale(pixel_loss).backward()
        # update generator model weights
        scaler.step(unet_optimizer)
        scaler.update()
        # end training generator model

        if config["MODEL"]["EMA"]["ENABLE"]:
            # update exponentially averaged model weights
            ema_model.update_parameters(unet_model)

        # record the loss value
        pixel_losses.update(pixel_loss.item(), batch_size)

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
            # write training log
            iters = batch_index + epoch * batches
            writer.add_scalar("Train/Loss", pixel_loss.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1

if __name__ == "__main__":
    main()
