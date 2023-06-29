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
# ============================================================================
import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import config
import model
import vgg as vgg19
from unet import unet_model

from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, save_checkpoint, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    # ----------------------------------
    # prepration for dataset
    # ----------------------------------
    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")
    
    # ----------------------------------
    # pre-trained model for FSRCNN ESPCN
    # ----------------------------------
    fsrcnn, espcn = build_pretrained_model()
    print(f"Build fsrcnn espcn model successfully.")
    
    print("Check whether to load pretrained model weights...")
    if config.pretrained_espcn_weights_path:
        espcn_model = load_state_dict(espcn, config.pretrained_espcn_weights_path, load_mode="pretrained")
        espcn_model.eval()
        print(f"Loaded `{config.pretrained_espcn_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")
        
    print("Check whether to load pretrained model weights...")
    if config.pretrained_fsrcnn_weights_path:
        fsrcnn_model = load_state_dict(fsrcnn, config.pretrained_fsrcnn_weights_path, load_mode="pretrained")
        fsrcnn_model.eval()
        print(f"Loaded `{config.pretrained_fsrcnn_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")
    
    # --------------------------------
    # build generator and dicriminator
    # --------------------------------
    unet = build_model()
    print(f"Build `{config.unet_arch_name}` model successfully.")

    # ---------------------------
    # definition of loss function
    # ---------------------------
    pixel_criterion, content_criterion, gradient_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")
    
    # -----------------------
    # definition of optimizer
    # -----------------------
    unet_optimizer = optim.Adam(unet.parameters(),
                             config.model_lr,
                             config.model_betas,
                             config.model_eps,
                             config.model_weight_decay)
    
    unet_scheduler = lr_scheduler.StepLR(unet_optimizer,
                                      config.lr_scheduler_step_size,
                                      config.lr_scheduler_gamma)    
    # --------------------------------------
    # load pretrained models to resume train
    # --------------------------------------

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device)
    ssim_model = ssim_model.to(device=config.device)

    for epoch in range(start_epoch, config.epochs):
        # train(unet,
        #       espcn_model,
        #       fsrcnn_model,
        #       train_prefetcher,
        #       pixel_criterion,
        #       unet_optimizer,
        #       epoch,
        #       scaler,
        #       writer)

        psnr, ssim = validate(unet,
                              espcn_model,
                              fsrcnn_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              "Test")
        print("\n")

        # Update LR
        unet_scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": unet.state_dict(),
                         "optimizer": unet_optimizer.state_dict(),
                         "scheduler": unet_scheduler.state_dict()},
                        f"unet_epoch_{epoch + 1}.pth.tar",
                        config.samples_dir,
                        config.results_dir,
                        "unet_best.pth.tar",
                        "unet_last.pth.tar",
                        is_best,
                        is_last,
                        epoch + 1)

def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_gt_images_dir,
                                            config.gt_image_size,
                                            config.upscale_factor,
                                            "Train")
    test_datasets = TestImageDataset(config.test_gt_images_dir, config.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher

def build_model() -> [nn.Module]:
    unet = unet_model.__dict__["UNet"](config.in_channels, config.out_channels)
    unet = unet.to(device=config.device)
    return unet

def build_pretrained_model() -> [nn.Module, nn.Module]:
    fsrcnn = model.__dict__["FSRCNN"](upscale_factor=4)
    espcn = model.__dict__["ESPCN"](in_channels = 1, out_channels = 1, channels = 64, upscale_factor = 4)

    fsrcnn = fsrcnn.to(device=config.device)
    espcn = espcn.to(device=config.device)

    return fsrcnn, espcn

def define_loss() -> [nn.MSELoss, model.content_loss, model.gradient_loss, nn.BCEWithLogitsLoss]:
    pixel_criterion = nn.MSELoss()
    content_criterion = model.content_loss(feature_model_extractor_node=config.feature_model_extractor_node,
                                           feature_model_normalize_mean=config.feature_model_normalize_mean,
                                           feature_model_normalize_std=config.feature_model_normalize_std)
    gradient_criterion = model.gradient_loss(config.lambda_gp, device=config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=config.device)
    content_criterion = content_criterion.to(device=config.device)
    gradient_criterion = gradient_criterion.to(device=config.device)
    adversarial_criterion = adversarial_criterion.to(device=config.device)

    return pixel_criterion, content_criterion, gradient_criterion, adversarial_criterion,

def train(
        unet,
        espcn_model,
        fsrcnn_model,
        train_prefetcher: CUDAPrefetcher,
        pixel_criterion,
        unet_optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, 
                                       pixel_losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    unet.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
    # for i in range(101): # it exists for memory capability test
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)
        
        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=config.device, non_blocking=True)
        
        unet.zero_grad(set_to_none=True)
        
        sr_1 = espcn_model(lr.detach().clone())
        sr_2 = fsrcnn_model(lr.detach().clone())
        
        map = torch.cat((sr_1, sr_2), 1)
        # hypo = F.softmax(unet(map), dim=1)
        hypo = unet(map)
        
        sr = sr_1 * hypo + sr_2 * (1 - hypo)

        loss = torch.mul(config.pixel_weight, pixel_criterion(sr, gt))
        loss.backward()
        unet_optimizer.step()

        pixel_losses.update(loss.item(), sr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:
            # Record loss during training and output to file
            iters = batch_index + epoch * batches + 1

            writer.add_scalar("Train/Loss", loss.item(), iters)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        unet: nn.Module,
        espcn_model: nn.Module,
        fsrcnn_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    unet.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr_1 = espcn_model(lr.detach().clone())
                sr_2 = fsrcnn_model(lr.detach().clone())
                
                map = torch.cat((sr_1, sr_2), 1)
                # hypo = F.softmax(unet(map), dim=1)
                hypo = unet(map)
                
                sr = sr_1 * hypo + sr_2 * (1 - hypo)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.test_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
