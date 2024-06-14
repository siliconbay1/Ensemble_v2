# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import time
from typing import Any

import cv2
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, PairedImageDataset
from imgproc import tensor_to_image
from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary

def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_test_data_prefetcher


def build_model(config: Any, device: torch.device) -> nn.Module or Any:
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    g_model = g_model.to(device)

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)

    return g_model


def test(
        unet_model: nn.Module,
        ensemble_model: list[nn.Module],
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        lpips_model: nn.Module,
        device: torch.device,
        config: Any,
) -> [float, float]:
    if config["TEST"]["SAVE_IMAGE"] and config["TEST"]["SAVE_DIR_PATH"] is None:
        raise ValueError("Image save location cannot be empty!")

    if config["TEST"]["SAVE_IMAGE"]:
        save_dir_path = os.path.join(config["SAVE_DIR_PATH"], config["EXP_NAME"])
        make_directory(save_dir_path)
    else:
        save_dir_path = None

    # Calculate the number of iterations per epoch
    batches = len(test_data_prefetcher)
    # Interval printing
    if batches > 100:
        print_freq = 100
    else:
        print_freq = batches
    print_freq = 1
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    lpipses = AverageMeter("LPIPS", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(len(test_data_prefetcher),
                             [batch_time, psnres, ssimes, lpipses],
                             prefix=f"Test: ")

    # set the model as validation model
    unet_model.eval()

    espcn_model, fsrcnn_model, vdsr_model = ensemble_model
    
    # print(torch.cuda.memory_summary())
    
    with torch.no_grad():
        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        # Record the start time of verifying a batch
        end = time.time()

        while batch_data is not None:
            # Load batches of data
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)
            lr_up = batch_data["lr_up"].to(device, non_blocking=True)

            # Reasoning
            # sr_1 = espcn_model(lr[:,0:1,:,:])
            sr_2 = fsrcnn_model(lr[:,0:1,:,:])
            sr_3 = vdsr_model(lr_up[:,0:1,:,:])

            residual_1 = sr_2 - lr_up[:,0:1,:,:]
            residual_2 = sr_3 - lr_up[:,0:1,:,:]

            # map = torch.cat((sr_1.detach().clone(), sr_2.detach().clone(), sr_3.detach().clone()), 1)
            map = torch.cat((sr_2.detach().clone(), sr_3.detach().clone()), 1)
            
            # hypo = F.softmax(unet(map), dim=1)
            hypo = unet_model(map)
            # sr = hypo[:,0:1,:,:] * sr_1 + hypo[:,1:2,:,:] * sr_2 + hypo[:,2:3,:,:] * sr_3
            # sr = hypo[:,0:1,:,:] * sr_1 + hypo[:,1:2,:,:] * sr_3
            sr = lr_up[:,0:1,:,:] + hypo[:,0:1,:,:] * residual_1 + hypo[:,1:2,:,:] * residual_2
            # Calculate the image sharpness evaluation index
    
            psnr = psnr_model(torch.clamp(sr, min=0, max=1), gt[:,0:1,:,:])
            ssim = ssim_model(torch.clamp(sr, min=0, max=1), gt[:,0:1,:,:])
            lpips = lpips_model(torch.cat((torch.clamp(sr, min=0, max=1), lr_up[:,1:3,:,:]), 1), gt) # js.seo added

            # record current metrics
            psnres.update(psnr.item(), sr.size(0))
            ssimes.update(ssim.item(), sr.size(0))
            lpipses.update(lpips.item(), sr.size(0)) # js.seo added

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_freq == 0:
                progress.display(batch_index)

            # Save the processed image after super-resolution
            if config["TEST"]["SAVE_IMAGE"] and batch_data["image_name"] is None:
                raise ValueError("The image_name is None, please check the dataset.")
            if config["TEST"]["SAVE_IMAGE"]:
                image_name = os.path.basename(batch_data["image_name"][0])
                sr_image = tensor_to_image(sr, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir_path, image_name), sr_image)

            # Preload the next batch of data
            batch_data = test_data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg, lpipses.avg


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/test/SRRESNET_X4.yaml",
                        required=True,
                        help="Path to test config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(config, device)
    g_model = build_model(config, device)
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["G"]["COMPILED"], config["MODEL_WEIGHTS_PATH"])

    # Create a directory for saving test results
    save_dir_path = os.path.join(config["SAVE_DIR_PATH"], config["EXP_NAME"])
    if config["SAVE_IMAGE"]:
        make_directory(save_dir_path)

    test(g_model,
         test_data_prefetcher,
         psnr_model,
         ssim_model,
         device,
         config)

    # test(
    #     unet_model: nn.Module,
    #     ensemble_model: list[nn.Module],
    #     test_data_prefetcher: CUDAPrefetcher,
    #     psnr_model: nn.Module,
    #     ssim_model: nn.Module,
    #     lpips_model: nn.Module,
    #     device: torch.device,
    #     config: Any,


if __name__ == "__main__":
    main()
