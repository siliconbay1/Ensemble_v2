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

import cv2
import torch
# from flatbuffers.builder import np
import numpy as np
from torch import nn

import config
import imgproc
import model
from unet import unet_model
from utils import load_state_dict

import torch.nn.functional as F

from torchvision.utils import save_image

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    espcn = model.__dict__["ESPCN"](in_channels = 1, out_channels = 1, channels = 64, upscale_factor = 4)
    fsrcnn = model.__dict__["FSRCNN"](upscale_factor=4)
    g_model = unet_model.__dict__["UNet"](config.in_channels, config.out_channels)
    
    espcn = espcn.to(device)
    fsrcnn = fsrcnn.to(device)
    g_model = g_model.to(device)

    return espcn, fsrcnn, g_model


def main(args):
    device = choice_device(args.device_type)

    # Initialize the model
    espcn_model, fsrcnn_model, g_model = build_model(device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    espcn_model = load_state_dict(espcn_model, args.espcn_model_weights_path)
    fsrcnn_model = load_state_dict(fsrcnn_model, args.fsrcnn_model_weights_path)
    g_model = load_state_dict(g_model, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    espcn_model.eval()
    fsrcnn_model.eval()
    g_model.eval()

    lr_y_tensor, lr_cb_image, lr_cr_image = imgproc.preprocess_one_image(args.inputs_path, device)

    bic_cb_image = cv2.resize(lr_cb_image,
                              (int(lr_cb_image.shape[1] * args.upscale_factor),
                               int(lr_cb_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    bic_cr_image = cv2.resize(lr_cr_image,
                              (int(lr_cr_image.shape[1] * args.upscale_factor),
                               int(lr_cr_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    # Use the model to generate super-resolved images
    
    print("lr_y_tensor.shape=", lr_y_tensor.shape)
    with torch.no_grad():
        sr_1 = espcn_model(lr_y_tensor.detach().clone())
        sr_2 = fsrcnn_model(lr_y_tensor.detach().clone())
        map = torch.cat((sr_1, sr_2), 1)
        hypo = g_model(map)
        sr_y_tensor = sr_1 * hypo + sr_2 * (1 - hypo)
    print("sr_y_tensor.shape=", sr_y_tensor.shape)
    
    # Save image
    # sr_y_tensor = F.interpolate(lr_y_tensor, scale_factor=4, mode='bicubic')
    sr_y_image = imgproc.tensor_to_image(sr_y_tensor, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    
    # bic_cb_image = np.full_like(bic_cb_image, 128 / 255.0)
    # bic_cr_image = np.full_like(bic_cr_image, 128 / 255.0)

    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(args.output_path, sr_image * 255.0)

    print(f"SR image save to `{args.output_path}`")

    sr_y_image = imgproc.tensor_to_image(sr_1, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0

    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(args.output_espcn_path, sr_image * 255.0)

    print(f"SR image save to `{args.output_espcn_path}`")
    
    sr_y_image = imgproc.tensor_to_image(sr_2, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0

    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(args.output_fsrcnn_path, sr_image * 255.0)

    print(f"SR image save to `{args.output_fsrcnn_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="espcn_x4")
    parser.add_argument("--upscale_factor",
                        type=int,
                        default=4)
    parser.add_argument("--inputs_path",
                        type=str,
                        default="./figure/comic.png",
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default="./figure/sr_comic.png",
                        help="Super-resolution image path.")
    parser.add_argument("--output_espcn_path",
                        type=str,
                        default="./figure/sr_espcn_comic.png",
                        help="Super-resolution image path.")
    parser.add_argument("--output_fsrcnn_path",
                        type=str,
                        default="./figure/sr_fsrcnn_comic.png",
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./samples/ensemble_x4/g_epoch_30.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--espcn_model_weights_path",
                        type=str,
                        default="./pretrained/ESPCN_x4-T91-64bf5ee4.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--fsrcnn_model_weights_path",
                        type=str,
                        default="./pretrained/fsrcnn_x4-T91-97a30bfb.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
