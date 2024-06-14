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
from utils import load_state_dict, load_pretrained_state_dict

import torch.nn.functional as F

from torchvision.utils import save_image
from utils import build_iqa_model

print(torch.__version__)

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

def extract_prefix_from_file_path(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    prefix = file_name_without_extension.split('_')[0]
    return prefix

def extract_filename_from_file_path(file_path):
    file_name = os.path.basename(file_path)
    file_name_without_extension = os.path.splitext(file_name)[0]
    return file_name_without_extension

def build_model(device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    espcn = model.__dict__["ESPCN"](in_channels = 1, out_channels = 1, channels = 64, upscale_factor = 4)
    fsrcnn = model.__dict__["FSRCNN"](upscale_factor=4)
    vdsr = model.__dict__["VDSR"]()
    g_model = model.__dict__["UNet"](n_channels=2, n_classes=2)
    
    espcn = espcn.to(device)
    fsrcnn = fsrcnn.to(device)
    vdsr = vdsr.to(device)
    g_model = g_model.to(device)

    return espcn, fsrcnn, vdsr, g_model

def main(args):
    device = torch.device(args.device)

    # Initialize the model
    espcn_model, fsrcnn_model, vdsr_model, g_model = build_model(device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    espcn_model = load_pretrained_state_dict(espcn_model, False, args.espcn_model_weights_path)
    fsrcnn_model = load_pretrained_state_dict(fsrcnn_model, False, args.fsrcnn_model_weights_path)
    vdsr_model = load_pretrained_state_dict(vdsr_model, False, args.vdsr_model_weights_path)
    g_model = load_pretrained_state_dict(g_model, False, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    espcn_model.eval()
    fsrcnn_model.eval()
    vdsr_model.eval()
    g_model.eval()

    lr_tensor, lr_y_tensor, lr_y_image, lr_cb_image, lr_cr_image = imgproc.old_preprocess_one_image(args.inputs_path, device)
    
    bic_y_image = cv2.resize(lr_y_image,
                              (int(lr_cb_image.shape[1] * args.upscale_factor),
                               int(lr_cb_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)

    bic_cb_image = cv2.resize(lr_cb_image,
                              (int(lr_cb_image.shape[1] * args.upscale_factor),
                               int(lr_cb_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    bic_cr_image = cv2.resize(lr_cr_image,
                              (int(lr_cr_image.shape[1] * args.upscale_factor),
                               int(lr_cr_image.shape[0] * args.upscale_factor)),
                              interpolation=cv2.INTER_CUBIC)
    
    bic_y_tensor = F.interpolate(lr_y_tensor, None, args.upscale_factor, mode='bicubic', align_corners=False)
    bic_tensor = F.interpolate(lr_tensor, None, args.upscale_factor, mode='bicubic', align_corners=False)

    # Use the model to generate super-resolved images
    
    with torch.no_grad():
        sr_1 = espcn_model(lr_y_tensor.detach().clone())
        sr_2 = fsrcnn_model(lr_y_tensor.detach().clone())
        sr_3 = vdsr_model(bic_y_tensor.detach().clone())
        map = torch.cat((sr_1.detach().clone(), sr_2.detach().clone()), 1)
        hypo = g_model(map)
        sr_y_tensor = sr_1 * hypo[:,0:1,:,:] + sr_2 * hypo[:,1:2,:,:]
    
    # Save image
    # sr_y_tensor = F.interpolate(lr_y_tensor, scale_factor=4, mode='bicubic')
    
    # bic_cb_image = np.full_like(bic_cb_image, 128 / 255.0)
    # bic_cr_image = np.full_like(bic_cr_image, 128 / 255.0)
    
    prefix = extract_prefix_from_file_path(args.inputs_path)
    suffix = extract_filename_from_file_path(args.output_path)
    
    gt_image = cv2.imread(f"./figure/{prefix}.png").astype(np.float32) / 255.0
    gt_ycbcr_image = imgproc.bgr_to_ycbcr(gt_image, only_use_y_channel=False)
    gt = imgproc.image_to_tensor(gt_ycbcr_image, False, False).unsqueeze_(0)
    gt = gt.to(device, non_blocking=True)
    
    print("gt.shape=", gt.shape)
    print("sr_y_tensor.shape=", sr_y_tensor.shape)

    psnr_model, ssim_model, lpips_model = build_iqa_model(4, False, device)
    
    sr_ycbcr_image = cv2.merge([bic_y_image, bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(f"./figure/{prefix}_x{args.upscale_factor}_bicubic.png", sr_image * 255.0)
    print(f"SR image save to {prefix}_x{args.upscale_factor}_bicubic.png")
    psnr = psnr_model(torch.clamp(bic_y_tensor, min=0, max=1), gt[:,0:1,:,:])
    ssim = ssim_model(torch.clamp(bic_y_tensor, min=0, max=1), gt[:,0:1,:,:])
    lpips = lpips_model(torch.cat((torch.clamp(bic_y_tensor, min=0, max=1), torch.clamp(bic_tensor[:,1:3,:,:], min=0, max=1)), 1), gt)
    print("(psnr, ssim) = ({:.4f}, {:.4f})".format(round(psnr.item(), 4), round(ssim.item(), 4)))

    sr_y_image = imgproc.tensor_to_image(sr_1, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(f"./figure/{prefix}_x{args.upscale_factor}_espcn.png", sr_image * 255.0)
    print(f"SR image save to {prefix}_x{args.upscale_factor}_espcn.png`")
    
    psnr = psnr_model(torch.clamp(sr_1, min=0, max=1), gt[:,0:1,:,:])
    ssim = ssim_model(torch.clamp(sr_1, min=0, max=1), gt[:,0:1,:,:])
    lpips = lpips_model(torch.cat((torch.clamp(sr_1, min=0, max=1), torch.clamp(bic_tensor[:,1:3,:,:], min=0, max=1)), 1), gt)
    print("(psnr, ssim) = ({:.4f}, {:.4f})".format(round(psnr.item(), 4), round(ssim.item(), 4)))
    
    sr_y_image = imgproc.tensor_to_image(sr_2, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(f"./figure/{prefix}_x{args.upscale_factor}_fsrcnn.png", sr_image * 255.0)
    print(f"SR image save to {prefix}_x{args.upscale_factor}_fsrcnn.png")

    psnr = psnr_model(torch.clamp(sr_2, min=0, max=1), gt[:,0:1,:,:])
    ssim = ssim_model(torch.clamp(sr_2, min=0, max=1), gt[:,0:1,:,:])
    lpips = lpips_model(torch.cat((torch.clamp(sr_2, min=0, max=1), torch.clamp(bic_tensor[:,1:3,:,:], min=0, max=1)), 1), gt)
    print("(psnr, ssim) = ({:.4f}, {:.4f})".format(round(psnr.item(), 4), round(ssim.item(), 4)))
    
    sr_y_image = imgproc.tensor_to_image(sr_3, range_norm=False, half=False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image[:, :, 0], bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(f"./figure/{prefix}_x{args.upscale_factor}_vdsr.png", sr_image * 255.0)
    print(f"SR image save to {prefix}_x{args.upscale_factor}_vdsr.png")

    psnr = psnr_model(torch.clamp(sr_3, min=0, max=1), gt[:,0:1,:,:])
    ssim = ssim_model(torch.clamp(sr_3, min=0, max=1), gt[:,0:1,:,:])
    lpips = lpips_model(torch.cat((torch.clamp(sr_3, min=0, max=1), torch.clamp(bic_tensor[:,1:3,:,:], min=0, max=1)), 1), gt)
    print("(psnr, ssim) = ({:.4f}, {:.4f})".format(round(psnr.item(), 4), round(ssim.item(), 4)))

    sr_y_image = imgproc.tensor_to_image(sr_y_tensor.detach().clone(), range_norm=False, half=True)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image, bic_cb_image, bic_cr_image])
    sr_image = imgproc.ycbcr_to_bgr(sr_ycbcr_image)
    cv2.imwrite(f"./figure/{prefix}_x{args.upscale_factor}_{suffix}.png", sr_image * 255.0)
    print(f"SR image save to {prefix}_x{args.upscale_factor}_{suffix}.png")
    
    psnr = psnr_model(torch.clamp(sr_y_tensor, min=0, max=1), gt[:,0:1,:,:])
    ssim = ssim_model(torch.clamp(sr_y_tensor, min=0, max=1), gt[:,0:1,:,:])
    lpips = lpips_model(torch.clamp(bic_tensor, min=0, max=1), gt)
    print("(psnr, ssim) = ({:.4f}, {:.4f})".format(round(psnr.item(), 4), round(ssim.item(), 4)))

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
                        default="./figure/sr.png",
                        help="Super-resolution image path.")
    parser.add_argument("--output_espcn_path",
                        type=str,
                        default="./figure/sr_espcn.png",
                        help="Super-resolution image path.")
    parser.add_argument("--output_fsrcnn_path",
                        type=str,
                        default="./figure/sr_fsrcnn.png",
                        help="Super-resolution image path.")
    parser.add_argument("--output_vdsr_path",
                        type=str,
                        default="./figure/sr_vdsr.png",
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
    parser.add_argument("--vdsr_model_weights_path",
                        type=str,
                        default="./pretrained/vdsr-TB291-fef487db.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run model.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    args = parser.parse_args()

    main(args)
    
    
    # def old_preprocess_one_image(image_path: str, device: torch.device) -> [Tensor, ndarray, ndarray]:
    #     image = cv2.imread(image_path).astype(np.float32) / 255.0

    # # BGR to YCbCr
    # ycbcr_image = bgr_to_ycbcr(image, only_use_y_channel=False)

    # # Split YCbCr image data
    # y_image, cb_image, cr_image = cv2.split(ycbcr_image)

    # # Convert image data to pytorch format data
    # y_tensor = image_to_tensor(y_image, False, False).unsqueeze_(0)

    # # Transfer tensor channel image format data to CUDA device
    # y_tensor = y_tensor.to(device=device, non_blocking=True)
    
    # # Convert image data to pytorch format data
    # tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    # # Transfer tensor channel image format data to CUDA device
    # tensor = tensor.to(device=device, non_blocking=True)

    # return tensor, y_tensor, y_image, cb_image, cr_image
