# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import math
from math import sqrt
import torch
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

__all__ = [
    "ESPCN", "FSRCNN", "Generator", "Discriminator",
    "espcn_x2", "espcn_x3", "espcn_x4", "espcn_x8",
]

class SRResNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            num_rcb: int,
            upscale_factor: int
    ) -> None:
        super(SRResNet, self).__init__()
        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(channels))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

        # zoom block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(channels, out_channels, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        # out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)

class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out
# ----------------------------------
# this discriminator came from SRGAN
# ----------------------------------
class _Discriminator(nn.Module):
    def __init__(self) -> None:
        super(_Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Input image size must equal 96
        assert x.shape[2] == 96 and x.shape[3] == 96, "Image shape must equal 96x96"

        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        in_channels = 1
        
        self.conv_layer = nn.ModuleList()
        in_filters = in_channels
        dim = 32
        for i, out_filters in enumerate([dim, dim, dim*2, dim*2, dim*4, dim*4, dim*8, dim*8, dim*4, dim*4, dim*2, dim*2, dim, dim]):
        # for i, out_filters in enumerate([32, 32, 64, 64, 64, 64, 32, 32]):
            self.conv_layer.append(nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1))
            if i == 8 or i == 10 or i == 12:
                in_filters = out_filters * 2
            else:
                in_filters = out_filters

        self.Relu = nn.ReLU()
        # reconstruction block
        self.final_conv_layer = nn.Conv2d(in_filters, in_channels, 3, stride=1, padding=1)

    def forward(self, img):
        skip_conn = []
        for i in range(len(self.conv_layer)):
            img = self.conv_layer[i](img)
            if i == 1 or i == 3 or i == 5:
                skip_conn.append(img)
            img = nn.LayerNorm(img.shape[3]).cuda()(img)
            if i == 8 or i == 10 or i == 12:
                img = self.Relu(img)
                img = torch.cat((img, skip_conn.pop()), 1)
            else:
                img = self.Relu(img)
        img = self.final_conv_layer(img)
        return img

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.in_channels, self.in_height, self.in_width = self.input_shape[0], self.input_shape[1], self.input_shape[2]
        print("in_channels, in_height, in_width=", self.in_channels, self.in_height, self.in_width)
        in_channels = self.in_channels
        patch_h, patch_w = int(self.in_height / 2 ** 4), int(self.in_width / 2 ** 4)
        patch_h, patch_w = int(patch_h/patch_h), int(patch_w/patch_w)
        self.output_shape = (in_channels, patch_h, patch_w)
        print("Discriminator.output_shape=",self.output_shape)

        def ConvBlock(in_filters, out_filters, _stride=False):
            if _stride:
                return nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)
            else:
                return nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)

        self.conv_layer = nn.ModuleList()
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512, 1]):
            self.conv_layer.append(ConvBlock(in_filters, out_filters, _stride=(i != 3 and i != 4)))
            in_filters = out_filters
            
        self.LRelu = nn.LeakyReLU(0.2, inplace=True)
        self.AAP = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, img):
        for i in range(len(self.conv_layer)):
            img = self.conv_layer[i](img)
            img = nn.LayerNorm(img.shape[3]).cuda()(img)
            img = self.LRelu(img)
        img = self.AAP(img)
        return img

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class FSRCNN(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(56, 1, (9, 9), (upscale_factor, upscale_factor), (4, 4), (upscale_factor - 1, upscale_factor - 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


class ESPCN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            upscale_factor: int,
    ) -> None:
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))

        # Feature mapping
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )

        # Sub-pixel convolution layer
        self.sub_pixel = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

        # Initial model weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(module.weight.data,
                                    0.0,
                                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature_maps(x)
        x = self.sub_pixel(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x

def _discriminator(*kwargs) -> _Discriminator:
    model = _Discriminator(*kwargs)
    
def discriminator(*kwargs) -> Discriminator:
    model = Discriminator(*kwargs)

    return model

def generator() -> Generator:
    model = Generator()

    return model

def espcn_x4(**kwargs) -> ESPCN:
    model = ESPCN(upscale_factor=4, **kwargs)

    return model

class _ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(_ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. 
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F_torch.mse_loss(sr_feature, gt_feature)

        return loss

class _GradientLoss(nn.Module):
    def __init__(self, gp_weight: int , device) -> None:
        super(_GradientLoss, self).__init__()
        self.gp_weight = gp_weight
        self.device = device

    # def forward(self, D:nn.Module, real_samples: Tensor, fake_samples: Tensor) -> Tensor:
    #     """Calculates the gradient penalty loss for WGAN GP"""
    #     # Random weight term for interpolation between real and fake samples
    #     alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device=config.device, non_blocking=True)
    #     # Get random interpolation between real and fake samples
    #     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #     d_interpolates = D(interpolates)
    #     fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False).to(device=config.device, non_blocking=True)
    #     # Get gradient w.r.t. interpolates
    #     gradients = autograd.grad(
    #         outputs=d_interpolates,
    #         inputs=interpolates,
    #         grad_outputs=fake,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True,
    #         )[0]
    #     gradients = gradients.view(gradients.size(0), -1)
    #     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    #     return gradient_penalty
    
    def forward(self, D:nn.Module, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(device=self.device, non_blocking=True)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True).to(device=self.device, non_blocking=True)

        # Calculate probability of interpolated examples
        prob_interpolated = D(interpolated)
        fake = Variable(Tensor(real_data.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False).to(device=self.device, non_blocking=True)

        # Calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, 
                                inputs=interpolated,
                               grad_outputs=fake,
                               create_graph=True, 
                               retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
def content_loss(**kwargs) -> _ContentLoss:
    content_loss = _ContentLoss(**kwargs)

    return content_loss

def gradient_loss(gp_weight: int , device: torch.device) -> _GradientLoss:
    gradient_loss = _GradientLoss(gp_weight, device)

    return gradient_loss