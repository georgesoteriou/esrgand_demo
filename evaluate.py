

# !pip install https://github.com/PyTorchLightning/pytorch-lightning/archive/master.zip

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset, DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as FT
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
import pytorch_lightning as pl
from torchmetrics.functional import ssim, psnr


def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = 1 / depth - min_disp
    return disp / (max_disp - min_disp)


def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(
        device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_block6 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding=18, dilation=18)

        self.conv1x1 = nn.Conv2d(out_channels*4, out_channels, 1, 1)

    def forward(self, features):
        features_1 = self.atrous_block18(features[0])
        features_2 = self.atrous_block12(features[1])
        features_3 = self.atrous_block6(features[2])
        features_4 = self.atrous_block1(features[3])

        output_feature = [features_1, features_2, features_3, features_4]
        output_feature = torch.cat(output_feature, 1)

        return self.conv1x1(output_feature)


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # Prepare Coordinates shape [b,3,h*w]
        meshgrid = np.meshgrid(
            range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / \
            (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(
            self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        # normalize
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


class fSEModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(fSEModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [upsample(high_features)]
        features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))


class ResNetMultiImageInput(models.ResNet):

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock,
                  50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(
        block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(
            models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(
                num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(
            self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


encoder_urls = {
    "640x192": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_CS_K_MS_640x192/encoder.pth",
    "1280x384": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_K_M_1280x384/encoder.pth",
    "1024x320": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_K_MS_1024x320/encoder.pth"
}


def encoder(num_layers, arch="1280x384", pretrained=True):
    encoder = ResnetEncoder(num_layers, False)
    if pretrained:
        state_dict = load_state_dict_from_url(
            encoder_urls[arch], progress=True, map_location=torch.device("cpu"))
        feed_height = state_dict['height']
        feed_width = state_dict['width']
        filtered_dict_enc = {
            k: v for k, v in state_dict.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
    return encoder, feed_height, feed_width


class HRDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, mobile_encoder=False):
        super(HRDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.scales = scales
        self.mobile_encoder = mobile_encoder
        if mobile_encoder:
            self.num_ch_dec = np.array([4, 12, 20, 40, 80])
        else:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.all_position = ["01", "11", "21", "31",
                             "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]

        self.convs = nn.ModuleDict()
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(
                    num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(
                        num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                    + self.num_ch_dec[row]*2*(col-1),
                                                                    output_channel=self.num_ch_dec[row] * 2)
            else:
                self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                    + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if mobile_encoder:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(
                    self.num_ch_enc[row] + self.num_ch_enc[row + 1] // 2 +
                    self.num_ch_dec[row]*2*(col-1), self.num_ch_dec[row] * 2)
            else:
                if col == 1:
                    self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                                     self.num_ch_enc[row], self.num_ch_dec[row + 1])
                else:
                    self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                   + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                    self.convs["X_{}{}_Conv_1".format(
                        row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        if self.mobile_encoder:
            self.convs["dispConvScale0"] = Conv3x3(4, self.num_output_channels)
            self.convs["dispConvScale1"] = Conv3x3(8, self.num_output_channels)
            self.convs["dispConvScale2"] = Conv3x3(
                24, self.num_output_channels)
            self.convs["dispConvScale3"] = Conv3x3(
                40, self.num_output_channels)
        else:
            for i in range(4):
                self.convs["dispConvScale{}".format(i)] = Conv3x3(
                    self.num_ch_dec[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features):
        outputs = {}
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = input_features[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1 and not self.mobile_encoder:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(
                    conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        outputs[("disparity", "Scale0")] = self.sigmoid(
            self.convs["dispConvScale0"](x))
        outputs[("disparity", "Scale1")] = self.sigmoid(
            self.convs["dispConvScale1"](features["X_04"]))
        outputs[("disparity", "Scale2")] = self.sigmoid(
            self.convs["dispConvScale2"](features["X_13"]))
        outputs[("disparity", "Scale3")] = self.sigmoid(
            self.convs["dispConvScale3"](features["X_22"]))
        return outputs


decoder_urls = {
    "640x192": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_CS_K_MS_640x192/depth.pth",
    "1024x320": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_K_MS_1024x320/depth.pth",
    "1280x384": "http://www.doc.ic.ac.uk/~gs2617/models/HR_Depth_K_M_1280x384/depth.pth",
}


def decoder(num_ch_enc, arch="640x192", pretrained=True):
    depth_decoder = HRDepthDecoder(num_ch_enc=num_ch_enc)
    if pretrained:
        state_dict = load_state_dict_from_url(
            decoder_urls[arch], progress=True, map_location=torch.device("cpu"))
        depth_decoder.load_state_dict(state_dict)
    return depth_decoder


class ResidualDenseBlock(nn.Module):

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):

        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(
            channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        return conv5 * self.scale_ratio + x


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels, scale_ratio)

    def forward(self, x: torch.Tensor):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out * 0.2 + x


model_urls_d = {
    "esrgan16": "https://www.doc.ic.ac.uk/~gs2617/models/esrgan16_generator_lr_depth_before_upsamplin-epoch=269-val_psnr=26.300203.pt",
    "esrgan16_trunk": "https://www.doc.ic.ac.uk/~gs2617/models/esrgan16_generator_lr_depth_trunk.pt",
    "esrgan23_trunk": "https://www.doc.ic.ac.uk/~gs2617/models/esrgan23_generator_lr_depth_trunk.pt"
}


class Generator_d(nn.Module):
    def __init__(self, num_rrdb_blocks: int = 16):
        r""" This is an esrgan model defined by the author himself.
        We use two settings for our generator – one of them contains 8 residual blocks, with a capacity similar
        to that of SRGAN and the other is a deeper model with 16/23 RRDB blocks.
        Args:
            num_rrdb_blocks (int): How many residual in residual blocks are combined. (Default: 16).
        Notes:
            Use `num_rrdb_blocks` is 16 for TITAN 2080Ti.
            Use `num_rrdb_blocks` is 23 for Tesla A100.
        """
        super(Generator_d, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16 or 23 ResidualInResidualDenseBlock layer.
        trunk = []
        for _ in range(num_rrdb_blocks):
            trunk += [ResidualInResidualDenseBlock(
                channels=64, growth_channels=32, scale_ratio=0.2)]
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ===== Depth Layers =====

        self.conv1_d = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # (16 or 23) // 3 ResidualInResidualDenseBlock layer.
        # Since depth is initially only 1 channel instead of 3, we device by 3
        trunk_d = []
        for _ in range(num_rrdb_blocks // 3):
            trunk_d += [ResidualInResidualDenseBlock(
                channels=64, growth_channels=32, scale_ratio=0.2)]
        self.trunk_d = nn.Sequential(*trunk_d)

        self.conv2_d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # ===== End Depth Layers =====

        # Upsampling layers
        self.up1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, lr: torch.Tensor, depth: torch.Tensor):
        out1 = self.conv1(lr)
        trunk = self.trunk(out1)
        out2 = self.conv2(trunk)
        out = torch.add(out1, out2)
        out1_d = self.conv1_d(depth)
        trunk_d = self.trunk_d(out1_d)
        out2_d = self.conv2_d(trunk_d)
        out_d = torch.add(out1_d, out2_d)
        out = torch.add(out, out_d)
        out = F.leaky_relu(self.up1(F.interpolate(
            out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.up2(F.interpolate(
            out, scale_factor=2, mode="nearest")), negative_slope=0.2, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


def _gan_d(arch, num_residual_block, pretrained, progress):
    model = Generator_d(num_residual_block)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls_d[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        # model.load_state_dict(state_dict, strict=False)
    return model


def esrgan_d(num_residual_block: int = 16, pretrained: bool = False, progress: bool = True, arch="esrgan16"):
    return _gan_d(arch, num_residual_block, pretrained, progress)


class DiscriminatorForVGG(nn.Module):
    def __init__(self, image_size: int = 128):
        super(DiscriminatorForVGG, self).__init__()

        feature_map_size = image_size // 32

        self.features = nn.Sequential(
            # input is (3) x 128 x 128
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                      bias=False),  # state size. (64) x 64 x 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1,
                      bias=False),  # state size. (128) x 32 x 32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1,
                      bias=False),  # state size. (256) x 16 x 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1,
                      bias=False),  # state size. (512) x 8 x 8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1,
                      bias=False),  # state size. (512) x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_map_size * feature_map_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def discriminator_for_vgg(image_size: int = 128):
    model = DiscriminatorForVGG(image_size)
    return model


class VGGLoss(torch.nn.Module):

    def __init__(self, feature_layer: int = 35):

        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(
            *list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        vgg_loss = torch.nn.functional.l1_loss(
            self.features(source), self.features(target))

        return vgg_loss


class My_Dataset(Dataset):

    def __init__(self, image):
        self.image = image
        self.scaling_factor = 4

    def __getitem__(self, i):
        # Load Image
        hr = self.image.convert('RGB')

        hr_transforms = transforms.Compose([
            transforms.Resize((hr.height // self.scaling_factor * self.scaling_factor,
                               hr.width // self.scaling_factor * self.scaling_factor),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        lr_transforms = transforms.Compose([
            transforms.Resize((hr.height // self.scaling_factor,
                               hr.width // self.scaling_factor),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        sr_transforms = transforms.Compose([
            transforms.Resize((hr.height * self.scaling_factor,
                               hr.width * self.scaling_factor),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        lr = lr_transforms(hr)
        sr = sr_transforms(hr)
        hr = hr_transforms(hr)

        return lr, hr, sr

    def __len__(self):
        return 1


"""# Lit Model"""


def map_depth_colour(depth):
    disp_resized_np = depth.squeeze().cpu().numpy()
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[
                      :, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colormapped_im)


def to_y_tensor(img: torch.Tensor):
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966])
    return (torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.).unsqueeze(0)


class Lit_Model(pl.LightningModule):
    def __init__(self, esrgan_arch="esrgan16_trunk", num_residual_block=16):
        super().__init__()
        self.encoder_lr, self.feed_height_lr, self.feed_width_lr = encoder(
            18, arch="640x192", pretrained=True)
        self.decoder_lr = decoder(
            self.encoder_lr.num_ch_enc, arch="640x192", pretrained=True)

        self.encoder_hr, self.feed_height_hr, self.feed_width_hr = encoder(
            18, arch="1280x384", pretrained=True)
        self.decoder_hr = decoder(
            self.encoder_hr.num_ch_enc, arch="1280x384", pretrained=True)

        self.generator_d = esrgan_d(
            num_residual_block, pretrained=True, arch=esrgan_arch)
        self.stats = []

    def predict_depth_lr(self, lr):
        original_height, original_width = lr.shape[2], lr.shape[3]
        lr_feed = FT.resize(lr, (self.feed_height_lr, self.feed_width_lr))
        features = self.encoder_lr(lr_feed)
        outputs = self.decoder_lr(features)
        disp = outputs[("disparity", "Scale0")]
        disp_resized = FT.resize(disp, (original_height, original_width))
        return disp_resized

    def predict_depth_hr(self, hr):
        original_height, original_width = hr.shape[2], hr.shape[3]
        hr_feed = FT.resize(hr, (self.feed_height_hr, self.feed_width_hr))
        features = self.encoder_hr(hr_feed)
        outputs = self.decoder_hr(features)
        disp = outputs[("disparity", "Scale0")]
        disp_resized = FT.resize(disp, (original_height, original_width))
        return disp_resized

    def forward(self, batch):
        lr, hr, sr_bicubic = batch
        height, width = hr.shape[2], hr.shape[3]

        # # Lr to SR
        # bicubic_enlarge = transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC)
        # depth_lr = self.predict_depth_lr(lr)
        # lrx4 = self.generator_d(lr, depth_lr).clamp(0, 1)
        # depth_lr_bicubic = map_depth_colour(bicubic_enlarge(depth_lr))
        # lr_bicubic = bicubic_enlarge(lr).clamp(0, 1)
        # psnr_metric = psnr(to_y_tensor(lrx4), to_y_tensor(hr), data_range=255.)
        # ssim_metric = ssim(to_y_tensor(lrx4), to_y_tensor(hr), data_range=255.)

        # HR to SR
        depth_hr = self.predict_depth_hr(hr)
        hrx4 = self.generator_d(hr, depth_hr).clamp(0, 1)
        depth_hr = map_depth_colour(depth_hr)

        # return (lr_bicubic, depth_lr_bicubic, lrx4, psnr_metric, ssim_metric, hr, depth_hr, hrx4)
        return FT.to_pil_image(hrx4[0]), depth_hr, FT.to_pil_image(sr_bicubic[0])


def enhance(img):
    dataset = My_Dataset(img)
    dataloader = DataLoader(dataset, batch_size=1,
                            num_workers=4, shuffle=False)

    model = Lit_Model(esrgan_arch=f"esrgan23_trunk", num_residual_block=23)

    trainer = pl.Trainer(
        gpus=0,
        progress_bar_refresh_rate=0)
    return trainer.predict(model, dataloader)
