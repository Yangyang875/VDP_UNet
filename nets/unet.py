import torch
import torch.nn as nn

from nets.resnet import resnet50
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Unet(nn.Module):
    def __init__(self, num_classes = 21, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone

    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True

#加入CBAM
# import torch
# import torch.nn as nn
#
# from nets.resnet import resnet50
# from nets.vgg import VGG16
#
#
# # 实现CBAM模块
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes)
#         self.sa = SpatialAttention()
#
#     def forward(self, x):
#         x = self.ca(x) * x
#         x = self.sa(x) * x
#         return x
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#         # 添加CBAM注意力机制
#         self.cbam = CBAM(out_size)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         # 应用CBAM注意力机制
#         outputs = self.cbam(outputs)
#         return outputs
#
#
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
#         super(Unet, self).__init__()
#         if backbone == 'vgg':
#             self.vgg = VGG16(pretrained=pretrained)
#             in_filters = [192, 384, 768, 1024]
#         elif backbone == "resnet50":
#             self.resnet = resnet50(pretrained=pretrained)
#             in_filters = [192, 512, 1024, 3072]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
#         out_filters = [64, 128, 256, 512]
#
#         # upsampling
#         # 64,64,512
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         # 128,128,256
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         # 256,256,128
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         # 512,512,64
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])
#
#         if backbone == 'resnet50':
#             self.up_conv = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#             )
#         else:
#             self.up_conv = None
#
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)
#
#         self.backbone = backbone
#
#     def forward(self, inputs):
#         if self.backbone == "vgg":
#             [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
#         elif self.backbone == "resnet50":
#             [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         if self.up_conv != None:
#             up1 = self.up_conv(up1)
#
#         final = self.final(up1)
#
#         return final
#
#     def freeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = False
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = True
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = True



# #方法2：加入CBAM注意力机制，先 Conv2d + ReLU，再应用 CBAM
#
# import torch
# import torch.nn as nn
#
# from nets.resnet import resnet50
# from nets.vgg import VGG16
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(CBAM, self).__init__()
#         # 通道注意力模块（Channel Attention）
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
#             nn.Sigmoid()
#         )
#
#         # 空间注意力模块（Spatial Attention）
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # 计算通道注意力
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         ch_att = avg_out + max_out
#         x = x * ch_att  # 乘上通道注意力
#
#         # 计算空间注意力
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         max_pool, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_att = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))
#         x = x * spatial_att  # 乘上空间注意力
#         return x
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#         # 添加 CBAM 注意力模块
#         self.cbam = CBAM(out_size)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)  # 跳跃连接+上采样
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#
#         # CBAM 注意力机制
#         outputs = self.cbam(outputs)
#         return outputs
#
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
#         super(Unet, self).__init__()
#         if backbone == 'vgg':
#             self.vgg = VGG16(pretrained=pretrained)
#             in_filters = [192, 384, 768, 1024]
#         elif backbone == "resnet50":
#             self.resnet = resnet50(pretrained=pretrained)
#             in_filters = [192, 512, 1024, 3072]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
#         out_filters = [64, 128, 256, 512]
#
#         # upsampling
#         # 64,64,512
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         # 128,128,256
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         # 256,256,128
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         # 512,512,64
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])
#
#         if backbone == 'resnet50':
#             self.up_conv = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#             )
#         else:
#             self.up_conv = None
#
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)
#
#         self.backbone = backbone
#
#     def forward(self, inputs):
#         if self.backbone == "vgg":
#             [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
#         elif self.backbone == "resnet50":
#             [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         if self.up_conv != None:
#             up1 = self.up_conv(up1)
#
#         final = self.final(up1)
#
#         return final
#
#     def freeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = False
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = True
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = True

# DANet（空洞注意力，适用于多尺度目标）

'''
✅ 多尺度感受野：采用不同空洞率的卷积，扩大感受野。
✅ 适用于高分辨率遥感/医学影像：更好地处理复杂场景。
✅ 避免计算量过大：比 Non-local 计算更快。
'''

# import torch
# import torch.nn as nn
#
# from nets.resnet import resnet50
# from nets.vgg import VGG16
#
#
# class DA_Attention(nn.Module):
#     def __init__(self, in_channels):
#         super(DA_Attention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=4, dilation=4)
#         self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=8, dilation=8)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv_final = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
#
#     def forward(self, x):
#         x1 = self.relu(self.conv1(x))
#         x2 = self.relu(self.conv2(x))
#         x3 = self.relu(self.conv3(x))
#         out = torch.cat([x1, x2, x3], dim=1)
#         out = self.conv_final(out)
#         return out + x  # 残差连接
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#         # DANet 空洞注意力
#         self.da_att = DA_Attention(out_size)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)  # 跳跃连接+上采样
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#
#         # 添加 DANet 注意力
#         outputs = self.da_att(outputs)
#         return outputs
#
#
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
#         super(Unet, self).__init__()
#         if backbone == 'vgg':
#             self.vgg = VGG16(pretrained=pretrained)
#             in_filters = [192, 384, 768, 1024]
#         elif backbone == "resnet50":
#             self.resnet = resnet50(pretrained=pretrained)
#             in_filters = [192, 512, 1024, 3072]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
#         out_filters = [64, 128, 256, 512]
#
#         # upsampling
#         # 64,64,512
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3])
#         # 128,128,256
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2])
#         # 256,256,128
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1])
#         # 512,512,64
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0])
#
#         if backbone == 'resnet50':
#             self.up_conv = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#             )
#         else:
#             self.up_conv = None
#
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)
#
#         self.backbone = backbone
#
#     def forward(self, inputs):
#         if self.backbone == "vgg":
#             [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
#         elif self.backbone == "resnet50":
#             [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         if self.up_conv != None:
#             up1 = self.up_conv(up1)
#
#         final = self.final(up1)
#
#         return final
#
#     def freeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = False
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = True
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = True


#使用UNet原始主干网络
# import torch
# import torch.nn as nn
#
#
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(unetUp, self).__init__()
#         # in_size 改成自动计算通道数
#         self.conv1  = nn.Conv2d(in_size + out_size, out_size, kernel_size=3, padding=1)
#         self.conv2  = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up     = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu   = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         upsampled = self.up(inputs2)
#         outputs = torch.cat([inputs1, upsampled], 1)  # 这里通道数会自动适应
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs
#
#
#
# class Unet(nn.Module):
#     def __init__(self, num_classes=21):
#         super(Unet, self).__init__()
#
#         # UNet 原始编码器
#         self.enc1 = self.conv_block(3, 64)
#         self.enc2 = self.conv_block(64, 128)
#         self.enc3 = self.conv_block(128, 256)
#         self.enc4 = self.conv_block(256, 512)
#         self.enc5 = self.conv_block(512, 1024)
#
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.up_concat4 = unetUp(1024, 512)
#         self.up_concat3 = unetUp(512, 256)
#         self.up_concat2 = unetUp(256, 128)
#         self.up_concat1 = unetUp(128, 64)
#
#         self.final = nn.Conv2d(64, num_classes, 1)
#
#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, inputs):
#         enc1 = self.enc1(inputs)
#         enc2 = self.enc2(self.pool(enc1))
#         enc3 = self.enc3(self.pool(enc2))
#         enc4 = self.enc4(self.pool(enc3))
#         enc5 = self.enc5(self.pool(enc4))
#
#         up4 = self.up_concat4(enc4, enc5)
#         up3 = self.up_concat3(enc3, up4)
#         up2 = self.up_concat2(enc2, up3)
#         up1 = self.up_concat1(enc1, up2)
#
#         final = self.final(up1)
#         return final
#


# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import init
# from nets.resnet import resnet50
# from nets.vgg import VGG16
#
# # -------------------- Polarized Self-Attention -------------------- #
# class ParallelPolarizedSelfAttention(nn.Module):
#     def __init__(self, channel=512):
#         super().__init__()
#         self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
#         self.ch_wq = nn.Conv2d(channel, 1, kernel_size=1)
#         self.softmax_channel = nn.Softmax(1)
#         self.softmax_spatial = nn.Softmax(-1)
#         self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=1)
#         self.ln = nn.LayerNorm(channel)
#         self.sigmoid = nn.Sigmoid()
#         self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=1)
#         self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=1)
#         self.agp = nn.AdaptiveAvgPool2d((1, 1))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         # Channel-only Self-Attention
#         channel_wv = self.ch_wv(x).reshape(b, c // 2, -1)
#         channel_wq = self.ch_wq(x).reshape(b, -1, 1)
#         channel_wq = self.softmax_channel(channel_wq)
#         channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
#         channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2, 1).reshape(b, c, 1, 1)
#         channel_out = channel_weight * x
#
#         # Spatial-only Self-Attention
#         spatial_wv = self.sp_wv(x).reshape(b, c // 2, -1)
#         spatial_wq = self.agp(self.sp_wq(x)).permute(0, 2, 3, 1).reshape(b, 1, c // 2)
#         spatial_wq = self.softmax_spatial(spatial_wq)
#         spatial_wz = torch.matmul(spatial_wq, spatial_wv).reshape(b, 1, h, w)
#         spatial_weight = self.sigmoid(spatial_wz)
#         spatial_out = spatial_weight * x
#
#         out = spatial_out + channel_out
#         return out
#
# # -------------------- Attention Wrapper for 2D Features -------------------- #
# class PSA2D(nn.Module):
#     def __init__(self, in_channels):
#         super(PSA2D, self).__init__()
#         self.psa = ParallelPolarizedSelfAttention(channel=in_channels)
#
#     def forward(self, x):
#         return self.psa(x)
#
# # -------------------- U-Net 上采样模块，集成 PSA -------------------- #
# class unetUp(nn.Module):
#     def __init__(self, in_size, out_size, use_attention=False):
#         super(unetUp, self).__init__()
#         self.use_attention = use_attention
#         if self.use_attention:
#             self.att = PSA2D(in_channels=in_size)
#         self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
#         self.up = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         outputs = torch.cat([inputs1, self.up(inputs2)], 1)
#         if self.use_attention:
#             outputs = self.att(outputs)
#         outputs = self.conv1(outputs)
#         outputs = self.relu(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.relu(outputs)
#         return outputs
#
# # -------------------- 主体 UNet 网络 -------------------- #
# class Unet(nn.Module):
#     def __init__(self, num_classes=21, pretrained=False, backbone='vgg', use_attention=True):
#         super(Unet, self).__init__()
#         self.use_attention = use_attention
#         if backbone == 'vgg':
#             self.vgg = VGG16(pretrained=pretrained)
#             in_filters = [192, 384, 768, 1024]
#         elif backbone == "resnet50":
#             self.resnet = resnet50(pretrained=pretrained)
#             in_filters = [192, 512, 1024, 3072]
#         else:
#             raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
#
#         out_filters = [64, 128, 256, 512]
#
#         self.up_concat4 = unetUp(in_filters[3], out_filters[3], use_attention=self.use_attention)
#         self.up_concat3 = unetUp(in_filters[2], out_filters[2], use_attention=self.use_attention)
#         self.up_concat2 = unetUp(in_filters[1], out_filters[1], use_attention=self.use_attention)
#         self.up_concat1 = unetUp(in_filters[0], out_filters[0], use_attention=self.use_attention)
#
#         if backbone == 'resnet50':
#             self.up_conv = nn.Sequential(
#                 nn.UpsamplingBilinear2d(scale_factor=2),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
#                 nn.ReLU(),
#             )
#         else:
#             self.up_conv = None
#
#         self.final = nn.Conv2d(out_filters[0], num_classes, 1)
#         self.backbone = backbone
#
#     def forward(self, inputs):
#         if self.backbone == "vgg":
#             [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
#         elif self.backbone == "resnet50":
#             [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
#
#         up4 = self.up_concat4(feat4, feat5)
#         up3 = self.up_concat3(feat3, up4)
#         up2 = self.up_concat2(feat2, up3)
#         up1 = self.up_concat1(feat1, up2)
#
#         if self.up_conv is not None:
#             up1 = self.up_conv(up1)
#
#         final = self.final(up1)
#         return final
#
#     def freeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = False
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = False
#
#     def unfreeze_backbone(self):
#         if self.backbone == "vgg":
#             for param in self.vgg.parameters():
#                 param.requires_grad = True
#         elif self.backbone == "resnet50":
#             for param in self.resnet.parameters():
#                 param.requires_grad = True
#
#
#
