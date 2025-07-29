import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# 并未使用M1的设计
# class Fusion_Attention_M1(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Fusion_Attention_M1, self).__init__()

#         self.Fusion1 = nn.Sequential(
#             nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=True, dilation=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         # self.Fusion2 = nn.Sequential(
#         #     nn.Conv2d(out_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.ReLU(inplace=True)
#         # )

#         # self.Attention1 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.Sigmoid()
#         # )
#         # self.Attention2 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.Sigmoid()
#         # )

#     def forward(self, x1, x2):

#         x = torch.cat((x1, x2), dim=1)
#         x = self.Fusion1(x)

#         # x_attention_1 = self.Attention1(x)
#         # x_attention_2 = self.Attention2(x)

#         # x_output_1 = x1 * x_attention_1
#         # x_output_2 = x2 * x_attention_2

#         # x = torch.cat((x_output_1, x_output_2), dim=1)
#         # x = self.Fusion2(x)

#         return x
    
class Fusion_Attention_M2_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion_Attention_M2_3D, self).__init__()

        self.Fusion1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=True), # groups=1, dilation=1
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.Fusion2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1, groups=2, bias=True), # groups=1, dilation=1
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # self.Attention_query1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.Sigmoid()
        # )
        # self.Attention_key1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.Sigmoid()
        # )

        # self.Attention_query2 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.Sigmoid()
        # )
        # self.Attention_key2 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(in_channels),
        #     nn.Sigmoid()
        # )

        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), # groups=1, dilation=1
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_2 = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), # groups=1, dilation=1
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.softmax  = nn.Softmax(dim=-1)
        
        self.Attention1_C = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )
        self.Attention2_C = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )

        self.Attention1_S = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )
        self.Attention2_S = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=True),
            nn.BatchNorm3d(in_channels),
            nn.Sigmoid()
        )

        self.Fusion3 = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x1, x2):
        # x1 = self.conv_1(x1)
        # x2 = self.conv_2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.Fusion1(x)

        x_attention_C1 = self.Attention1_C(x)
        x_attention_C2 = self.Attention2_C(x)
        x_C1 = x1 * x_attention_C1
        x_C2 = x2 * x_attention_C2
        # x = torch.cat((x_C1, x_C2), dim=1)

        # m_batchsize, C, height, width = x.size()

        # x_query1 = self.Attention_query1(x)     # 这里做一个自注意力的事情怎么样？
        # x_key1 = self.Attention_key1(x)
        
        # proj_query1 = x_query1.view(m_batchsize, C, -1)
        # proj_key1 = x_key1.view(m_batchsize, C, -1).permute(0, 2, 1)
        # proj_value1 = x1.view(m_batchsize, C, -1)

        # energy1 = torch.bmm(proj_query1, proj_key1)
        # energy1 = energy1  / (C ** 0.5)
        # attention1 = self.softmax(energy1)
        # energy_new1 = torch.max(energy1, -1, keepdim=True)[0].expand_as(energy1)-energy1
        # attention1 = self.softmax(energy_new1)

        # out1 = torch.bmm(attention1, proj_value1)
        # out1 = out1.view(m_batchsize, C, height, width) + x1

        # x_query2 = self.Attention_query2(x)
        # x_key2 = self.Attention_key2(x)

        # proj_query2 = x_query2.view(m_batchsize, C, -1)
        # proj_key2 = x_key2.view(m_batchsize, C, -1).permute(0, 2, 1)
        # proj_value2 = x2.view(m_batchsize, C, -1)

        # energy2 = torch.bmm(proj_query2, proj_key2)
        # energy2 = energy2  / (C ** 0.5)
        # attention2 = self.softmax(energy2)
        # energy_new2 = torch.max(energy2, -1, keepdim=True)[0].expand_as(energy2)-energy2
        # attention2 = self.softmax(energy_new2)

        # out2 = torch.bmm(attention2, proj_value2)
        # out2 = out2.view(m_batchsize, C, height, width) + x2

        # x = torch.cat((out1, out2), dim=1)

        x = torch.stack((x_C1, x_C2), dim=0)  
        x = x.permute(1, 0, 2, 3, 4, 5)  
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4], x.shape[5]) 

        x = self.Fusion2(x)
        x_attention_S1 = self.Attention1_S(x)
        x_attention_S2 = self.Attention2_S(x)
        x_S1 = x1 * x_attention_S1
        x_S2 = x2 * x_attention_S2
        x = torch.cat((x_S1, x_S2), dim=1)

        x = self.Fusion3(x)

        return x

# class MFwaveNet3(nn.Module):
#     def __init__(self, in_channels=3, num_classes=2):
#         super(MFwaveNet3, self).__init__()
#         # Raw network
#         # x = torch.cat((out1, out2), dim=1)

#         x = self.Fusion2(x)

#         return x

# class MFwaveNet2_3D(nn.Module):
#     def __init__(self, in_channels=1, output_channels=3, init_features=24):
#         super(MFwaveNet2_3D, self).__init__()
#         self.features = init_features

#         self.R_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.R_encoder1 = MFwaveNet2_3D._block(in_channels, self.features, name="enc1")
#         self.R_encoder2 = MFwaveNet2_3D._block(self.features, self.features * 2, name="enc2")
#         self.R_encoder3 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc3")
#         self.R_encoder4 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc4")
#         self.R_bottleneck = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="bottleneck")

#         self.RS_Up5 = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
#         self.RS_Up_conv5 = MFwaveNet2_3D._block(self.features * 8 + self.features * 8 * 4, self.features * 8, name="dec4") # self.features * 16 + 
#         self.RS_Up4 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
#         self.RS_Up_conv4 = MFwaveNet2_3D._block(self.features * 4 + self.features * 4 * 4, self.features * 4, name="dec3") # self.features * 8 + 
#         self.RS_Up3 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
#         self.RS_Up_conv3 = MFwaveNet2_3D._block(self.features * 2 + self.features * 2 * 4, self.features * 2, name="dec2") # self.features * 4 + 
#         self.RS_Up2 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
#         self.RS_Up_conv2 = MFwaveNet2_3D._block(self.features + self.features * 4, self.features * 2, name="dec1") # self.features * 2 + 
#         self.RS_Conv_1x1 = nn.Conv3d(self.features * 2, output_channels, kernel_size=1, stride=1, padding=0)

#         # Select network
#         self.S_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.S_encoder1 = MFwaveNet2_3D._block(in_channels, self.features, name="enc1")
#         self.S_encoder2 = MFwaveNet2_3D._block(self.features, self.features * 2, name="enc2")
#         self.S_encoder3 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc3")
#         self.S_encoder4 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc4")
#         self.S_bottleneck = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="bottleneck")

#         # Mixup network
#         self.M_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.M_encoder1 = MFwaveNet2_3D._block(in_channels, self.features, name="enc1")
#         self.M_encoder2 = MFwaveNet2_3D._block(self.features, self.features * 2, name="enc2")
#         self.M_encoder3 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc3")
#         self.M_encoder4 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc4")
#         self.M_bottleneck = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="bottleneck")

#         self.M_Up5 = nn.ConvTranspose3d(self.features * 8 * 2, self.features * 8, kernel_size=2, stride=2)
#         self.RSM_Up_conv5 = MFwaveNet2_3D._block(self.features * 8 * 2 + self.features * 8, self.features * 8, name="dec4") #  + self.features * 8
#         self.RSM_Up4 = nn.ConvTranspose3d(self.features * 4 * 2, self.features * 4, kernel_size=2, stride=2)
#         self.RSM_Up_conv4 = MFwaveNet2_3D._block(self.features * 4 * 2 + self.features * 4, self.features * 4, name="dec3") #  + self.features * 4
#         self.RSM_Up3 = nn.ConvTranspose3d(self.features * 2 * 2, self.features * 2, kernel_size=2, stride=2)
#         self.RSM_Up_conv3 = MFwaveNet2_3D._block(self.features * 2 * 2 + self.features * 2, self.features * 2, name="dec2") #  + self.features * 2
#         self.RSM_Up2 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
#         self.RSM_Up_conv2 = MFwaveNet2_3D._block(self.features * 2 + self.features, self.features, name="dec1") #  + self.features
#         self.RSM_Conv_1x1 = nn.Conv3d(self.features, output_channels, kernel_size=1, stride=1, padding=0)

#         # fusion
#         self.fusion_RS1 = Fusion_Attention_M2_3D(self.features, self.features) 
#         # self.fusion_RSM1 = Fusion_Attention_M1(64, 64) 
#         self.fusion_RS2 = Fusion_Attention_M2_3D(self.features * 2, self.features * 2)
#         # self.fusion_RSM2 = Fusion_Attention_M1(128, 128)
#         self.fusion_RS3 = Fusion_Attention_M2_3D(self.features * 4, self.features * 4)
#         # self.fusion_RSM3 = Fusion_Attention_M1(256, 256)
#         self.fusion_RS4 = Fusion_Attention_M2_3D(self.features * 8, self.features * 8)
#         # self.fusion_RSM4 = Fusion_Attention_M1(512, 512)
#         self.fusion_RS5 = Fusion_Attention_M2_3D(self.features * 16, self.features * 16)
#         # self.fusion_RSM5 = Fusion_Attention_M1(1024, 1024)
#         # self.RS_top = conv_block(ch_in=1024, ch_out=1024)

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv3d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=True,
#                         ),
#                     ),
#                     (name + "norm1", nn.BatchNorm3d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv3d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=True,
#                         ),
#                     ),
#                     (name + "norm2", nn.BatchNorm3d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )

#     def forward(self, x_R, x_S, x_M):
#         # main encoder
#         R_x1 = self.R_encoder1(x_R)
#         R_x2 = self.R_Maxpool(R_x1)
#         R_x2 = self.R_encoder2(R_x2)
#         R_x3 = self.R_Maxpool(R_x2)
#         R_x3 = self.R_encoder3(R_x3)
#         R_x4 = self.R_Maxpool(R_x3)
#         R_x4 = self.R_encoder4(R_x4)
#         R_x5 = self.R_Maxpool(R_x4)
#         R_x5 = self.R_bottleneck(R_x5)

#         # L encoder
#         S_x1 = self.S_encoder1(x_S)
#         S_x2 = self.S_Maxpool(S_x1)
#         S_x2 = self.S_encoder2(S_x2)
#         S_x3 = self.S_Maxpool(S_x2)
#         S_x3 = self.S_encoder3(S_x3)
#         S_x4 = self.S_Maxpool(S_x3)
#         S_x4 = self.S_encoder4(S_x4)
#         S_x5 = self.S_Maxpool(S_x4)
#         S_x5 = self.S_bottleneck(S_x5)

#         # H encoder
#         M_x1 = self.M_encoder1(x_M)
#         M_x2 = self.M_Maxpool(M_x1)
#         M_x2 = self.M_encoder2(M_x2)
#         M_x3 = self.M_Maxpool(M_x2)
#         M_x3 = self.M_encoder3(M_x3)
#         M_x4 = self.M_Maxpool(M_x3)
#         M_x4 = self.M_encoder4(M_x4)
#         M_x5 = self.M_Maxpool(M_x4)
#         M_x5 = self.M_bottleneck(M_x5)

#         # fusion
#         RS_x1 = self.fusion_RS1(R_x1, S_x1)
#         RS_x2 = self.fusion_RS2(R_x2, S_x2)
#         RS_x3 = self.fusion_RS3(R_x3, S_x3)
#         RS_x4 = self.fusion_RS4(R_x4, S_x4)
#         RS_x5 = self.fusion_RS5(R_x5, S_x5)

#         # RSM_x1 = self.fusion_RSM1(RS_x1, M_x1)
#         # RSM_x2 = self.fusion_RSM2(RS_x2, M_x2)
#         # RSM_x3 = self.fusion_RSM3(RS_x3, M_x3)
#         # RSM_x4 = self.fusion_RSM4(RS_x4, M_x4)
#         # RSM_x5 = self.fusion_RSM5(RS_x5, M_x5)

#         # RS_x5_ = self.RS_top(RS_x5)

#         # RS decoder
#         RS_d5 = self.RS_Up5(RS_x5)
#         RS_d5 = torch.cat((R_x4, S_x4, RS_d5, RS_x4, M_x4), dim=1)  # R_x4, S_x4, RS_d5, RS_x4, M_x4
#         RS_d5 = self.RS_Up_conv5(RS_d5)

#         RS_d4 = self.RS_Up4(RS_d5)
#         RS_d4 = torch.cat((R_x3, S_x3, RS_d4, RS_x3, M_x3), dim=1)  # R_x3, S_x3, RS_d4, RS_x3, M_x3
#         RS_d4 = self.RS_Up_conv4(RS_d4)

#         RS_d3 = self.RS_Up3(RS_d4)
#         RS_d3 = torch.cat((R_x2, S_x2, RS_d3, RS_x2, M_x2), dim=1)  # R_x2, S_x2, RS_d3, RS_x2, M_x2
#         RS_d3 = self.RS_Up_conv3(RS_d3)

#         RS_d2 = self.RS_Up2(RS_d3)
#         RS_d2 = torch.cat((R_x1, S_x1, RS_d2, RS_x1, M_x1), dim=1)  # R_x1, S_x1, RS_d2, RS_x1, M_x1
#         RS_d2 = self.RS_Up_conv2(RS_d2)

#         RS_d1 = self.RS_Conv_1x1(RS_d2)

#         # RSM decoder
#         M_d5 = self.M_Up5(M_x5)
#         RSM_d5 = torch.cat((M_x4, M_d5, RS_x4), dim=1)  # M_x4, M_d5, RS_x4
#         RSM_d5 = self.RSM_Up_conv5(RSM_d5)

#         RSM_d4 = self.RSM_Up4(RSM_d5)
#         RSM_d4 = torch.cat((M_x3, RSM_d4, RS_x3), dim=1) # M_x3, RSM_d4, RS_x3
#         RSM_d4 = self.RSM_Up_conv4(RSM_d4)

#         RSM_d3 = self.RSM_Up3(RSM_d4)
#         RSM_d3 = torch.cat((M_x2, RSM_d3, RS_x2), dim=1) # M_x2, RSM_d3, RS_x2
#         RSM_d3 = self.RSM_Up_conv3(RSM_d3)

#         RSM_d2 = self.RSM_Up2(RSM_d3)
#         RSM_d2 = torch.cat((M_x1, RSM_d2, RS_x1), dim=1) # M_x1, RSM_d2, RS_x1
#         RSM_d2 = self.RSM_Up_conv2(RSM_d2)

#         RSM_d1 = self.RSM_Conv_1x1(RSM_d2)

#         return RS_d1, RSM_d1

class MFwaveNet2_3D(nn.Module):
    def __init__(self, in_channels=1, output_channels=3, init_features=12):  # 我好像应该是12的，而不是24
        super(MFwaveNet2_3D, self).__init__()
        self.features = init_features

        self.R_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.R_encoder1 = MFwaveNet2_3D._block(in_channels, self.features, name="enc1")
        self.R_encoder2 = MFwaveNet2_3D._block(self.features, self.features * 2, name="enc2")
        self.R_encoder3 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc3")
        self.R_encoder4 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc4")
        self.R_bottleneck = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="bottleneck")

        self.RS_Up5 = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.RS_Up_conv5 = MFwaveNet2_3D._block(self.features * 16 + self.features * 8 * 4, self.features * 8, name="dec4") # self.features * 16 + 
        self.RS_Up4 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.RS_Up_conv4 = MFwaveNet2_3D._block(self.features * 8 + self.features * 4 * 4, self.features * 4, name="dec3") # self.features * 8 + 
        self.RS_Up3 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.RS_Up_conv3 = MFwaveNet2_3D._block(self.features * 4 + self.features * 2 * 4, self.features * 2, name="dec2") # self.features * 4 + 
        self.RS_Up2 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.RS_Up_conv2 = MFwaveNet2_3D._block(self.features * 2 + self.features * 4, self.features * 2, name="dec1") # self.features * 2 + 
        self.RS_Conv_1x1 = nn.Conv3d(self.features * 2, output_channels, kernel_size=1, stride=1, padding=0)

        # Select network
        self.S_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.S_encoder1 = MFwaveNet2_3D._block(in_channels, self.features, name="enc1")
        self.S_encoder2 = MFwaveNet2_3D._block(self.features, self.features * 2, name="enc2")
        self.S_encoder3 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc3")
        self.S_encoder4 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc4")
        self.S_bottleneck = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="bottleneck")

        # Mixup network
        self.M_Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.M_encoder1 = MFwaveNet2_3D._block(in_channels * 2, self.features * 2, name="enc1")
        self.M_encoder2 = MFwaveNet2_3D._block(self.features * 2, self.features * 4, name="enc2")
        self.M_encoder3 = MFwaveNet2_3D._block(self.features * 4, self.features * 8, name="enc3")
        self.M_encoder4 = MFwaveNet2_3D._block(self.features * 8, self.features * 16, name="enc4")
        self.M_bottleneck = MFwaveNet2_3D._block(self.features * 16, self.features * 32, name="bottleneck")

        self.M_Up5 = nn.ConvTranspose3d(self.features * 16 * 2, self.features * 16, kernel_size=2, stride=2)
        self.RSM_Up_conv5 = MFwaveNet2_3D._block(self.features * 16 * 2 + self.features * 8, self.features * 16, name="dec4") #  + self.features * 8
        self.RSM_Up4 = nn.ConvTranspose3d(self.features * 8 * 2, self.features * 8, kernel_size=2, stride=2)
        self.RSM_Up_conv4 = MFwaveNet2_3D._block(self.features * 8 * 2 + self.features * 4, self.features * 8, name="dec3") #  + self.features * 4
        self.RSM_Up3 = nn.ConvTranspose3d(self.features * 4 * 2, self.features * 4, kernel_size=2, stride=2)
        self.RSM_Up_conv3 = MFwaveNet2_3D._block(self.features * 4 * 2 + self.features * 2, self.features * 4, name="dec2") #  + self.features * 2
        self.RSM_Up2 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.RSM_Up_conv2 = MFwaveNet2_3D._block(self.features * 2 * 2 + self.features, self.features * 2, name="dec1") #  + self.features
        self.RSM_Conv_1x1 = nn.Conv3d(self.features * 2, output_channels, kernel_size=1, stride=1, padding=0)

        # fusion
        self.fusion_RS1 = Fusion_Attention_M2_3D(self.features, self.features) 
        # self.fusion_RSM1 = Fusion_Attention_M1(64, 64) 
        self.fusion_RS2 = Fusion_Attention_M2_3D(self.features * 2, self.features * 2)
        # self.fusion_RSM2 = Fusion_Attention_M1(128, 128)
        self.fusion_RS3 = Fusion_Attention_M2_3D(self.features * 4, self.features * 4)
        # self.fusion_RSM3 = Fusion_Attention_M1(256, 256)
        self.fusion_RS4 = Fusion_Attention_M2_3D(self.features * 8, self.features * 8)
        # self.fusion_RSM4 = Fusion_Attention_M1(512, 512)
        self.fusion_RS5 = Fusion_Attention_M2_3D(self.features * 16, self.features * 16)
        # self.fusion_RSM5 = Fusion_Attention_M1(1024, 1024)
        # self.RS_top = conv_block(ch_in=1024, ch_out=1024)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),
                    (name + "ReLU1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                    (name + "ReLU2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x_R, x_S, x_M):
        # main encoder
        R_x1 = self.R_encoder1(x_R)
        R_x2 = self.R_Maxpool(R_x1)
        R_x2 = self.R_encoder2(R_x2)
        R_x3 = self.R_Maxpool(R_x2)
        R_x3 = self.R_encoder3(R_x3)
        R_x4 = self.R_Maxpool(R_x3)
        R_x4 = self.R_encoder4(R_x4)
        R_x5 = self.R_Maxpool(R_x4)
        R_x5 = self.R_bottleneck(R_x5)

        # L encoder
        S_x1 = self.S_encoder1(x_S)
        S_x2 = self.S_Maxpool(S_x1)
        S_x2 = self.S_encoder2(S_x2)
        S_x3 = self.S_Maxpool(S_x2)
        S_x3 = self.S_encoder3(S_x3)
        S_x4 = self.S_Maxpool(S_x3)
        S_x4 = self.S_encoder4(S_x4)
        S_x5 = self.S_Maxpool(S_x4)
        S_x5 = self.S_bottleneck(S_x5)

        # H encoder
        M_x1 = self.M_encoder1(x_M)
        M_x2 = self.M_Maxpool(M_x1)
        M_x2 = self.M_encoder2(M_x2)
        M_x3 = self.M_Maxpool(M_x2)
        M_x3 = self.M_encoder3(M_x3)
        M_x4 = self.M_Maxpool(M_x3)
        M_x4 = self.M_encoder4(M_x4)
        M_x5 = self.M_Maxpool(M_x4)
        M_x5 = self.M_bottleneck(M_x5)

        # fusion
        RS_x1 = self.fusion_RS1(R_x1, S_x1)
        RS_x2 = self.fusion_RS2(R_x2, S_x2)
        RS_x3 = self.fusion_RS3(R_x3, S_x3)
        RS_x4 = self.fusion_RS4(R_x4, S_x4)
        RS_x5 = self.fusion_RS5(R_x5, S_x5)

        # RSM_x1 = self.fusion_RSM1(RS_x1, M_x1)
        # RSM_x2 = self.fusion_RSM2(RS_x2, M_x2)
        # RSM_x3 = self.fusion_RSM3(RS_x3, M_x3)
        # RSM_x4 = self.fusion_RSM4(RS_x4, M_x4)
        # RSM_x5 = self.fusion_RSM5(RS_x5, M_x5)

        # RS_x5_ = self.RS_top(RS_x5)

        # RS decoder
        RS_d5 = self.RS_Up5(RS_x5)
        RS_d5 = torch.cat((R_x4, S_x4, RS_d5, RS_x4, M_x4), dim=1)  # R_x4, S_x4, RS_d5, RS_x4, M_x4
        RS_d5 = self.RS_Up_conv5(RS_d5)

        RS_d4 = self.RS_Up4(RS_d5)
        RS_d4 = torch.cat((R_x3, S_x3, RS_d4, RS_x3, M_x3), dim=1)  # R_x3, S_x3, RS_d4, RS_x3, M_x3
        RS_d4 = self.RS_Up_conv4(RS_d4)

        RS_d3 = self.RS_Up3(RS_d4)
        RS_d3 = torch.cat((R_x2, S_x2, RS_d3, RS_x2, M_x2), dim=1)  # R_x2, S_x2, RS_d3, RS_x2, M_x2
        RS_d3 = self.RS_Up_conv3(RS_d3)

        RS_d2 = self.RS_Up2(RS_d3)
        RS_d2 = torch.cat((R_x1, S_x1, RS_d2, RS_x1, M_x1), dim=1)  # R_x1, S_x1, RS_d2, RS_x1, M_x1
        RS_d2 = self.RS_Up_conv2(RS_d2)

        RS_d1 = self.RS_Conv_1x1(RS_d2)

        # RSM decoder
        M_d5 = self.M_Up5(M_x5)
        RSM_d5 = torch.cat((M_x4, M_d5, RS_x4), dim=1)  # M_x4, M_d5, RS_x4
        RSM_d5 = self.RSM_Up_conv5(RSM_d5)

        RSM_d4 = self.RSM_Up4(RSM_d5)
        RSM_d4 = torch.cat((M_x3, RSM_d4, RS_x3), dim=1) # M_x3, RSM_d4, RS_x3
        RSM_d4 = self.RSM_Up_conv4(RSM_d4)

        RSM_d3 = self.RSM_Up3(RSM_d4)
        RSM_d3 = torch.cat((M_x2, RSM_d3, RS_x2), dim=1) # M_x2, RSM_d3, RS_x2
        RSM_d3 = self.RSM_Up_conv3(RSM_d3)

        RSM_d2 = self.RSM_Up2(RSM_d3)
        RSM_d2 = torch.cat((M_x1, RSM_d2, RS_x1), dim=1) # M_x1, RSM_d2, RS_x1
        RSM_d2 = self.RSM_Up_conv2(RSM_d2)

        RSM_d1 = self.RSM_Conv_1x1(RSM_d2)

        return RS_d1, RSM_d1

def mfwaveNet_3D(in_channels, num_classes):
    model = MFwaveNet2_3D(in_channels, num_classes)
    init_weights(model, 'kaiming')
    return model

if __name__ == '__main__':
    model = MFwaveNet2_3D(1, 2)
    model.eval()
    input1 = torch.rand(2, 1, 96, 96, 96)
    input2 = torch.rand(2, 1, 96, 96, 96)
    input3 = torch.rand(2, 2, 96, 96, 96)

    # gt = torch.rand(2,2,128,128)
    output1, output2, = model(input1, input2, input3)
    # output1 = output1.data.cpu().numpy()
    # loss = ((output1 - gt)**2).mean()
    # loss.backward()
    print(output1.shape)
    print(output2.shape)
