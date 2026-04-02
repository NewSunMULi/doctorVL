import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    3D 双卷积块，包含两个卷积层和批量归一化
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """
    下采样模块，包含最大池化和双卷积块
    """
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    上采样模块，包含上采样和双卷积块
    """
    def __init__(self, in_channels, out_channels, trilinear=True):
        super(Up3D, self).__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 调整 x1 的大小以匹配 x2
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2, diff_z // 2, diff_z - diff_z // 2])
        # 拼接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    输出卷积层，用于生成最终的分割掩码
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class nnUNet3D(nn.Module):
    """
    nnU-Net 3D 模型
    """
    def __init__(self, in_channels=1, out_channels=2):
        super(nnUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 编码器
        self.inc = DoubleConv3D(in_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.down4 = Down3D(512, 1024)

        # 解码器
        self.up1 = Up3D(1024 + 512, 512)
        self.up2 = Up3D(512 + 256, 256)
        self.up3 = Up3D(256 + 128, 128)
        self.up4 = Up3D(128 + 64, 64)

        # 输出层
        self.outc = OutConv3D(64, out_channels)

    def forward(self, x):
        # 编码器前向传播
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器前向传播，包含跳跃连接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 输出分割掩码
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # 测试 nnUNet3D 模型
    model = nnUNet3D(in_channels=1, out_channels=2)
    # 创建一个随机输入 [batch_size, channels, depth, height, width]
    input = torch.randn(1, 1, 32, 64, 64)
    output = model(input)
    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")
