"""
简易 U-Net 卷积神经网络 (无数据增强)，搭配 Qwen3-VL-8B使用
该卷积网络可直接支持的图像大小必须 <= 256x256
"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
from torchvision import transforms
import tqdm as t
import matplotlib.pyplot as plt


def double_conv(in_channels, out_channels):
    """
    双层卷积
    :param in_channels: 输入通道
    :param out_channels: 输出通道
    :return: 双层卷积对象
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    标准 U-Net 卷积神经网络
    """
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        # encoding
        self.en1 = double_conv(in_channels, 64)
        self.pl1 = nn.MaxPool2d(2, 2)
        self.en2 = double_conv(64, 128)
        self.pl2 = nn.MaxPool2d(2, 2)
        self.en3 = double_conv(128, 256)
        self.pl3 = nn.MaxPool2d(2, 2)
        self.en4 = double_conv(256, 512)
        self.pl4 = nn.MaxPool2d(2, 2)
        self.bottom = double_conv(512, 1024)
        self.drop1 = nn.Dropout2d(0.5)

        # decoding
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.de4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.de3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.de2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.de1 = double_conv(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.en1(x)
        u1 = self.pl1(e1)
        e2 = self.en2(u1)
        u2 = self.pl2(e2)
        e3 = self.en3(u2)
        u3 = self.pl3(e3)
        e4 = self.en4(u3)
        u4 = self.pl4(e4)
        bottom = self.bottom(u4)
        dp = self.drop1(bottom)

        p4 = self.up4(dp)
        d4 = self.de4(torch.cat([e4, p4], 1))
        p3 = self.up3(d4)
        d3 = self.de3(torch.cat([e3, p3], 1))
        p2 = self.up2(d3)
        d2 = self.de2(torch.cat([e2, p2], 1))
        p1 = self.up1(d2)
        d1 = self.de1(torch.cat([e1, p1], 1))
        output = self.output(d1)
        return output


class NibImage(Dataset):
    """
    医学数据集
    """
    def __init__(self, url_train, url_label='', size=256, is_train=True):
        super(NibImage, self).__init__()
        self.is_train = is_train
        self.data: torch.Tensor = self.__resize_img(self.__get_data(url_train).reshape((204, 1, 340, 340)), size)
        if is_train:
            self.label: torch.Tensor = self.__resize_img(self.__get_data(url_label).reshape((204, 340, 340)), size)
            self.tolong()

    @staticmethod
    def __get_data(url):
        object_nib = nib.load(url)
        np_data = object_nib.get_fdata()
        data1 = torch.from_numpy(np_data.T).float()
        return data1

    @staticmethod
    def __resize_img(data1, size):
        result = transforms.Compose([
            transforms.Resize((size, size)),
        ])
        return result(data1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.is_train:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def transform_to_array(self):
        if self.is_train:
            return self.data.numpy(), self.label.numpy()
        else:
            return self.data.numpy()

    def tolong(self):
        self.label = self.label.ceil().long()


if __name__ == '__main__':
    train = NibImage('../../dataset/image/train/50/P1.nii.gz', '../../dataset/image/train/50/tumor.nii.gz', 64)
    model = UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load('../../model/unet/unet.pth'))
    # 训练代码
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # model.train()
    # for epoch in range(5):
    #     with t.tqdm(total=len(train)//2) as pbar:
    #         pbar.set_description(f"epoch {epoch+1}")
    #         for i, (data, label) in enumerate(DataLoader(train, batch_size=2, shuffle=True)):
    #             optimizer.zero_grad()
    #             output = model(data)
    #             loss = criterion(output, label)
    #             loss.backward()
    #             optimizer.step()
    #             pbar.update(1)
    #             pbar.set_postfix(loss=loss.item())
    # torch.save(model.state_dict(), '../../model/unet/unet.pth')
    model.eval()
    test = NibImage('../../dataset/image/train/50/P2.nii.gz', size=64, is_train=False)
    with torch.no_grad():
        op1 = model(test[104:105])
        d = test.transform_to_array()[0]
        op = op1.argmax(dim=1)
        plt.subplot(131)
        plt.imshow(test[104][0], cmap='gray')
        plt.subplot(132)
        plt.imshow(op.numpy().squeeze(), cmap='gray')
        plt.subplot(133)
        l = train.transform_to_array()[1]
        plt.imshow(l[104], cmap='gray')
        plt.show()