"""
简易 U-Net 卷积神经网络，适配 ImgDataset 数据集格式
支持数据增强，搭配 Qwen3-VL-8B 使用
该卷积网络可直接支持的图像大小必须 <= 256x256
"""
import torch.nn as nn
import torch
from modelscope.preprocessors.nlp.space import batch
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import tqdm as t
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from scripts.dataset.imgData import ImgDataset


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


def calculate_iou(pred, target, num_classes):
    iou_list = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = torch.sum(pred_cls & target_cls).float()
        union = torch.sum(pred_cls | target_cls).float()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append((intersection / union).item())
    return iou_list


def calculate_dice(pred, target, num_classes):
    dice_list = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = torch.sum(pred_cls & target_cls).float()
        dice = (2 * intersection) / (torch.sum(pred_cls).float() + torch.sum(target_cls).float())
        if torch.sum(pred_cls) + torch.sum(target_cls) == 0:
            dice_list.append(float('nan'))
        else:
            dice_list.append(dice.item())
    return dice_list


def test_model(model, test_loader, device, num_classes=2):
    model.eval()
    total_iou = []
    total_dice = []
    total_samples = 0

    with torch.no_grad():
        for i, (data, mask) in enumerate(test_loader):
            data = data.to(device).float()
            mask = mask.to(device).squeeze(1).long()

            output = model(data)
            pred = torch.argmax(output, dim=1)

            iou_list = calculate_iou(pred, mask, num_classes)
            dice_list = calculate_dice(pred, mask, num_classes)

            for j in range(len(iou_list)):
                if not torch.isnan(torch.tensor(iou_list[j])):
                    total_iou.append(iou_list[j])
                if not torch.isnan(torch.tensor(dice_list[j])):
                    total_dice.append(dice_list[j])

            total_samples += data.size(0)

    avg_iou = sum(total_iou) / len(total_iou) if total_iou else 0.0
    avg_dice = sum(total_dice) / len(total_dice) if total_dice else 0.0

    return avg_iou, avg_dice, total_samples


if __name__ == '__main__':
    dataset_path = project_root / 'dataset' / 'image' / 'train' / '50'
    nii_list = [str(dataset_path / f'P{i}.nii.gz') for i in range(1, 4)]
    mask_list = [str(dataset_path / 'tumor.nii.gz')] * 3

    # train = ImgDataset(nii_list, mask_list, use_augmentation=True, rotation_prob=0.5)
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=2).to(device)
    weight_file = project_root / 'model' / 'unet' / 'unet.pth'
    #
    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # model.train()
    #
    # print(f"Start training on {device}...")
    # print(f"Dataset size: {len(train)}")
    # batch_size1 = 4
    #
    # for epoch in range(20):
    #     with t.tqdm(total=len(train) / batch_size1) as pbar:
    #         pbar.set_description(f"epoch {epoch+1}")
    #         for i, (data, mask) in enumerate(DataLoader(train, batch_size=batch_size1, shuffle=True)):
    #             optimizer.zero_grad()
    #             data = data.to(device).float()
    #             mask = mask.to(device).squeeze(1).long()
    #             output = model(data)
    #             loss = criterion(output, mask)
    #             loss.backward()
    #             optimizer.step()
    #             pbar.update(1)
    #             pbar.set_postfix(loss=loss.item(), shape=output.shape, data_shape=data.shape, mask_shape=mask.shape)
    #
    # model_path = project_root / 'model' / 'unet' / 'unet.pth'
    # model_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(model.state_dict(), str(model_path))
    # print(f"Model saved to {model_path}")

    model.load_state_dict(torch.load(weight_file))
    model.eval()
    test_list = [str(dataset_path / f'P{i}.nii.gz') for i in range(4, 8)]
    mt_list = [str(dataset_path / 'tumor.nii.gz')] * 4
    test = ImgDataset(test_list, mt_list, use_augmentation=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    print("\n" + "="*50)
    print("Testing Model")
    print("="*50)

    avg_iou, avg_dice, num_samples = test_model(model, test_loader, device, num_classes=2)

    print(f"Test Samples: {num_samples}")
    print(f"Average IoU:  {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print("="*50)

    with torch.no_grad():
        test_idx = 10
        test_img = test[test_idx][0].unsqueeze(0).to(device)
        pre = model(test_img.float())
        pre = torch.argmax(pre, dim=1)
        pre = pre.detach().cpu()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(test[test_idx][0][0].numpy(), cmap='gray')
        plt.title("Input Image")
        plt.subplot(1, 3, 2)
        plt.imshow(pre[0].numpy(), cmap='gray')
        plt.title("Prediction")
        plt.subplot(1, 3, 3)
        plt.imshow(test[test_idx][1][0].numpy(), cmap='gray')
        plt.title("Ground Truth")
        plt.tight_layout()
        plt.show()