import matplotlib
import numpy as np
from PIL import Image
import nibabel as nib
import os


def open_img(img_path):
    img = Image.open(img_path).convert("RGB")
    return img


def img_process(img: Image.Image, mask):
    img = img.convert('RGBA')
    masks = 255 * mask.cpu().numpy().astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('rainbow').resampled(masks.shape[0])

    colors = [
        tuple(int(x * 255) for x in cmap(i)[:3]) for i in range(masks.shape[0])
    ]

    for mask1, color in zip(masks, colors):
        mask = Image.fromarray(mask1)
        overlay = Image.new("RGBA", img.size, color + (0,))
        alpha = mask.point(lambda i: int(i * 0.5))
        overlay.putalpha(alpha)
        img = Image.alpha_composite(img, overlay)

    return img


def process_nii_gz(nii_path):
    """处理nii.gz文件，输出可以作为sam3模型输入的图像tensor"""
    # 读取nii.gz文件
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 确保数据是三维的
    if len(data.shape) != 3:
        raise ValueError("nii.gz文件必须是三维的")
    
    print(data.shape)


if __name__ == "__main__":
    process_nii_gz("./dataset/image/train/50/P2.nii.gz")