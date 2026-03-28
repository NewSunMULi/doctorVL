import matplotlib
import numpy as np
from PIL import Image
import nibabel as nib


def open_img(img_path):
    """
    打开图像文件并转换为 RGB 格式
    
    Args:
        img_path: 图像文件路径
        
    Returns:
        转换为 RGB 格式的 PIL 图像对象
    """
    img = Image.open(img_path).convert("RGB")
    return img


def img_process(img: Image.Image, mask):
    """
    处理图像和掩码，添加彩色覆盖层
    
    Args:
        img: PIL 图像对象
        mask: 模型生成的掩码张量
        
    Returns:
        添加了彩色掩码覆盖层的 PIL 图像对象
    """
    # 将图像转换为 RGBA 格式以支持透明度
    img = img.convert('RGBA')
    # 将掩码转换为 numpy 数组并缩放到 0-255 范围
    masks = 255 * mask.cpu().numpy().astype(np.uint8)
    # 获取彩虹色映射并根据掩码数量重新采样
    cmap = matplotlib.colormaps.get_cmap('rainbow').resampled(masks.shape[0])

    # 为每个掩码生成不同的颜色
    colors = [
        tuple(int(x * 255) for x in cmap(i)[:3]) for i in range(masks.shape[0])
    ]

    # 为每个掩码添加彩色覆盖层
    for mask1, color in zip(masks, colors):
        # 将掩码转换为 PIL 图像
        mask = Image.fromarray(mask1)
        # 创建带有指定颜色的覆盖层
        overlay = Image.new("RGBA", img.size, color + (0,))
        # 创建透明度掩码，值越高透明度越低
        alpha = mask.point(lambda i: int(i * 0.5))
        # 设置覆盖层的透明度
        overlay.putalpha(alpha)
        # 将覆盖层与原始图像复合
        img = Image.alpha_composite(img, overlay)

    return img


def process_nii_gz(nii_path):
    """
    处理nii.gz文件,输出可以作为sam3模型输入的np.array
    
    Args:
        nii_path: NIfTI 格式文件路径
        
    Returns:
        三维 numpy 数组，表示处理后的医学影像数据
    """
    # 读取nii.gz文件
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 确保数据是三维的
    if len(data.shape) != 3:
        raise ValueError("nii.gz文件必须是三维的")
    
    return data


if __name__ == "__main__":
    process_nii_gz("./dataset/image/train/50/P2.nii.gz")
