"""
视觉分割-连接-LLM 整体流程
UNet 分割 -> ConnectModel 特征提取与关键帧选择 -> Qwen3-VL 分析
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from scripts.img.unet.unet import UNet
from scripts.connect.model import FeatureExtraction, TemporalTransformer, KeyChoice, ConnectModel
from scripts.llm.Qwen.model import QWen3Doctor


class SegmentationConnectLLM:
    """
    整体流程：UNet分割 -> ConnectModel关键帧选择 -> Qwen3-VL分析
    """

    def __init__(
        self,
        unet_path: str | Path = None,
        qwen_path: str | Path = None,
        feature_dim: int = 512,
        num_key_frames: int = 8,
        threshold_ratio: float = 0.1,
        device: str = None
    ):
        """
        初始化各模块

        Args:
            unet_path: UNet模型权重路径
            qwen_path: Qwen3-VL模型路径
            feature_dim: 特征维度
            num_key_frames: 关键帧数量
            threshold_ratio: 关键帧选择阈值比例
            device: 设备类型
        """
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.unet = UNet(in_channels=1, out_channels=2).to(self.device)
        if unet_path and Path(unet_path).exists():
            self.unet.load_state_dict(torch.load(unet_path, map_location=self.device))
            print(f"UNet权重已加载: {unet_path}")
        self.unet.eval()

        self.connect_model = ConnectModel(
            in_channels=1,
            feature_dim=feature_dim,
            num_key_frames=num_key_frames,
            threshold_ratio=threshold_ratio
        ).to(self.device)
        self.connect_model.eval()

        self.qwen = None
        if qwen_path and Path(qwen_path).exists():
            print(f"正在加载Qwen3-VL模型: {qwen_path}")
            self.qwen = QWen3Doctor(qwen_path)

        self.num_key_frames = num_key_frames
        self.feature_dim = feature_dim

    def segment(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用UNet进行分割

        Args:
            images: (N, 1, H, W) 输入图像

        Returns:
            masks: (N, 1, H, W) 二值分割掩码
        """
        with torch.no_grad():
            images = images.to(self.device).float()
            output = self.unet(images)
            pred = torch.argmax(output, dim=1, keepdim=True)
            masks = (pred == 1).float()
        return masks

    def extract_key_frames(self, images: torch.Tensor, masks: torch.Tensor = None):
        """
        使用ConnectModel提取关键帧

        Args:
            images: (N, 1, H, W) 输入图像
            masks: (N, 1, H, W) 分割掩码，如果为None则自动分割

        Returns:
            key_frames: 关键帧图像列表
            key_indices: 关键帧索引
            summary: 文字摘要
            features: 关键帧特征
        """
        if masks is None:
            masks = self.segment(images)

        with torch.no_grad():
            images = images.to(self.device).float()
            key_features, key_indices, summary = self.connect_model(images, masks)

        key_frames = []
        for idx in key_indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            key_frames.append(images[idx])

        return key_frames, key_indices, summary, key_features

    def build_messages(self, key_frames, summary, question: str = None):
        """
        构建Qwen3-VL输入消息

        Args:
            key_frames: 关键帧图像列表
            summary: 时序摘要
            question: 用户问题

        Returns:
            messages: 格式化的消息列表
        """
        if question is None:
            question = "请分析以下超声图像序列中肿瘤的变化趋势，并给出诊断建议。"

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的医学影像分析助手，擅长分析超声图像序列中的病变变化趋势。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    * [{"type": "image", "image": frame.cpu()} for frame in key_frames],
                    {"type": "text", "text": f"时序分析摘要：{summary}"}
                ]
            }
        ]

        return messages

    def __call__(
        self,
        images: torch.Tensor,
        question: str = None,
        use_stream: bool = True,
        new_token_num: int = 512
    ):
        """
        整体流程推理

        Args:
            images: (N, 1, H, W) 输入图像序列
            question: 用户问题
            use_stream: 是否使用流式输出
            new_token_num: 最大生成token数

        Returns:
            生成的文本（如果不使用流式）或 0（使用流式）
        """
        masks = self.segment(images)
        key_frames, key_indices, summary, _ = self.extract_key_frames(images, masks)

        messages = self.build_messages(key_frames, summary, question)

        if self.qwen is None:
            print("Qwen模型未加载，无法生成分析")
            return {
                "key_indices": key_indices,
                "summary": summary,
                "num_key_frames": len(key_frames)
            }

        if use_stream:
            text_stream = self.qwen.get_text_stream()
            self.qwen(messages, text_stream, new_token_num)
            return text_stream
        else:
            return self.qwen(messages, None, new_token_num)

    def analyze(self, images: torch.Tensor, question: str = None):
        """
        非流式分析，直接返回结果

        Args:
            images: (N, 1, H, W) 输入图像序列
            question: 用户问题

        Returns:
            analysis: 生成的文本分析
        """
        return self(images, question=question, use_stream=False)


def test_integration():
    """集成测试"""
    print("=" * 60)
    print("视觉分割-连接-LLM 集成测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet_path = project_root / "model" / "unet" / "unet.pth"
    qwen_path = project_root / "model" / "Qwen" / "Qwen3-VL-2B-Instruct"

    print("\n[1] 初始化整体模型...")
    model = SegmentationConnectLLM(
        unet_path=unet_path if unet_path.exists() else None,
        qwen_path=qwen_path if qwen_path.exists() else None,
        feature_dim=512,
        num_key_frames=8,
        threshold_ratio=0.1
    )
    print(f"    使用设备: {device}")

    print("\n[2] 加载测试数据...")
    from scripts.dataset.imgData import ImgDataset
    dataset_path = project_root / "dataset" / "image" / "train" / "50"
    nii_list = [str(dataset_path / f"P{i}.nii.gz") for i in range(1, 2)]
    mask_list = [str(dataset_path / "tumor.nii.gz")]

    dataset = ImgDataset(nii_list, mask_list, use_augmentation=False)
    print(f"    数据集大小: {len(dataset)}")

    images = []
    masks = []
    for i in range(min(16, len(dataset))):
        img, msk = dataset[i]
        images.append(img)
        masks.append(msk)

    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    print(f"    测试图像形状: {images.shape}")
    print(f"    测试掩码形状: {masks.shape}")

    print("\n[3] 测试分割模块 (UNet)...")
    with torch.no_grad():
        pred_masks = model.segment(images)
        pred_class = (pred_masks > 0.5).float()
        intersection = (pred_class * masks).sum()
        union = pred_class.sum() + masks.sum() - intersection
        iou = intersection / (union + 1e-8)
    print(f"    预测掩码形状: {pred_masks.shape}")
    print(f"    分割IoU: {iou.item():.4f}")

    print("\n[4] 测试关键帧提取 (ConnectModel)...")
    key_frames, key_indices, summary, key_features = model.extract_key_frames(images, pred_masks)
    print(f"    选出关键帧数量: {len(key_frames)}")
    print(f"    关键帧索引: {[idx.item() for idx in key_indices]}")
    print(f"    关键帧特征形状: {key_features.shape}")
    print(f"    摘要: {summary}")

    print("\n[5] 测试Qwen3-VL输入构建...")
    messages = model.build_messages(key_frames, summary)
    print(f"    构建消息数: {len(messages)}")
    print(f"    用户消息内容项数: {len(messages[1]['content'])}")

    if model.qwen is not None:
        print("\n[6] 测试Qwen3-VL推理...")
        print("    (跳过实际推理，仅测试流程)")
        print("    如需完整测试，请运行 QWen3Doctor 的交互测试")
        op = model.qwen(messages, None, 512)
        print(op)
    else:
        print("\n[6] Qwen模型未加载，跳过LLM推理测试")

    print("\n" + "=" * 60)
    print("集成测试完成!")
    print("=" * 60)

    return model, {
        "key_frames": key_frames,
        "key_indices": key_indices,
        "summary": summary,
        "iou": iou.item()
    }


if __name__ == "__main__":
    test_integration()