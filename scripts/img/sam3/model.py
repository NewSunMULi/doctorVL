import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3Processor, Sam3Model, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.dataset.imgData import ImgDataset


class Sam3Doctor:
    """
    Sam3 模型包装类，用于医学影像分割
    
    该类封装了 Sam3 模型的加载、推理和训练相关功能
    """

    def __init__(self, model_path):
        """
        初始化 Sam3Doctor
        
        Args:
            model_path: 模型路径
        """
        # 使用半精度加载模型，减少内存消耗
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Sam3Model.from_pretrained(model_path, device_map="auto")
        self.processor = Sam3Processor.from_pretrained(model_path)

    def __call__(self, img: Image.Image, text=""):
        """
        推理函数
        
        Args:
            img: 输入图像
            text: 文本提示，用于引导分割
            
        Returns:
            预测的掩码
        """
        # 预处理输入
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 只返回第一个实例的掩码
        return outputs.pred_masks[:, 0:1, :, :]

    def load_lora(self, lora_path):
        """
        加载训练好的LoRA适配器
        
        Args:
            lora_path: LoRA适配器的路径
        """
        # 加载LoRA适配器
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.print_trainable_parameters()
        self.processor = Sam3Processor.from_pretrained(lora_path)
        print(f"LoRA适配器已从 {lora_path} 加载")

    def train(self, nii_list, mask_list, output_dir, batch_size=4, epochs=10, learning_rate=1e-4, lora_r=8, lora_alpha=16, text_prompt=""):
        """
        训练Sam3模型
        
        Args:
            nii_list: nii.gz 图像文件路径列表
            mask_list: nii.gz 掩码文件路径列表
            output_dir: 模型保存目录
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            lora_r: LoRA的秩
            lora_alpha: LoRA的alpha参数
            text_prompt: 文本提示，用于引导分割
        """        
        # 冻结模型原始参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # 包装模型
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 创建数据集和数据加载器
        dataset = ImgDataset(nii_list, mask_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 计算总步数并设置学习率调度器
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # 开始训练
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
                # 将数据移至设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 为批次中的每个元素创建文本提示
                text_prompts = [text_prompt] * images.shape[0]
                # 预处理输入
                inputs = self.processor(images=images, text=text_prompts, return_tensors="pt").to(self.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 处理输出形状：选择第一个实例并调整大小以匹配标签
                # 输出形状: [batch_size, num_instances, height, width]
                # 我们只关注第一个实例（肿瘤）
                pred_masks = outputs.pred_masks[:, 0:1, :, :]  # 选择第一个实例，保持通道维度
                
                # 调整大小以匹配标签形状
                pred_masks = F.interpolate(pred_masks, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # 应用sigmoid激活函数，确保输出在[0,1]范围内
                pred_masks = torch.sigmoid(pred_masks)
                
                # 计算损失
                loss = criterion(pred_masks, masks)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        
        # 保存模型
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        print(f"模型已保存到 {output_dir}")

    def test(self, nii_list, mask_list, threshold=0.5):
        """
        测试模型，计算分割准确率
        
        Args:
            nii_list: nii.gz 图像文件路径列表
            mask_list: nii.gz 掩码文件路径列表
            threshold: 预测阈值
            
        Returns:
            iou: 平均IoU值
            dice: 平均Dice系数
        """
        # 创建数据集和数据加载器
        dataset = ImgDataset(nii_list, mask_list)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 初始化评估指标
        total_iou = 0
        total_dice = 0
        num_samples = 0
        
        # 测试模式
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(dataloader):
                # 将数据移至设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 预处理输入
                inputs = self.processor(images=images, text="tumor", return_tensors="pt").to(self.device)
                
                # 前向传播
                outputs = self.model(**inputs)
                
                # 处理输出形状：选择第一个实例并调整大小以匹配标签
                pred_masks = outputs.pred_masks[:, 0:1, :, :]  # 选择第一个实例，保持通道维度
                
                # 调整大小以匹配标签形状
                pred_masks = F.interpolate(pred_masks, size=masks.shape[2:], mode='bilinear', align_corners=False)
                
                # 应用sigmoid激活函数并阈值化
                pred_masks = torch.sigmoid(pred_masks) > threshold
                pred_masks = pred_masks.float()
                
                # 计算IoU
                intersection = torch.sum(pred_masks * masks)
                union = torch.sum(pred_masks) + torch.sum(masks) - intersection
                iou = intersection / (union + 1e-8)
                
                # 计算Dice系数
                dice = (2 * intersection) / (torch.sum(pred_masks) + torch.sum(masks) + 1e-8)
                
                # 累加指标
                total_iou += iou.item()
                total_dice += dice.item()
                num_samples += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"Tested {batch_idx + 1}/{len(dataloader)} samples")
        
        # 计算平均值
        avg_iou = total_iou / num_samples
        avg_dice = total_dice / num_samples
        
        print(f"\nTest Results:")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average Dice: {avg_dice:.4f}")
        
        return avg_iou, avg_dice


if __name__ == "__main__":
    # 测试 Sam3Doctor
    model = Sam3Doctor("./model/sam3/sam3-8b5")
    
    # 测试训练
    test_nii_list = ["./dataset/image/train/50/T2.nii.gz"]
    test_mask_list = ["./dataset/image/train/50/tumor.nii.gz"]
        
    dataset1 = ImgDataset(test_nii_list, test_mask_list)
    
    # 测试加载LoRA适配器
    lora_path = './model/sam3_lora'
    try:
        model.load_lora(lora_path)
        print("LoRA加载测试成功")
    except Exception as e:
        print(f"LoRA加载测试失败: {e}")
        print("注意：这是正常的，因为可能还没有训练过LoRA模型")

    # 测试模型性能
    print("\nTesting model performance...")
    iou, dice = model.test(test_nii_list, test_mask_list)
    print(f"Test completed with IoU: {iou:.4f}, Dice: {dice:.4f}")

    # 可视化测试结果
    img = dataset1[5]
    op = model(img[0], text="tumor")
    print(f"Model output shape: {op.shape}")
    
    # 调整大小并阈值化
    pred_masks = F.interpolate(op, size=img[1].unsqueeze(0).shape[2:], mode='bilinear', align_corners=False)
    pred_masks = torch.sigmoid(pred_masks) > 0.5
    pred_masks = pred_masks.float()
    print(f"Processed mask shape: {pred_masks.shape}")

    # 计算单个样本的IoU和Dice
    masks = img[1].unsqueeze(0)
    intersection = torch.sum(pred_masks * masks)
    union = torch.sum(pred_masks) + torch.sum(masks) - intersection
    iou_single = intersection / (union + 1e-8)
    dice_single = (2 * intersection) / (torch.sum(pred_masks) + torch.sum(masks) + 1e-8)
    print(f"Single sample IoU: {iou_single.item():.4f}")
    print(f"Single sample Dice: {dice_single.item():.4f}")

    # 可视化结果
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img[0][0].numpy(), cmap="gray")
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_masks[0][0].numpy(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img[1][0].numpy(), cmap="gray")
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
        