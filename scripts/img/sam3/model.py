import torch
from transformers import Sam3Processor, Sam3Model, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
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
        ip = self.processor(images=img, text=text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**ip)

        # 后处理输出
        output = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # 置信度阈值
            mask_threshold=0.5,  # 掩码阈值
            target_sizes=ip.get("original_sizes").tolist()
        )[0]

        return output['masks']

    def train(self, dataset, batch_size=2, epochs=10, learning_rate=1e-4, lora_r=1, lora_alpha=1, text_prompts=None):
        """
        训练函数，使用 LoRA 微调 Sam3 模型
        
        Args:
            dataset: 数据集，包含图像和掩码
            batch_size: 批次大小，默认为 2（CPU 训练）
            epochs: 训练轮数
            learning_rate: 学习率
            lora_r: LoRA 秩
            lora_alpha: LoRA alpha 参数
            text_prompts: 文本提示列表，用于引导分割
            
        Returns:
            训练后的模型
        """
        # 配置 LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        # 应用 LoRA 到模型
        lora_model = get_peft_model(self.model, lora_config)
        lora_model.train()
        
        # 准备数据集
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(lora_model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # 默认文本提示
        if text_prompts is None:
            text_prompts = ["medical image segmentation"]
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, (images, masks) in enumerate(dataloader):
                # 移动数据到设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                print(images.shape)
                # 预处理输入
                inputs = self.processor(
                    images=images,
                    text=[text_prompts[i % len(text_prompts)] for i in range(images.shape[0])],
                    return_tensors="pt"
                ).to(self.device)
                print("数据处理完成")
                # 前向传播
                outputs = lora_model(**inputs)
                print('前向完成')
                
                # 计算损失
                loss = outputs.loss
                print('损失计算完成')
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                print('优化完成')
                epoch_loss += loss.item()
                print('批次完成')
                if step % 10 == 0:  # 每 10 步打印一次，适合 CPU 训练
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # 更新模型
        self.model = lora_model
        return self.model



if __name__ == "__main__":
    # 测试 Sam3Doctor
    model = Sam3Doctor("../../../model/sam3/sam3-8b5")
    
    # 测试训练
        # 创建简单的测试数据集
        # 这里使用相同的文件作为图像和掩码，仅用于测试
    test_nii_list = ["../../../dataset/image/train/50/P1.nii.gz"] * 4
    test_mask_list = ["../../../dataset/image/train/50/P1.nii.gz"] * 4
        
    dataset = ImgDataset(test_nii_list, test_mask_list)
    print(f"Dataset size: {len(dataset)}")
        
        # 运行训练
    print("Starting training...")
    trained_model = model.train(dataset, epochs=1, batch_size=1)
    print("Training completed!")
