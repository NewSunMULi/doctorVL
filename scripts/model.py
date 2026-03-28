import torch
import torchvision.transforms as transforms
from dataset.dataset import DoctorDataset
from llm.Qwen.model import QWen3Doctor
from img.sam3.model import Sam3Doctor
from img.imgProcess import img_process


class DoctorVL:
    """
    医学影像处理与分析的联合模型类
    
    该类整合了 Sam3 图像分割模型和 Qwen3 大语言模型，用于医学影像的分析和问答
    """
    
    def __init__(self, root):
        """
        初始化 DoctorVL 模型
        
        Args:
            root: 模型文件的根目录路径
        """
        self.root = root
        # 初始化 Qwen3 大语言模型，用于处理文本问答
        self.toker = QWen3Doctor(root + "/Qwen/Qwen3-VL-2B-Instruct")
        # 初始化 Sam3 图像分割模型，用于处理医学影像
        self.eye = Sam3Doctor(root + "/sam3/sam3-8b5")
        # 预定义图像 resize 变换，用于将图像调整到固定大小
        self.resize = transforms.Resize((256, 256))

    def get_img(self, ip):
        """
        处理输入图像，返回分割后的图像
        
        Args:
            ip: 输入图像
            
        Returns:
            处理后的图像
        """
        # 使用 Sam3 模型分割图像中的人物
        masks = self.eye(ip, "people")
        # 对图像进行后处理
        return img_process(ip, masks)

    def get_answer(self, message):
        """
        获取模型对输入消息的回答
        
        Args:
            message: 输入的文本消息
            
        Returns:
            模型生成的回答
        """
        return self.toker(message)
    
    def _process_images(self, images, labels):
        """
        处理图像数据，准备 Sam3 模型的输入
        
        Args:
            images: 3D 医学影像数据，形状为 (batch, 204, 1, 340, 340)
            labels: 对应的分割标签，形状与 images 相同
            
        Returns:
            images_3channel: 处理后的 3 通道图像，形状为 (batch, 3, 256, 256)
            masks_resized: 处理后的标签，形状为 (batch, 1, 256, 256)
            batch_size: 批次大小
        """
        # 获取批次大小
        batch_size = images.size(0)
        
        # 检查输入数据形状是否正确
        if images.dim() != 5:
            raise ValueError(f"Expected 5D tensor (batch, 204, 1, 340, 340), got {images.shape}")
        
        # 取中间切片作为 2D 图像
        slice_idx = images.size(1) // 2
        images_2d = images[:, slice_idx, 0, :, :]  # 提取 2D 图像
        masks_2d = labels[:, slice_idx, 0, :, :]   # 提取 2D 标签
        
        # 调整大小到 256x256
        images_resized = self.resize(images_2d)
        masks_resized = self.resize(masks_2d)
        
        # 添加通道维度并复制到 3 通道以适应 SAM3
        images_3channel = images_resized.unsqueeze(1).repeat(1, 3, 1, 1)
        masks_resized = masks_resized.unsqueeze(1)
        
        # 移动到设备，使用非阻塞传输提高性能
        images_3channel = images_3channel.to(self.eye.device, non_blocking=True)
        masks_resized = masks_resized.to(self.eye.device, non_blocking=True)
        
        return images_3channel, masks_resized, batch_size
    
    def _calculate_sam_loss(self, sam_predicted_masks, masks_resized, sam_criterion, batch_size):
        """
        计算 Sam3 模型的损失
        
        Args:
            sam_predicted_masks: Sam3 模型的预测掩码
            masks_resized: 处理后的真实标签
            sam_criterion: 损失函数
            batch_size: 批次大小
            
        Returns:
            sam_loss: Sam3 模型的平均损失
        """
        sam_loss = 0.0
        # 遍历每个预测结果
        for i, pred_mask in enumerate(sam_predicted_masks):
            if 'masks' in pred_mask and len(pred_mask['masks']) > 0:
                # 将预测掩码转换为张量并移动到设备
                pred_mask_tensor = torch.from_numpy(pred_mask['masks'][0]).float().unsqueeze(0).to(self.eye.device, non_blocking=True)
                # 计算损失
                sam_loss += sam_criterion(pred_mask_tensor, masks_resized[i])
            else:
                # 如果没有预测结果，使用零掩码计算损失
                sam_loss += sam_criterion(torch.zeros_like(masks_resized[i]), masks_resized[i])
        
        # 返回平均损失
        return sam_loss / batch_size
    
    def _build_qwen_messages(self, messages, sam_predicted_masks, batch_size):
        """
        构建 Qwen3 模型的输入消息
        
        Args:
            messages: 文本消息列表
            sam_predicted_masks: Sam3 模型的预测掩码
            batch_size: 批次大小
            
        Returns:
            qwen_messages: Qwen3 模型的输入消息列表
        """
        qwen_messages = []
        # 为每个样本构建消息
        for i in range(batch_size):
            message_content = [{'type': 'text', 'text': messages[i]}]
            
            # 将 Sam3 的输出作为 Qwen 的输入
            if i < len(sam_predicted_masks):
                message_content.append({'type': 'image', 'mask': sam_predicted_masks[i]})
            
            qwen_messages.append({'role': 'user', 'content': message_content})
        
        return qwen_messages
    
    def train(self, train_dataset, num_epochs: int = 10, 
              batch_size: int = 4, learning_rate: float = 1e-4, 
              val_dataset = None, 
              save_path: str = None, 
              warmup_steps: int = 100, 
              gradient_accumulation_steps: int = 1):
        """
        联合训练 Sam3 和 Qwen3 模型
        
        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数，默认 10
            batch_size: 批次大小，默认 4
            learning_rate: 学习率，默认 1e-4
            val_dataset: 验证数据集，默认 None
            save_path: 模型保存路径，默认 None
            warmup_steps: 学习率预热步数，默认 100
            gradient_accumulation_steps: 梯度累积步数，默认 1
        """
        # 为两个模型设置 LoRA
        if self.eye.peft_model is None:
            self.eye.setup_lora()
        if self.toker.peft_model is None:
            self.toker.setup_lora()
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,  # 使用固定内存提高数据加载速度
        )
        
        # 定义优化器
        sam_optimizer = torch.optim.AdamW(
            self.eye.peft_model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01  # 权重衰减，防止过拟合
        )
        
        qwen_optimizer = torch.optim.AdamW(
            self.toker.peft_model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 定义学习率调度器，使用线性预热
        sam_scheduler = torch.optim.lr_scheduler.LinearLR(
            sam_optimizer, 
            start_factor=0.1,  # 初始学习率因子
            total_iters=warmup_steps  # 预热步数
        )
        
        qwen_scheduler = torch.optim.lr_scheduler.LinearLR(
            qwen_optimizer, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        
        # 定义损失函数
        sam_criterion = torch.nn.BCEWithLogitsLoss()  # Sam3 使用二元交叉熵损失
        qwen_criterion = torch.nn.CrossEntropyLoss()  # Qwen3 使用交叉熵损失
        
        # 混合精度训练
        scaler = torch.amp.GradScaler()
        
        # 设置模型为训练模式
        self.eye.peft_model.train()
        self.toker.peft_model.train()
        global_step = 0
        
        # 开始训练轮次
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # 遍历训练数据
            for batch_idx, batch in enumerate(train_loader):
                # 解包批次数据
                images, labels, messages = batch
                
                # 使用混合精度训练
                with torch.amp.autocast('cpu'):
                    # 1. 处理图像数据
                    images_3channel, masks_resized, batch_size = self._process_images(images, labels)
                    
                    # 2. 准备 Sam3 模型输入
                    sam_inputs = self.eye.processor(
                        images=images_3channel, 
                        text=["people"] * batch_size,  # 分割目标为人物
                        return_tensors="pt"
                    ).to(self.eye.device)
                    
                    # 3. 前向传播 Sam3 模型
                    sam_outputs = self.eye.peft_model(**sam_inputs)
                    # 后处理分割结果
                    sam_predicted_masks = self.eye.processor.post_process_instance_segmentation(
                        sam_outputs, 
                        threshold=0.5,  # 置信度阈值
                        mask_threshold=0.5,  # 掩码阈值
                        target_sizes=[(256, 256)] * batch_size  # 目标大小
                    )
                    
                    # 4. 计算 Sam3 损失
                    sam_loss = self._calculate_sam_loss(sam_predicted_masks, masks_resized, sam_criterion, batch_size)
                    
                    # 5. 构建 Qwen3 的输入消息
                    qwen_messages = self._build_qwen_messages(messages, sam_predicted_masks, batch_size)
                    
                    # 6. 准备 Qwen3 模型输入
                    qwen_inputs = self.toker.get_inputs(qwen_messages)
                    qwen_inputs = qwen_inputs.to(self.toker.model.device)
                    
                    # 7. 前向传播 Qwen3 模型
                    qwen_outputs = self.toker.peft_model(**qwen_inputs)
                    
                    # 8. 简化的 Qwen 损失计算（实际应用中需要根据具体任务调整）
                    qwen_loss = torch.tensor(0.0, device=self.toker.model.device)
                    
                    # 9. 计算总损失
                    total_loss = sam_loss + qwen_loss
                
                # 缩放损失以适应混合精度
                scaled_loss = scaler.scale(total_loss / gradient_accumulation_steps)
                scaled_loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪，防止梯度爆炸
                    scaler.unscale_(sam_optimizer)
                    scaler.unscale_(qwen_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.eye.peft_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.toker.peft_model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    scaler.step(sam_optimizer)
                    scaler.step(qwen_optimizer)
                    scaler.update()
                    
                    # 更新学习率
                    sam_scheduler.step()
                    qwen_scheduler.step()
                    
                    # 清零梯度
                    sam_optimizer.zero_grad(set_to_none=True)  # 使用 set_to_none 提高内存使用效率
                    qwen_optimizer.zero_grad(set_to_none=True)
                    
                    global_step += 1
                
                # 累计损失
                epoch_loss += total_loss.item() * batch_size
                num_batches += batch_size
                
                # 打印日志
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                          f"Total Loss: {total_loss.item():.4f}, Sam Loss: {sam_loss.item():.4f}, "
                          f"Qwen Loss: {qwen_loss.item():.4f}, Global Step: {global_step}")
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_epoch_loss:.4f}")
            
            # 验证
            if val_dataset:
                val_loss = self._validate(val_dataset, batch_size, sam_criterion, qwen_criterion)
                print(f"Validation Loss: {val_loss:.4f}")
            
            # 保存模型
            if save_path:
                self.save_model(save_path)
        
        print("Joint training completed!")

    def _validate(self, val_dataset, batch_size: int, sam_criterion, qwen_criterion) -> float:
        """
        验证联合模型
        
        Args:
            val_dataset: 验证数据集
            batch_size: 批次大小
            sam_criterion: Sam3 损失函数
            qwen_criterion: Qwen3 损失函数
            
        Returns:
            val_loss: 验证损失
        """
        # 根据 CPU 核心数调整工作线程数
        num_workers = min(4, torch.cuda.device_count() * 4)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True  # 保持工作线程活跃，提高性能
        )
        
        # 设置模型为评估模式
        self.eye.peft_model.eval()
        self.toker.peft_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        # 不计算梯度
        with torch.no_grad():
            for batch in val_loader:
                # 解包批次数据
                images, labels, messages = batch
                
                # 使用混合精度推理
                with torch.cuda.amp.autocast():
                    # 1. 处理图像数据
                    images_3channel, masks_resized, batch_size = self._process_images(images, labels)
                    
                    # 2. 准备 Sam3 模型输入
                    sam_inputs = self.eye.processor(
                        images=images_3channel, 
                        text=["people"] * batch_size,
                        return_tensors="pt"
                    ).to(self.eye.device)
                    
                    # 3. 前向传播 Sam3 模型
                    sam_outputs = self.eye.peft_model(**sam_inputs)
                    # 后处理分割结果
                    sam_predicted_masks = self.eye.processor.post_process_instance_segmentation(
                        sam_outputs, 
                        threshold=0.5, 
                        mask_threshold=0.5,
                        target_sizes=[(256, 256)] * batch_size
                    )
                    
                    # 4. 计算 Sam3 损失
                    sam_loss = self._calculate_sam_loss(sam_predicted_masks, masks_resized, sam_criterion, batch_size)
                    
                    # 5. 构建 Qwen3 的输入消息
                    qwen_messages = self._build_qwen_messages(messages, sam_predicted_masks, batch_size)
                    
                    # 6. 准备 Qwen3 模型输入
                    qwen_inputs = self.toker.get_inputs(qwen_messages)
                    qwen_inputs = qwen_inputs.to(self.toker.model.device)
                    
                    # 7. 简化的 Qwen 损失计算
                    qwen_loss = torch.tensor(0.0, device=self.toker.model.device)
                
                # 累计损失
                total_loss += (sam_loss + qwen_loss).item() * batch_size
                num_samples += batch_size
        
        # 恢复训练模式
        self.eye.peft_model.train()
        self.toker.peft_model.train()
        # 返回平均验证损失
        return total_loss / num_samples if num_samples > 0 else 0.0

    def save_model(self, save_path: str):
        """
        保存联合模型
        
        Args:
            save_path: 模型保存路径
        """
        import os
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存 Sam3 模型
        sam_save_path = os.path.join(save_path, "sam3")
        self.eye.save_model(sam_save_path)
        
        # 保存 Qwen 模型
        qwen_save_path = os.path.join(save_path, "qwen")
        self.toker.save_model(qwen_save_path)
        
        print(f"Joint model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        加载联合模型
        
        Args:
            load_path: 模型加载路径
        """
        # 加载 Sam3 模型
        sam_load_path = load_path + "/sam3"
        self.eye.load_lora_model(sam_load_path)
        
        # 加载 Qwen 模型
        qwen_load_path = load_path + "/qwen"
        self.toker.load_lora_model(qwen_load_path)
        
        print(f"Joint model loaded from {load_path}")


if __name__ == "__main__":
    # 创建数据集实例
    dataset = DoctorDataset(root="../", data_path="dataset/llm/qa_pairs.json")
    # 创建模型实例
    model = DoctorVL("../model")
    # 开始训练
    model.train(train_dataset=dataset)
