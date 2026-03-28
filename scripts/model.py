import torch

from llm.Qwen.model import QWen3Doctor
from img.sam3.model import Sam3Doctor
from img.imgProcess import *

class DoctorVL:
    def __init__(self, root):
        self.root = root
        self.toker = QWen3Doctor(root + "/Qwen/Qwen3-VL-2B-Instruct")
        self.eye = Sam3Doctor(root + "/sam3/sam3-8b5")

    def get_img(self, ip):
        masks = self.eye(ip, "people")
        return img_process(ip, masks)

    def get_answer(self, message):
        answer = self.toker(message)
        return answer
    
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
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            val_dataset: 验证数据集
            save_path: 模型保存路径
            warmup_steps: 学习率预热步数
            gradient_accumulation_steps: 梯度累积步数
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
            num_workers=4,
            pin_memory=True
        )
        
        # 定义优化器
        sam_optimizer = torch.optim.AdamW(
            self.eye.peft_model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        qwen_optimizer = torch.optim.AdamW(
            self.toker.peft_model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 定义学习率调度器
        sam_scheduler = torch.optim.lr_scheduler.LinearLR(
            sam_optimizer, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        
        qwen_scheduler = torch.optim.lr_scheduler.LinearLR(
            qwen_optimizer, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        
        # 定义损失函数
        sam_criterion = torch.nn.BCEWithLogitsLoss()
        qwen_criterion = torch.nn.CrossEntropyLoss()
        
        # 开始训练
        self.eye.peft_model.train()
        self.toker.peft_model.train()
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                images = batch['images']
                masks = batch['masks']
                messages = batch['messages']
                labels = batch['labels']
                
                # 1. 训练 Sam3 模型
                images = images.to(self.eye.device)
                masks = masks.to(self.eye.device)
                
                sam_inputs = self.eye.processor(
                    images=images, 
                    text=["people"] * len(images),
                    return_tensors="pt"
                ).to(self.eye.device)
                
                sam_outputs = self.eye.peft_model(**sam_inputs)
                sam_predicted_masks = self.eye.processor.post_process_instance_segmentation(
                    sam_outputs, 
                    threshold=0.5, 
                    mask_threshold=0.5,
                    target_sizes=[(256, 256)] * len(images)
                )
                
                sam_loss = 0.0
                for i, pred_mask in enumerate(sam_predicted_masks):
                    if 'masks' in pred_mask and len(pred_mask['masks']) > 0:
                        pred_mask_tensor = torch.from_numpy(pred_mask['masks'][0]).float().unsqueeze(0).to(self.eye.device)
                        sam_loss += sam_criterion(pred_mask_tensor, masks[i])
                    else:
                        sam_loss += sam_criterion(torch.zeros_like(masks[i]), masks[i])
                
                sam_loss = sam_loss / len(images)
                sam_loss = sam_loss / gradient_accumulation_steps
                sam_loss.backward(retain_graph=True)
                
                # 2. 获取 Sam3 的预测结果
                sam_masks = self.eye(images, "people")
                
                # 3. 将 Sam3 的输出作为 Qwen 的输入
                for i in range(len(messages)):
                    if i < len(sam_masks):
                        messages[i]['content'].append({
                            'type': 'image',
                            'mask': sam_masks[i]
                        })
                
                # 4. 训练 Qwen3 模型
                qwen_inputs = self.toker.get_inputs(messages)
                qwen_inputs = qwen_inputs.to(self.toker.model.device)
                labels = labels.to(self.toker.model.device)
                
                qwen_outputs = self.toker.peft_model(**qwen_inputs)
                qwen_logits = qwen_outputs.logits
                
                qwen_loss = qwen_criterion(qwen_logits.view(-1, qwen_logits.size(-1)), labels.view(-1))
                qwen_loss = qwen_loss / gradient_accumulation_steps
                qwen_loss.backward()
                
                # 5. 计算总损失
                total_loss = sam_loss + qwen_loss
                
                # 6. 参数更新
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.eye.peft_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.toker.peft_model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    sam_optimizer.step()
                    qwen_optimizer.step()
                    
                    # 更新学习率
                    sam_scheduler.step()
                    qwen_scheduler.step()
                    
                    # 清零梯度
                    sam_optimizer.zero_grad()
                    qwen_optimizer.zero_grad()
                    
                    global_step += 1
                
                # 累计损失
                epoch_loss += total_loss.item() * len(images)
                num_batches += len(images)
                
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
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        self.eye.peft_model.eval()
        self.toker.peft_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images']
                masks = batch['masks']
                messages = batch['messages']
                labels = batch['labels']
                
                # 1. 验证 Sam3 模型
                images = images.to(self.eye.device)
                masks = masks.to(self.eye.device)
                
                sam_inputs = self.eye.processor(
                    images=images, 
                    text=["people"] * len(images),
                    return_tensors="pt"
                ).to(self.eye.device)
                
                sam_outputs = self.eye.peft_model(**sam_inputs)
                sam_predicted_masks = self.eye.processor.post_process_instance_segmentation(
                    sam_outputs, 
                    threshold=0.5, 
                    mask_threshold=0.5,
                    target_sizes=[(256, 256)] * len(images)
                )
                
                sam_loss = 0.0
                for i, pred_mask in enumerate(sam_predicted_masks):
                    if 'masks' in pred_mask and len(pred_mask['masks']) > 0:
                        pred_mask_tensor = torch.from_numpy(pred_mask['masks'][0]).float().unsqueeze(0).to(self.eye.device)
                        sam_loss += sam_criterion(pred_mask_tensor, masks[i])
                    else:
                        sam_loss += sam_criterion(torch.zeros_like(masks[i]), masks[i])
                
                sam_loss = sam_loss / len(images)
                
                # 2. 获取 Sam3 的预测结果
                sam_masks = self.eye(images, "people")
                
                # 3. 将 Sam3 的输出作为 Qwen 的输入
                for i in range(len(messages)):
                    if i < len(sam_masks):
                        messages[i]['content'].append({
                            'type': 'image',
                            'mask': sam_masks[i]
                        })
                
                # 4. 验证 Qwen3 模型
                qwen_inputs = self.toker.get_inputs(messages)
                qwen_inputs = qwen_inputs.to(self.toker.model.device)
                labels = labels.to(self.toker.model.device)
                
                qwen_outputs = self.toker.peft_model(**qwen_inputs)
                qwen_logits = qwen_outputs.logits
                
                qwen_loss = qwen_criterion(qwen_logits.view(-1, qwen_logits.size(-1)), labels.view(-1))
                
                # 5. 计算总损失
                total_loss += (sam_loss + qwen_loss).item() * len(images)
                num_samples += len(images)
        
        self.eye.peft_model.train()
        self.toker.peft_model.train()
        return total_loss / num_samples if num_samples > 0 else 0.0

    def save_model(self, save_path: str):
        """
        保存联合模型
        
        Args:
            save_path: 模型保存路径
        """
        # 保存 Sam3 模型
        sam_save_path = save_path + "/sam3"
        self.eye.save_model(sam_save_path)
        
        # 保存 Qwen 模型
        qwen_save_path = save_path + "/qwen"
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
    msg = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': "图中有多少人？"
                }
            ]
        }
    ]
    doctor = DoctorVL("../model")
    print("sam3 finish")
    msg[0]['content'].append({
        'type': 'image',
        'url': 'img2.png',
    })
    print(doctor.get_answer(msg))