from typing import List
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import threading
import torch
from torch.utils.data import DataLoader


class QWen3Doctor:
    """
    Qwen3 模型包装类，用于处理医学影像相关的文本生成任务
    
    该类封装了 Qwen3-VL 模型的加载、推理和训练相关功能
    """
    
    def __init__(self, model_path_or_name: str):
        """
        初始化 QWen3Doctor
        
        Args:
            model_path_or_name: 模型路径或名称
        """
        # 使用半精度加载模型，减少内存消耗
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path_or_name, 
            dtype=torch.bfloat16,
            device_map="auto"  # 自动分配设备
        )
        self.processor = AutoProcessor.from_pretrained(model_path_or_name)
        self.tok = AutoTokenizer.from_pretrained(model_path_or_name)
        self.dir = model_path_or_name
        self.device = self.model.device

    def __call__(self, messages: List[dict], text_stream: TextIteratorStreamer = None, new_token_num: int = 128):
        """
        推理函数
        
        Args:
            messages: 输入消息列表，包含角色和内容
            text_stream: 文本流对象，用于流式输出
            new_token_num: 生成的新 token 数量，默认 128
            
        Returns:
            生成的文本列表（如果不使用文本流）或 0（如果使用文本流）
        """
        try:
            # 获取模型输入
            inputs = self.get_inputs(messages)
            inputs = inputs.to(self.device, non_blocking=True)
            
            # 使用混合精度推理
            if text_stream is not None:
                    # 使用文本流进行流式输出
                    args = dict(
                        **inputs,
                        streamer=text_stream,
                        max_new_tokens=new_token_num
                    )
                    # 创建线程进行生成
                    t = threading.Thread(target=self.model.generate, kwargs=args)
                    t.start()
                    return 0
            else:
                    # 直接生成文本
                    output = self.model.generate(**inputs)
                    # 解码输出并跳过特殊 token
                    return self.processor.batch_decode(output, skip_special_tokens=True)
        except Exception as e:
            print(f"Error in inference: {e}")
            return [] if text_stream is None else 0

    def get_text_stream(self):
        """
        获取文本流对象
        
        Returns:
            TextIteratorStreamer 对象，用于流式输出生成的文本
        """
        return TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

    def get_device(self):
        """
        获取模型所在的设备
        
        Returns:
            设备名称字符串
        """
        return str(self.device)

    def get_inputs(self, messages):
        """
        获取模型输入
        
        Args:
            messages: 输入消息列表
            
        Returns:
            模型输入字典
        """
        try:
            # 应用聊天模板并进行分词
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # 添加生成提示
                return_dict=True,
                return_tensors="pt"
            )
            return inputs
        except Exception as e:
            print(f"Error getting inputs: {e}")
            return {}

    def get_input_img(self, img):
        """
        获取图像输入
        
        Args:
            img: 输入图像
            
        Returns:
            模型输入字典
        """
        try:
            inputs = self.processor(images=img, text="请描述变化过程", return_tensors="pt")
            return inputs
        except Exception as e:
            print(f"Error getting image inputs: {e}")
            return {}

    def train(self, dataset, batch_size=8, epochs=10, learning_rate=1e-4, lora_r=8, lora_alpha=16):
        """
        训练函数，使用 LoRA 微调 QWen3 多模态模型
        
        Args:
            dataset: 训练数据集，包含文本和图像等信息
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            lora_r: LoRA 秩
            lora_alpha: LoRA alpha 参数
            
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
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, (inputs, labels) in enumerate(dataloader):
                # 移动数据到设备
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = lora_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
                
                # 计算损失
                loss = outputs.loss
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/len(dataloader):.4f}")
        
        # 保存模型
        lora_model.save_pretrained("./model/adapter/Qwen/Qwen3-VL-2B-Instruct")


if __name__ == "__main__":
    # 测试 QWen3Doctor
    model = QWen3Doctor("./model/Qwen/Qwen3-VL-2B-Instruct")
    text_stream = model.get_text_stream()
    msg = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': "请介绍你自己"
                }
            ]
        }
    ]
    model(msg, text_stream)
    for i in text_stream:
        print(i, end='', flush=True)
