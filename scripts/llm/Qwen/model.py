import sys
from pathlib import Path
# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
import threading
import torch
from torch.utils.data import DataLoader
from scripts.dataset.qaData import CyreneDataset


class QWen3Doctor:
    """
    Qwen3 模型包装类，用于处理医学影像相关的文本生成任务
    
    该类封装了 Qwen3-VL 模型的加载、推理和训练相关功能
    """
    
    def __init__(self, model_path_or_name: str | Path):
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
                    output = self.model.generate(**inputs, max_new_tokens=new_token_num)
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
        
        适配 qaData.py 中的 CyreneDataset 数据集格式，支持两种模式：
        1. chat 模式: 多轮对话格式 [{'role': 'system/user/assistant', 'content': ...}, ...]
        2. instruction 模式: 指令-响应格式 {'instruction': ..., 'response': ...}
        
        Args:
            dataset: 训练数据集 (CyreneDataset)，支持 chat 或 instruction 模式
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
        
        # 获取数据集模式（从第一个样本判断）
        if len(dataset) > 0:
            first_sample = dataset[0]
            if isinstance(first_sample, dict) and 'instruction' in first_sample:
                data_mode = 'instruction'
            elif isinstance(first_sample, list) and len(first_sample) > 0 and 'role' in first_sample[0]:
                data_mode = 'chat'
            else:
                data_mode = 'chat'
            print(f"Detected data mode: {data_mode}")
        else:
            data_mode = 'chat'
        
        # 训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                # 根据数据模式处理输入
                if data_mode == 'instruction':
                    # instruction 模式: {'instruction': ..., 'response': ...}
                    # 将 instruction + response 组合成对话格式
                    messages = []
                    for i in range(len(batch_data['instruction'])):
                        instruction = batch_data['instruction'][i]
                        response = batch_data['response'][i]
                        msg = [
                            {'role': 'user', 'content': instruction},
                            {'role': 'assistant', 'content': response}
                        ]
                        messages.append(msg)
                else:
                    # chat 模式: [{'role': ..., 'content': ...}, ...]
                    # 转换为列表形式
                    messages = []
                    type_num = len(batch_data)
                    num = len(batch_data[0]['role'])
                    for i in range(num):
                        msg_list = []
                        for j in range(type_num):
                            msg_list.append({
                                'role': batch_data[j]['role'][i],
                                'content': batch_data[j]['content'][i]
                            })
                        messages.append(msg_list)
                # 使用 processor 处理对话格式
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"padding": True}
                )
                
                inputs = inputs.to(self.device)
                labels = inputs.input_ids.clone()
                
                # 前向传播（Qwen3-VL 使用 causal language modeling 损失）
                outputs = lora_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels
                )
                
                # 计算损失
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                
                if step % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Step {step}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # 保存模型
        save_path = project_root / "model" / "adapter" / "Qwen" / "Qwen3-VL-2B-Instruct"
        save_path.mkdir(parents=True, exist_ok=True)
        lora_model.save_pretrained(str(save_path))
        print(f"Model saved to {save_path}")
        
        return lora_model


    def load_adapter(self, adapter_path: str | Path, merge_weights: bool = True, test_data=None):
        """
        加载已保存的 LoRA 适配器
        
        Args:
            adapter_path: 适配器路径
            merge_weights: 是否合并 LoRA 权重到基础模型（推荐 True）
            
        Returns:
            加载了适配器的模型
        """
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        self.model = PeftModel.from_pretrained(
            self.model,
            str(adapter_path),
            device_map="auto"
        )
        
        if merge_weights:
            print("Merging LoRA weights with base model...")
            self.model = self.model.merge_and_unload()
            print("Merge completed")

    def interactive_test(self, system=None):
        """
        交互式测试模式，允许用户手动输入进行对话测试
        支持多轮对话，输入 'quit' 或 'exit' 退出
        """
        print("\n" + "="*50)
        print("Interactive Testing Mode")
        print("="*50)
        print("Enter your message, type 'quit' or 'exit' to end the session")
        print("-"*50)

        conversation_history = []
        self.model.eval()

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Ending session...")
                break

            if not user_input:
                continue

            if system:
                conversation_history.append(system)

            conversation_history.append({
                'role': 'user',
                'content': user_input
            })

            print("\nAssistant: ", end='', flush=True)
            text_stream = self.get_text_stream()
            self(conversation_history, text_stream, new_token_num=512)

            response_text = ""
            for char in text_stream:
                print(char, end='', flush=True)
                response_text += char
            print()

            conversation_history.append({
                'role': 'assistant',
                'content': response_text
            })

        print("\n" + "="*50)
        print("Session ended")
        print("="*50)


if __name__ == "__main__":
    qwen_url = project_root / "model" / "Qwen" / "Qwen3-VL-2B-Instruct"
    # dataset_url = project_root / "dataset" / "llm" / "Cyrene.jsonl"
    adapter_url = project_root / "model" / "adapter" / "Qwen" / "Qwen3-VL-2B-Instruct"
    print("Choose mode:")
    print("1. Train model")
    print("2. Load adapter and test")
    print("3. Interactive chat (without adapter)")

    # dataset = CyreneDataset(dataset_url, "chat")

    test_sample = {'role': 'system', 'content': ""}
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    model = QWen3Doctor(qwen_url)

    if choice == "1":

        model.train(dataset, batch_size=2, epochs=15, lora_r=8, lora_alpha=16)
        
    elif choice == "2":
        model.load_adapter(adapter_url)
        model.interactive_test(system=test_sample)
        
    elif choice == "3":
        model.interactive_test()
        
    else:
        print("Invalid choice")
