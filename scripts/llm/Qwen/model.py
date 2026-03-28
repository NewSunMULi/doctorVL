from typing import List, Optional
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import threading
import torch
import os


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
        self.lora_config = None
        self.peft_model = None
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
            with torch.cuda.amp.autocast():
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

    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, 
                   target_modules: Optional[List[str]] = None):
        """
        设置 LoRA 配置
        
        Args:
            r: LoRA 秩，控制 LoRA 的容量
            lora_alpha: LoRA 缩放因子
            lora_dropout: LoRA dropout，防止过拟合
            target_modules: 目标模块列表，默认包含注意力和前馈网络模块
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            inference_mode=False
        )
        
        self.peft_model = get_peft_model(self.model, self.lora_config)
        self.peft_model.print_trainable_parameters()

    def save_model(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        try:
            if self.peft_model is not None:
                # 保存 LoRA 模型
                self.peft_model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                self.tok.save_pretrained(save_path)
                print(f"Model saved to {save_path}")
            else:
                # 保存基础模型
                self.model.save_pretrained(save_path)
                self.processor.save_pretrained(save_path)
                self.tok.save_pretrained(save_path)
                print(f"Base model saved to {save_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_lora_model(self, lora_path: str):
        """
        加载 LoRA 模型
        
        Args:
            lora_path: LoRA 模型路径
        """
        try:
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(
                self.model, 
                lora_path,
                torch_dtype=torch.bfloat16
            )
            self.peft_model.to(self.device)
            print(f"LoRA model loaded from {lora_path}")
        except Exception as e:
            print(f"Error loading LoRA model: {e}")


if __name__ == "__main__":
    # 测试 QWen3Doctor
    model = QWen3Doctor("../../../model/Qwen/Qwen3-VL-2B-Instruct")
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
        print(i, end='')
