from typing import List, Optional
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import threading
import torch


class QWen3Doctor:
    # noinspection PyNoneFunctionAssignment
    def __init__(self, model_path_or_name: str):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path_or_name, dtype=torch.bfloat16)
        self.model = self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained(model_path_or_name)
        self.tok = AutoTokenizer.from_pretrained(model_path_or_name)
        self.dir = model_path_or_name
        self.lora_config = None
        self.peft_model = None

    def __call__(self, messages: List[dict], text_stream: TextIteratorStreamer = None, new_token_num: int = 128):
        inputs = self.get_inputs(messages)
        inputs = inputs.to(self.model.device)
        if text_stream is not None:
            args = dict(
                **inputs,
                streamer=text_stream,
                max_new_tokens=new_token_num
            )
            t = threading.Thread(target=self.model.generate, kwargs=args)
            t.start()
            return 0
        else:
            output = self.model.generate(**inputs)
            return self.processor.batch_decode(output, skip_special_tokens=True)

    def get_text_stream(self):
        return TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

    def get_device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_inputs(self, messages):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        return inputs

    def get_input_img(self, img):
        inputs = self.processor(images=img, text="请描述变化过程", return_tensors="pt")
        return inputs

    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, 
                   target_modules: Optional[List[str]] = None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            inference_mode=False
        )
        
        self.peft_model = get_peft_model(self.model, self.lora_config)
        self.peft_model.print_trainable_parameters()

    def save_model(self, save_path: str):
        if self.peft_model is not None:
            self.peft_model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            self.tok.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        else:
            self.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            self.tok.save_pretrained(save_path)
            print(f"Base model saved to {save_path}")

    def load_lora_model(self, lora_path: str):
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(self.model, lora_path)
        self.peft_model.to(self.model.device)
        print(f"LoRA model loaded from {lora_path}")


if __name__ == "__main__":
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