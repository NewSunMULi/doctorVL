from typing import List

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer
import threading
import torch


class QWen3Doctor:
    def __init__(self, model_path_or_name: str):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path_or_name, dtype=torch.bfloat16)
        self.model = self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoProcessor.from_pretrained(model_path_or_name)
        self.tok = AutoTokenizer.from_pretrained(model_path_or_name)
        self.dir = model_path_or_name

    def __call__(self, messages: List[dict], text_stream: TextIteratorStreamer = None, new_token_num: int = 128):
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        args = dict(
            **inputs,
            streamer = text_stream,
            max_new_tokens=new_token_num
        )
        t = threading.Thread(target=self.model.generate, kwargs=args)
        t.start()

    def get_text_stream(self):
        return TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)

    def get_device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    a = QWen3Doctor("../model/Qwen3-2B")
    message1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请介绍你自己"
                },
            ],
        }
    ]

    stream1 = a.get_text_stream()
    a(message1, stream1, 2048)
    print("回复1")
    for s1 in stream1:
        print(s1, end="", flush=True)
