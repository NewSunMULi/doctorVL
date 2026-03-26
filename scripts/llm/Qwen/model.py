from typing import List
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer
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

