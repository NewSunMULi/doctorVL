from typing import List

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer
import threading
import torch
from scripts.image.translation import NibabelImage


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
    img1 = NibabelImage("../data/50/50/P2.nii.gz").to_video()
    print(img1.shape)
    msg = [
        {
            'role':'user',
            'content':[
                {
                    'type': 'text',
                    'text': "请根据视频，判断异常部位所在坐标"
                }
            ]
        }
    ]
    for frame in img1[20:141, :, :]:
        msg[0]['content'].append({
                    'type':'image',
                    'image':frame,
                })
    model1 = QWen3Doctor('../model/Qwen/Qwen3-VL-2B-Instruct')
    steam = model1.get_text_stream()
    model1(msg, steam, 1024)
    for i in steam:
        print(i, end="", flush=True)

