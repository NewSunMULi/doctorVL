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
