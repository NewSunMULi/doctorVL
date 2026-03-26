import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image


class Sam3Doctor:
    def __init__(self, model_path):
        self.model = Sam3Model.from_pretrained(model_path)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, img:Image.Image, text=""):
        ip = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**ip)
        output = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5,
                                                                 target_sizes=ip.get("original_sizes").tolist())[0]

        return output['masks']


if __name__ == "__main__":
    model = Sam3Doctor("../../../model/sam3/sam3-8b5")
    img1 = Image.open("img1.png")
    model(img1, "people")
