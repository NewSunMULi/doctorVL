import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image


class Sam3Doctor:
    def __init__(self, model_path):
        self.model = Sam3Model.from_pretrained(model_path)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, img, text=""):
        ip = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**ip)
        output = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5,
                                                                 target_sizes=ip.get("original_sizes").tolist())[0]

        plt.imshow(self.__img_process(img, output['masks']))
        plt.show()

    def __img_process(self, img:Image.Image, mask):
        img = img.convert('RGBA')
        masks = 255 * mask.cpu().numpy().astype(np.uint8)
        cmap = matplotlib.colormaps.get_cmap('rainbow').resampled(masks.shape[0])

        colors = [
            tuple(int(x * 255) for x in cmap(i)[:3]) for i in range(masks.shape[0])
        ]

        for mask1, color in zip(masks, colors):
            mask = Image.fromarray(mask1)
            overlay = Image.new("RGBA", img.size, color + (0,))
            alpha = mask.point(lambda i: int (i * 0.5))
            overlay.putalpha(alpha)
            img= Image.alpha_composite(img, overlay)

        return img


if __name__ == "__main__":
    model = Sam3Doctor("../../../model/sam3/sam3-8b5")
    img1 = Image.open("img1.png").convert('RGB')
    model(img1, "people")
