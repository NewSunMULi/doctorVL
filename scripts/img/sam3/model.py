import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Sam3Processor, Sam3Model
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Union


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, 
                 image_size: tuple = (256, 256), transform=None):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transform
        
        assert images.shape[0] == masks.shape[0], "Number of images and masks must match"
        
    def _numpy_to_image(self, array: np.ndarray) -> Image.Image:
        if array.dtype == np.uint8:
            pass
        elif array.dtype == np.float32 or array.dtype == np.float64:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
        
        if array.ndim == 2:
            image = Image.fromarray(array, mode='L').convert('RGB')
        elif array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array, mode='RGB')
        elif array.ndim == 3 and array.shape[2] == 4:
            image = Image.fromarray(array, mode='RGBA').convert('RGB')
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        
        return image
    
    def _process_mask(self, mask_array: np.ndarray) -> torch.Tensor:
        if mask_array.dtype == np.uint8:
            mask_array = mask_array.astype(np.float32) / 255.0
        elif mask_array.dtype != np.float32 and mask_array.dtype != np.float64:
            mask_array = mask_array.astype(np.float32)
        
        if mask_array.ndim == 2:
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        elif mask_array.ndim == 3:
            mask_tensor = torch.from_numpy(mask_array.squeeze(0))
        else:
            raise ValueError(f"Unsupported mask array shape: {mask_array.shape}")
        
        return mask_tensor
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Image.Image]]:
        image_array = self.images[idx]
        mask_array = self.masks[idx]
        
        image = self._numpy_to_image(image_array)
        original_size = image.size
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        
        mask = self._process_mask(mask_array)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        mask_resized = Image.fromarray((mask.squeeze(0).numpy() * 255).astype(np.uint8), mode='L')
        mask_resized = mask_resized.resize(self.image_size, Image.Resampling.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': mask_tensor,
            'original_size': original_size,
            'image_path': f"numpy_array_{idx}"
        }


class Sam3Doctor:
    def __init__(self, model_path):
        self.model = Sam3Model.from_pretrained(model_path)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.lora_config = None
        self.peft_model = None

    def __call__(self, img, text=""):
        ip = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**ip)
        output = self.processor.post_process_instance_segmentation(outputs, threshold=0.5, mask_threshold=0.5,
                                                                 target_sizes=ip.get("original_sizes").tolist())[0]

        return output['masks']

    def setup_lora(self, r: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1, 
                   target_modules: Optional[List[str]] = None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        self.lora_config = LoraConfig(
            task_type=TaskType.INSTANCE_SEGMENTATION_2D,
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
            print(f"Model saved to {save_path}")
        else:
            self.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            print(f"Base model saved to {save_path}")

    def load_lora_model(self, lora_path: str):
        from peft import PeftModel
        self.peft_model = PeftModel.from_pretrained(self.model, lora_path)
        self.peft_model.to(self.device)
        print(f"LoRA model loaded from {lora_path}")


if __name__ == "__main__":
    model = Sam3Doctor("./model/sam3/sam3-8b5")
    img1 = Image.open("./img/img1.png").convert("L")
    
    print("Sam3Doctor initialized successfully!")
    print("Available methods:")
    print("- __call__(img, text): Inference")
    print("- setup_lora(...): Configure LoRA")
    print("- save_model(path): Save model")
    print("- load_lora_model(path): Load LoRA model")