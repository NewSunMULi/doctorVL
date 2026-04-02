from dataset.dataset import DoctorDataset
from llm.Qwen.model import QWen3Doctor
from img.sam3.model import Sam3Doctor
from PIL import Image


class DoctorVL:
    """
    医学影像处理与分析的联合模型类
    
    该类整合了 Sam3 图像分割模型和 Qwen3 大语言模型，用于医学影像的分析和问答
    """
    
    def __init__(self, root=".", lora_path=None):
        """
        初始化 DoctorVL
        
        Args:
            root: 模型根目录
            lora_path: LoRA 适配器路径，如果提供则加载
        """
        # 初始化 Qwen3 大语言模型，用于处理文本问答
        self.toker = QWen3Doctor(root + "/Qwen/Qwen3-VL-2B-Instruct")
        # 初始化 Sam3 图像分割模型，用于处理医学影像
        self.eye = Sam3Doctor(root + "/sam3/sam3-8b5")
        
        # 如果提供了 LoRA 路径，则加载 LoRA 适配器
        if lora_path:
            self.eye.load_lora(lora_path)

    def train(self, database_path, epoch, mode="all", batch_size=4, learning_rate=1e-4):
        """
        训练模型
        
        Args:
            database_path: 数据库路径
            epoch: 训练轮数
            mode: 训练模式，"all"、"sam3" 或 "llm"
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            0 表示训练成功
        """
        train_dataset = DoctorDataset(database_path)
        if mode != "llm":
            print("SAM 3 TRAINING")
            # 提取训练数据
            nii_list = [item[0] for item in train_dataset]
            mask_list = [item[1] for item in train_dataset]
            # 训练 Sam3 模型
            self.eye.train(nii_list, mask_list, "./model/sam3_lora", 
                          batch_size=batch_size, epochs=epoch, 
                          learning_rate=learning_rate)
        if mode != "sam3":
            print("QWEN3 TRAINING")
            # 训练 Qwen3 模型
            self.toker.train(train_dataset, batch_size=batch_size, 
                           epochs=epoch, learning_rate=learning_rate)

        return 0

    def segment(self, img: Image.Image, text="tumor"):
        """
        对医学影像进行分割
        
        Args:
            img: 输入图像
            text: 文本提示，用于引导分割
            
        Returns:
            预测的掩码
        """
        return self.eye(img, text)

    def generate(self, messages, text_stream=None, new_token_num=128):
        """
        生成文本回答
        
        Args:
            messages: 输入消息列表
            text_stream: 文本流对象，用于流式输出
            new_token_num: 生成的新 token 数量
            
        Returns:
            生成的文本列表（如果不使用文本流）或 0（如果使用文本流）
        """
        return self.toker(messages, text_stream, new_token_num)

    def analyze(self, img: Image.Image, question: str, text="tumor"):
        """
        分析医学影像并回答问题
        
        Args:
            img: 输入图像
            question: 问题文本
            text: 文本提示，用于引导分割
            
        Returns:
            生成的回答文本
        """
        # 首先进行分割
        mask = self.segment(img, text)
        
        # 构建消息列表
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': img
                    },
                    {
                        'type': 'text',
                        'text': question
                    }
                ]
            }
        ]
        
        # 生成回答
        response = self.generate(messages)
        return response


if __name__ == "__main__":
    # 创建模型实例
    model = DoctorVL("../model")
