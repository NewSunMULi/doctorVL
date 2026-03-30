from dataset.dataset import DoctorDataset
from llm.Qwen.model import QWen3Doctor
from img.sam3.model import Sam3Doctor


class DoctorVL:
    """
    医学影像处理与分析的联合模型类
    
    该类整合了 Sam3 图像分割模型和 Qwen3 大语言模型，用于医学影像的分析和问答
    """
    
    def __init__(self, root="."):
        # 初始化 Qwen3 大语言模型，用于处理文本问答
        self.toker = QWen3Doctor(root + "/Qwen/Qwen3-VL-2B-Instruct")
        # 初始化 Sam3 图像分割模型，用于处理医学影像
        self.eye = Sam3Doctor(root + "/sam3/sam3-8b5")

    def train(self, database_path, epoch, mode="all"):
        train_dataset = DoctorDataset(database_path)
        if mode != "llm":
            print("SAM 3 TRAINING")
            self.eye.train(train_dataset, epochs=epoch)
        if mode != "sam3":
            print("QWEN3 TRAINING")
            self.toker.train(train_dataset, epochs=epoch)

        return 0


if __name__ == "__main__":
    # 创建模型实例
    model = DoctorVL("../model")
