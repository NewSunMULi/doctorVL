import json
from torch.utils.data import Dataset


class DoctorDataset(Dataset):
    """
    医学影像数据集类，用于加载和处理医学影像数据
    
    该类继承自 PyTorch 的 Dataset 类，支持数据的预加载和动态加载
    """
    
    def __init__(self, data_path, preload=False):
        self.data_path = data_path
        self.preload = preload
        self.data = self.__get_data()

    def __get_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
             return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["img_nii"]
        mask = self.data[idx]["mask_nii"]
        label = self.data[idx]["label"]
        llm_index = self.data[idx]["llm_index"]

if __name__ == "__main__":
    dataset = DoctorDataset(data_path="../../dataset/dataset.json", preload=True)
    print(dataset[0])
