import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_data(filename: str) -> np.ndarray:
    img = nib.load(filename)
    return img.get_fdata()

def show_img_one(img: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == "__main__":
    a = get_data("../../data/50/50/P2.nii.gz")
    show_img_one(a[30])
