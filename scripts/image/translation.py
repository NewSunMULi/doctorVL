import nibabel as nib
import numpy as np

class NibabelImage:
    def __init__(self, fpath: str):
        self.fpath = fpath
        self.img = nib.load(self.fpath)

    def to_video(self):
        return self.img.get_fdata()