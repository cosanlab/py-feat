import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os
from torchvision import transforms


class image_Loader(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.main_csv = csv_file
        self.img_dir = img_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(170),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_dir, str(self.main_csv["Subject"][index]), str(
            self.main_csv["Task"][index]), str(self.main_csv["Number"][index])+".jpg")
        img = Image.open(img_name)
        img = self.transform(img)
        aus = self.main_csv[["1", "2", "4", "6", "7",
                             "10", "12", "14", "15", "17", "23", "24"]]
        aus = aus.to_numpy(dtype='int')
        return(img, aus[index, :])

    def __len__(self) -> int:
        return self.main_csv.shape[0]
