from select import select
import torch
from torch.utils.data import Dataset
from PIL import Image

class ReIDDataset(Dataset):
    def __init__(self, dataset, transform=None, is_train=True):
        if is_train:
            self.dataset = dataset.trainset
        else:
            self.dataset = dataset.queryset + dataset.galleryset
            self.num_q = dataset.num_imgs_q
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im_path, vid, camid = self.dataset[index]
        image = Image.open(im_path)
        if self.transform:
            image = self.transform(image)
        
        return image, vid, camid
        