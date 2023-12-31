import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

class MelSpecDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dir = dir
        self.transform = transform
        self.genre_key = {
                "blues":0, 
                "country":1, 
                "classical":2, 
                "disco":3, 
                "jazz":4, 
                "hiphop":5,  
                "reggae":6,
                "pop":7,
                "metal":8,
                "rock":9
            }

        # read label file and pair with image
        self.labels = {}
        with open(f'{dir}/labels.txt', 'r') as file:
            for line in file:
                terms = line.strip().split(', ')
                if len(terms) == 2:
                    image_name, label = terms
                    self.labels[image_name] = int(self.genre_key[label])

        self.images = [file for file in os.listdir(dir) if file.endswith('.png') and file in self.labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.dir, img_name)
        image = Image.open(img_path)
        #image = Image.open(img_path).crop((125, 55, 900, 395))
        #print(image.size)
        #plt.imshow(image)
        #plt.show()
        if self.transform:
            image = self.transform(image)
        label = self.labels[img_name]
        return image, label    