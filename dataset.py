import os
import json
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def make_dataset(base_path, jsonfile):

    with open(jsonfile, 'r') as fhandler:
        dataset = json.load(fhandler)

    images = []
    for item in dataset:
        imgpath = os.path.join(base_path, item['path'])
        income = float(item['income'])
        images.append((imgpath, income))

    return images


class DollarDataset(object):
    def __init__(self, batch_size=64, valid_batch_size=16, img_size=300,
                 base_path='./data', train_json='train_set.json', valid_json='valid_set.json'):
        
        self.train_path = train_json
        self.valid_path = valid_json
        self.base_path = base_path
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.img_size = img_size

        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Scale(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def get_train_loader(self):
        transformed_dataset_train = OverSampleDataset(base_path=self.base_path,
                                                      jsonfile=self.train_path,
                                                      transform=self.transforms['train'],
                                                      is_train=True, batch_size=self.batch_size)
        dataloader = DataLoader(transformed_dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=5)
        return dataloader

    def get_valid_loader(self):
        transformed_dataset_valid = OverSampleDataset(base_path=self.base_path,
                                                    jsonfile=self.valid_path,
                                                    transform=self.transforms['val'],
                                                    is_train=False, batch_size=self.batch_size)
        dataloader = DataLoader(transformed_dataset_valid, batch_size=self.valid_batch_size, shuffle=False, num_workers=5)
        return dataloader


class OverSampleDataset(Dataset):
    def __init__(self, base_path, jsonfile, transform=None,
                 is_train=False, batch_size=64,
                 shapes=[(300,300), (350,350), (400,400)]):
        super(OverSampleDataset, self).__init__()

        self.base_path = base_path
        self.is_train = is_train
        self.transform = transform
        self.batch_size = batch_size
        self.shapes = shapes
        self.shape = shapes[0]
                
        self.imgs = make_dataset(base_path, jsonfile)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')

        if self.is_train and index % self.batch_size == 0:
            self.shape = self.shapes[random.randint(0,len(self.shapes)-1)]

        img = img.resize(self.shape, Image.ANTIALIAS)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)