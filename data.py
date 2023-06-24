import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PneumoniaTumorDataset_No(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.transform = transforms.Compose([
            transforms.Resize(size = (256,256)),
            transforms.RandomRotation(degrees = (-20,+20)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])       
        self._load_dataset()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB').resize((224,224))
        image = self.transform(image)
        label = torch.tensor(0)
        return image, label
    
    def _load_dataset(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg') or filename.startswith('I'):
                image_path = os.path.join(self.root_dir, filename)
                self.image_paths.append(image_path)


class PneumoniaTumorDataset_Yes(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.transform = transforms.Compose([
            transforms.Resize(size = (256,256)),
            transforms.RandomRotation(degrees = (-20,+20)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]) 
        self._load_dataset()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path).convert('RGB').resize((224,224))
        image = self.transform(image)
        label = torch.tensor(1)
        return image, label
    
    def _load_dataset(self):
        for filename in os.listdir(self.root_dir):
            if filename.endswith('.jpg') or filename.startswith('p'):
                image_path = os.path.join(self.root_dir, filename)
                self.image_paths.append(image_path)


class MixedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        
    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
    
    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]