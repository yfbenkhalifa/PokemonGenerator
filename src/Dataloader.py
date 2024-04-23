from torch.utils.data import Dataset
from datasets import Dataset as HfDataset


class ImageHuggingfaceDataset(Dataset):
    def __init__(self, dataset: HfDataset, label: str = 'caption', category: str = 'image', transform=None):
        self.len = len(dataset)
        self.img_labels = dataset[label]
        self.images = dataset[category]
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.img_labels[idx]
