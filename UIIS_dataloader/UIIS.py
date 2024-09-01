from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
import cv2
from sklearn.metrics import jaccard_score
import os
from utils.tools import visualize_binary_image


class UIISDataset(Dataset):
    def __init__(self, image_dir, ann_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = np.array(Image.open(image_path).convert("RGB"))
        
        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in anns:
            category_id = ann['category_id']
            c = self.coco.annToMask(ann) == 1
            mask[self.coco.annToMask(ann) == 1] = category_id
        if self.transform:
            image = self.transform(image)
        
        return image, mask, image_id
