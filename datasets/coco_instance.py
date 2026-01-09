import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
from pycocotools.coco import COCO

class CocoInstanceDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=256):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.anns = list(self.coco.anns.values())

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_info = self.coco.loadImgs(ann["image_id"])[0]

        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = read_image(img_path)  # Loads as RGB (C, H, W)
        image = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C) numpy for cropping

        # bounding box crop
        x, y, w, h = map(int, ann["bbox"])
        crop = image[y:y+h, x:x+w]

        # instance mask
        mask = self.coco.annToMask(ann)[y:y+h, x:x+w]

        # Resize using torchvision transforms
        crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
        crop = Resize((self.img_size, self.img_size))(crop).squeeze(0)  # (C, H, W)
        
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        mask = Resize((self.img_size, self.img_size))(mask).squeeze(0)  # (1, H, W)

        crop = crop / 255.0
        mask = mask

        return crop, mask
