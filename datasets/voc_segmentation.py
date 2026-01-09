import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

class VOCSegmentation(Dataset):
    def __init__(self, root, split="train", img_size=256):
        self.root = root
        self.img_size = img_size

        split_file = os.path.join(
            root, "ImageSets", "Segmentation", f"{split}.txt"
        )

        with open(split_file) as f:
            self.ids = [x.strip() for x in f.readlines()]

        # Image transforms
        self.image_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.BILINEAR),
            T.ToTensor(),  # converts to float32 and scales to [0, 1]
        ])

        # Mask transform (no normalization!)
        self.mask_transform = T.Resize(
            (img_size, img_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        img_path = os.path.join(self.root, "JPEGImages", f"{img_id}.jpg")
        mask_path = os.path.join(self.root, "SegmentationClass", f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to tensor WITHOUT scaling
        mask = torch.as_tensor(
            torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
        ).view(self.img_size, self.img_size).long()

        return image, mask
