import torch
from torch.utils.data import DataLoader
from datasets.coco_instance import CocoInstanceDataset
from models.Unet import UNet
from tqdm import tqdm
import accelerate as acc

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    dataset = CocoInstanceDataset(
        img_dir="data/coco2017/val2017",
        ann_file="data/coco2017/annotations/instances_val2017.json"
    )
    
    #regular setup
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    #accerlerator setup
    accelerator = acc.Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    #cuda
    
    for epoch in range(20):
        model.train()
        epoch_loss = 0

        for imgs, masks in tqdm(loader):

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

           # epoch_loss += loss.item()

        print(f"Epoch {epoch}: loss={100 / len(loader):.4f}")

    torch.save(model.state_dict(), "weights/unet_coco_v1.pth")

    