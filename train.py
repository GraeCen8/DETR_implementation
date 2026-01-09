import torch
from torch.utils.data import DataLoader
from datasets.voc_segmentation import VOCSegmentation
from models.Unet import Unet
from tqdm import tqdm
import os

device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_dir, "data", "VOC2012")

    dataset = VOCSegmentation(
        root=dataset_dir,
        split="train",
        img_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = Unet(out_channels=21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    for epoch in range(20):
        model.train()
        epoch_loss = 0.0

        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = model(imgs)
                loss = criterion(preds, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}: loss={epoch_loss / len(loader):.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/unet_voc_bf16.pth")
