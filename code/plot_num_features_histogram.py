import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.io as sio
from torchvision.transforms import v2
from tqdm import tqdm


class SegmentationDataset(Dataset):
    def __init__(self, folders, transforms=None):
        self.folders = folders
        self.transforms = transforms

    def __getitem__(self, idx):
        folder = self.folders[idx]
        image_path = os.path.join(folder, 'image.tif')
        img = Image.open(image_path).convert("RGB")

        masks = []
        labels = []
        for i in range(1, 5):
            mask_path = os.path.join(folder, f'class{i}.tif')
            if os.path.exists(mask_path):
                mask = sio.imread(mask_path)
                mask = np.array(mask) > 0
                labeled_mask, num_features = scipy.ndimage.label(mask)
                for j in range(1, num_features + 1):
                    component_mask = (labeled_mask == j)
                    masks.append(
                        torch.as_tensor(component_mask, dtype=torch.uint8))
                    labels.append(i)

        boxes = []
        valid_masks = []
        valid_labels = []
        for mask, label in zip(masks, labels):
            pos = torch.nonzero(mask)
            if pos.numel() == 0:
                continue
            else:
                xmin = torch.min(pos[:, 1])
                xmax = torch.max(pos[:, 1])
                ymin = torch.min(pos[:, 0])
                ymax = torch.max(pos[:, 0])
                if xmin < xmax and ymin < ymax:
                    boxes.append(torch.tensor([xmin, ymin, xmax, ymax],
                                              dtype=torch.float32))
                    valid_masks.append(mask)
                    valid_labels.append(label)

        if len(valid_masks) == 0:
            valid_masks = [torch.zeros((img.height, img.width),
                                       dtype=torch.uint8)]
            valid_labels = [1]
            boxes = [torch.tensor([0, 0, 1, 1], dtype=torch.float32)]

        masks = torch.stack(valid_masks)
        labels = valid_labels
        boxes = torch.stack(boxes)

        target = {
            'boxes': boxes,
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.folders)


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    root = 'data/train'
    folders = [os.path.join(root, d) for d in os.listdir(root)]
    transform = v2.Compose([v2.ToImage(), v2.ConvertImageDtype(torch.float)])
    dataset = SegmentationDataset(folders, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    num_features_list = []

    for _, targets in tqdm(dataloader, desc="Processing"):
        target = targets[0]
        num_boxes = target['boxes'].shape[0]
        num_features_list.append(num_boxes)

    plt.figure(figsize=(10, 6))
    plt.hist(num_features_list, bins=range(0, max(num_features_list)+2),
             edgecolor='black', align='left')
    plt.xlabel("Number of Features (Bounding Boxes) per Image")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Ground Truth Boxes per Image")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("num_features_histogram.png")


if __name__ == "__main__":
    main()
