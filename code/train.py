import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.transforms import v2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import skimage.io as sio
import scipy.ndimage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

BATCH_SIZE = 2
EPOCHS = 60
PATIENCE = 20
LEARNING_RATE = 1e-4
NAMING = "conv_b2"
DETECTIONS_PER_IMG = 350

train_transform = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float)
])
valid_transform = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float)
])

def set_seed(seed=77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
                    masks.append(torch.as_tensor(component_mask, dtype=torch.uint8))
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
                    boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
                    valid_masks.append(mask)
                    valid_labels.append(label)

        if len(valid_masks) == 0:
            valid_masks = [torch.zeros((img.height, img.width), dtype=torch.uint8)]
            valid_labels = [1]
            boxes = [torch.tensor([0, 0, 1, 1], dtype=torch.float32)]
            print(f"Warning: No valid masks found for {image_path}, using dummy mask")

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

class ConvNeXtBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        convnext = torchvision.models.convnext.convnext_base(weights='DEFAULT')
        self.feature0 = convnext.features[0]
        self.feature1 = convnext.features[1]
        self.feature2 = convnext.features[2]
        self.feature3 = convnext.features[3]
        self.feature4 = convnext.features[4]
        self.feature5 = convnext.features[5]
        self.feature6 = convnext.features[6]
        self.feature7 = convnext.features[7]

    def forward(self, x):
        out0 = self.feature0(x)
        out1 = self.feature1(out0)
        out2 = self.feature2(out1)
        out3 = self.feature3(out2)
        out4 = self.feature4(out3)
        out5 = self.feature5(out4)
        out6 = self.feature6(out5)
        out7 = self.feature7(out6)
        return {
            'feature1': out1,
            'feature3': out3,
            'feature5': out5,
            'feature7': out7
        }

class MaskRCNNModel(torch.nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(MaskRCNNModel, self).__init__()
        backbone = ConvNeXtBackbone()
        returned_layers = {
            'feature1': '0',
            'feature3': '1',
            'feature5': '2',
            'feature7': '3'
        }
        backbone_with_fpn = BackboneWithFPN(
            backbone,
            return_layers=returned_layers,
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256
        )
        
        # Initialize Mask R-CNN with the custom backbone
        self.model = MaskRCNN(
            backbone_with_fpn,
            num_classes=num_classes
        )
        
        # Replace the classification head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        
        # Replace the mask head
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        self.model.roi_heads.detections_per_img = DETECTIONS_PER_IMG

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, optimizer, data_loader, device, scaler):
    model.train()
    running_loss = 0.0
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    return running_loss / len(data_loader)

def create_coco_gt(dataset, device):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class{i}"} for i in range(1, 5)]
    }
    ann_id = 1
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        img_id = int(target['image_id'][0])
        coco_gt["images"].append({
            "id": img_id,
            "width": img.shape[-1],
            "height": img.shape[-2],
            "file_name": f"image_{img_id}.tif"
        })

        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        masks = target['masks'].cpu().numpy()
        areas = target['area'].cpu().numpy()
        iscrowd = target['iscrowd'].cpu().numpy()

        for box, label, mask, area, crowd in zip(boxes, labels, masks, areas, iscrowd):
            if label == 0:
                continue
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "segmentation": rle,
                "iscrowd": int(crowd)
            })
            ann_id += 1

    return coco_gt

def create_coco_dt(outputs, image_ids, device):
    coco_dt = []
    ann_id = 1
    for output, img_id in zip(outputs, image_ids):
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        masks = output['masks'].cpu().numpy()

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if label == 0 or score < 0.05:
                continue
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            if w <= 0 or h <= 0:
                continue
            mask = (mask.squeeze() > 0.5).astype(np.uint8)
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            coco_dt.append({
                "id": ann_id,
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score),
                "segmentation": rle
            })
            ann_id += 1

    return coco_dt

def validate(model, data_loader, device):
    model.eval()
    coco_gt = create_coco_gt(data_loader.dataset, device)
    coco = COCO()
    coco.dataset = coco_gt
    coco.createIndex()

    coco_dt = []
    image_ids = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validating", leave=False):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            image_ids.extend([t['image_id'].item() for t in targets])
            coco_dt.extend(create_coco_dt(outputs, [t['image_id'].item() for t in targets], device))

    if not coco_dt:
        return 0.0

    coco_dt_coco = coco.loadRes(coco_dt)
    coco_eval = COCOeval(coco, coco_dt_coco, 'segm')
    # coco_eval.params.iouThrs = np.array([0.5])  # 僅計算 IoU=0.5 的 mAP
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0] if coco_eval.stats[0] >= 0 else 0.0
    return float(mAP)

def main(seed=77):
    set_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    root = 'data/train'
    all_folders = [os.path.join(root, d) for d in os.listdir(root)]

    train_folders, valid_folders = train_test_split(all_folders, test_size=0.2, random_state=seed)

    train_dataset = SegmentationDataset(train_folders, transforms=train_transform)
    valid_dataset = SegmentationDataset(valid_folders, transforms=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = MaskRCNNModel(num_classes=5, pretrained=True).to(device)
    scaler = torch.amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)

    no_improvement_epochs = 0
    train_losses = []
    valid_maps = []
    max_map = 0.0

    for epoch in range(EPOCHS):
        train_loss = train(model, optimizer, train_loader, device, scaler)
        valid_map = validate(model, valid_loader, device)

        train_losses.append(train_loss)
        valid_maps.append(valid_map)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation mAP: {valid_map:.4f}")

        no_improvement_epochs += 1
        if valid_map > max_map:
            print(f"Saving model, Best mAP: {valid_map:.4f}")
            torch.save(model.state_dict(), f'{NAMING}_seed{seed}.pth')
            max_map = valid_map
            no_improvement_epochs = 0

        if no_improvement_epochs >= PATIENCE:
            print("Early stopping")
            break
    print(f"train_losses: {train_losses}")
    print(f"mAP: {valid_maps}")

if __name__ == "__main__":
    main(seed=77)