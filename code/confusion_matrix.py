import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from PIL import Image
import skimage.io as sio
import scipy.ndimage
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Configuration
MODEL_PATH = "conv_b2_95_350bbox_seed77.pth"
BATCH_SIZE = 8
SEED = 77
NUM_CLASSES = 5
SCORE_THRESHOLD = 0.5
DETECTIONS_PER_IMG = 350


def set_seed(seed=77):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


valid_transform = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float)
])


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
                        torch.as_tensor(
                            component_mask,
                            dtype=torch.uint8
                            ))
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
                    boxes.append(torch.tensor(
                        [xmin, ymin, xmax, ymax],
                        dtype=torch.float32))
                    valid_masks.append(mask)
                    valid_labels.append(label)

        if len(valid_masks) == 0:
            valid_masks = [torch.zeros(
                (img.height, img.width),
                dtype=torch.uint8)]
            valid_labels = [1]
            boxes = [torch.tensor([0, 0, 1, 1], dtype=torch.float32)]
            print(f"Warning: No valid masks \
                found for {image_path}, using dummy mask")

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
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        # Replace the mask head
        in_features_mask = self.model.\
            roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
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


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def generate_confusion_matrix(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader,
                                    desc="Generating Confusion Matrix"):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device)
                        for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                # Filter predictions by score threshold
                valid_preds = pred_scores > SCORE_THRESHOLD
                pred_boxes = pred_boxes[valid_preds]
                pred_labels = pred_labels[valid_preds]
                pred_scores = pred_scores[valid_preds]

                # Match predictions to ground truth using IoU
                matched_preds = []
                matched_labels = []

                if len(true_boxes) > 0 and len(pred_boxes) > 0:
                    # Compute IoU matrix
                    iou_matrix = np.zeros((len(true_boxes), len(pred_boxes)))
                    for i, t_box in enumerate(true_boxes):
                        for j, p_box in enumerate(pred_boxes):
                            iou_matrix[i, j] = compute_iou(t_box, p_box)

                    # Assign predictions to ground truth with highest IoU
                    used_preds = set()
                    for i in range(len(true_boxes)):
                        if iou_matrix.shape[1] == 0:
                            break
                        max_iou_idx = np.argmax(iou_matrix[i])
                        max_iou = iou_matrix[i, max_iou_idx]
                        if max_iou > 0.5 and max_iou_idx not in used_preds:
                            matched_preds.append(pred_labels[max_iou_idx])
                            matched_labels.append(true_labels[i])
                            used_preds.add(max_iou_idx)

                    for i in range(len(true_boxes)):
                        if i not in [i for i, _ in enumerate(matched_labels)]:
                            matched_labels.append(true_labels[i])
                            matched_preds.append(0)

                elif len(true_boxes) > 0:
                    matched_preds.extend([0] * len(true_boxes))
                    matched_labels.extend(true_labels)

                all_preds.extend(matched_preds)
                all_labels.extend(matched_labels)

    assert len(all_preds) == len(all_labels), \
        f"Inconsistent lengths: {len(all_preds)}\
            preds vs {len(all_labels)} labels"
    cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))

    return cm


def main():
    set_seed(SEED)
    device = torch.device('cuda')

    # Load validation dataset
    root = 'data/train'
    all_folders = [os.path.join(root, d) for d in os.listdir(root)]

    valid_dataset = SegmentationDataset(all_folders,
                                        transforms=valid_transform)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn)

    model = MaskRCNNModel(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    cm = generate_confusion_matrix(model, valid_loader, device)

    class_names = ['-1'] + [f'{i}' for i in range(1, NUM_CLASSES)]
    plot_confusion_matrix(cm, class_names)

    print("Confusion Matrix:")
    print(cm)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
