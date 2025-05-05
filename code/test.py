import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
import zipfile
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

MODEL_PATH = "conv_b2_seed77.pth"
DETECTIONS_PER_IMG = 350

class SegmentationDatasetTest(Dataset):
    def __init__(self, image_files, image_id_mapping, transforms=None):
        self.image_files = image_files
        self.image_id_mapping = image_id_mapping
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        img = Image.open(image_path).convert("RGB")

        file_name = os.path.basename(image_path)
        image_id = self.image_id_mapping[file_name]

        if self.transforms:
            img = self.transforms(img)

        return img, image_id

    def __len__(self):
        return len(self.image_files)

test_transform = v2.Compose([
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float)
])

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

def encode_mask(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def collate_fn(batch):
    return tuple(zip(*batch))

def zip_file():
    zip_filename = MODEL_PATH.replace('.pth', '.zip')
    files_to_zip = ["test-results.json"]

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file)

    print(f"finish zipping {zip_filename}")
    
def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with open("data/test_image_name_to_ids.json", 'r') as f:
        image_id_list = json.load(f)
    image_id_mapping = {item['file_name']: item['id'] for item in image_id_list}

    data_root = "data/test_release"
    image_files = [os.path.join(data_root, fname) for fname in os.listdir(data_root) if fname.endswith('.tif')]

    test_dataset = SegmentationDatasetTest(image_files, image_id_mapping, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    model = MaskRCNNModel(num_classes=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, image_id in zip(outputs, ids):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                masks = output['masks'].squeeze(1).cpu().numpy()  # (N, H, W)

                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    xmin, ymin, xmax, ymax = box
                    width = xmax - xmin
                    height = ymax - ymin

                    binary_mask = mask > 0.5
                    rle_mask = encode_mask(binary_mask)

                    result = {
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(xmin), float(ymin), float(width), float(height)],
                        "score": float(score),
                        "segmentation": rle_mask
                    }
                    results.append(result)

    with open("test-results.json", 'w') as f:
        json.dump(results, f)

    print(f"Saved results to test-results.json")

if __name__ == "__main__":
    test()
    zip_file()