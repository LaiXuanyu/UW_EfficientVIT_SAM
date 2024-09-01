import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval
from UIIS_dataloader.UIIS import UIISDataset
from models.UW_EffSAM import UW_EffSSAM
import numpy as np
import json
import os
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UW_EffSSAM(
    yolo_ckp='runs/detect/train4/weights/best.pt',
    yolo_type='YOLOv8',
    effsam_ckp="checkpoint/eff_sam_l0.pt",
    effsam_type="l0",
    multimask_output=False,
    input_type="image",
    conf=0.25
).to(device)

image_dir = "datasets/UIIS/UDW/val"
ann_file = "datasets/UIIS/UDW/annotations/val.json"

dataset = UIISDataset(image_dir, ann_file, transform=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

def convert_bbox_to_coco_format(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return [x1, y1, width, height]

def convert_mask_to_rle(mask):
    """
    Convert a binary mask to RLE format.

    Parameters:
    - mask (numpy array): Binary mask, values should be 0 or 1.

    Returns:
    - RLE encoding
    """
    mask = mask.astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('ascii')  # RLE counts must be ascii encoded strings
    return rle

def evaluate_model(model, dataloader, device, coco_gt):
    results = []
    with torch.no_grad():
        for images, gts, image_ids in dataloader:
            images = images.to(device)
            outputs = model(images,image_ids)

            for output, image_id in zip(outputs, image_ids):
                # Convert lists in output to NumPy arrays
                boxes = np.array(output["boxes"])
                scores = np.array(output["scores"])
                labels = np.array(output["labels"])
                masks = np.array(output["masks"])

                for box, score, label, mask in zip(boxes, scores, labels, masks):
                    bbox_coco_format = convert_bbox_to_coco_format(box)
                    rle_mask = convert_mask_to_rle(mask)  # Convert mask to RLE format
                    result = {
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": bbox_coco_format,
                        "score": float(score),
                        "segmentation": rle_mask
                    }
                    results.append(result)

    # Save results in COCO format
    try:
        with open("results.json", "w") as f:
            json.dump(results, f)
        print("Results saved to results.json")
    except Exception as e:
        print(f"Error saving results: {e}")

    # Load results and perform evaluation
    try:
        coco_dt = coco_gt.loadRes("results.json")
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")

# Load COCO annotations
coco_gt = COCO(ann_file)

# Evaluate model
evaluate_model(model, dataloader, device, coco_gt)
