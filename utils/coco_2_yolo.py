import json
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def convert_coco_json(json_file, image_dir, output_dir, subset):
    """Converts COCO JSON format to YOLO label format."""
    save_dir = make_dirs(os.path.join(output_dir, "images", subset))  # Create directory to save images
    label_dir = make_dirs(os.path.join(output_dir, "labels", subset))  # Create directory to save labels
    
    with open(json_file) as f:
        data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}
    imgToAnns = defaultdict(list)
    for ann in data['annotations']:
        imgToAnns[ann['image_id']].append(ann)

    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Processing {json_file}"):
        img = images[img_id]
        h, w, f = img['height'], img['width'], img['file_name']
        
        bboxes = []
        for ann in anns:
            if ann["iscrowd"]:
                continue

            box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # Convert to center coordinates
            box[[0, 2]] /= w  # Normalize x
            box[[1, 3]] /= h  # Normalize y

            if box[2] <= 0 or box[3] <= 0:
                continue

            cls = ann["category_id"] - 1  # Category index
            box = [cls] + box.tolist()
            bboxes.append(box)

        # Write YOLO formatted label file
        label_file_path = Path(label_dir) / Path(f).with_suffix(".txt").name
        with open(label_file_path, "w") as file:
            for bbox in bboxes:
                line = bbox  # Category index and normalized bounding box
                file.write(" ".join(map(str, line)) + "\n")
        
        # Copy image to the target directory
        img_src_path = os.path.join(image_dir, f)
        img_dst_path = os.path.join(save_dir, f)
        if not os.path.exists(img_dst_path):
            im = Image.open(img_src_path)
            im.save(img_dst_path)

    return categories

def create_data_yaml(output_dir, categories):
    data = {
        'train': os.path.join(output_dir, 'images', 'train'),
        'val': os.path.join(output_dir, 'images', 'val'),
        'nc': len(categories),
        'names': list(categories.values())
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

if __name__ == "__main__":
    base_dir = "UIIS/UDW"
    output_dir = "UIIS_yolo"
    
    train_categories = convert_coco_json(os.path.join(base_dir, "annotations/train.json"), os.path.join(base_dir, "train"), output_dir, 'train')
    val_categories = convert_coco_json(os.path.join(base_dir, "annotations/val.json"), os.path.join(base_dir, "val"), output_dir, 'val')
    
    assert train_categories == val_categories, "Categories in training and validation sets do not match"
    create_data_yaml(output_dir, train_categories)
