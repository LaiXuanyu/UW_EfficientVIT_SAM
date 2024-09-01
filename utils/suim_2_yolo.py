import os
import numpy as np
import torch
from PIL import Image

'''
(0, 0, 1): 0,  # HD: Human divers
(0, 1, 0): 1,  # PF: Plants/sea-grass
(0, 1, 1): 2,  # WR: Wrecks/ruins
(1, 0, 0): 3,  # RO: Robots/instruments
(1, 0, 1): 4,  # RI: Reefs and invertebrates
(1, 1, 0): 5,  # FV: Fish and vertebrates
(1, 1, 1): 6   # SR: Sand/sea-floor (& rocks)

names: ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor']  # Class names

nc: 7
'''

def get_bounding_boxes(ground_truth_maps: np.array) -> torch.Tensor:
    """
    Get the bounding boxes for multiple ground truth masks

    Arguments:
        ground_truth_maps: Ground truth masks in array format [num_classes, H, W]

    Return:
        bboxes: Tensor of bounding boxes for each mask in format [B, 5]
    """
    bboxes = []
    num_classes, H, W = ground_truth_maps.shape

    for class_idx in range(num_classes):
        ground_truth_map = ground_truth_maps[class_idx]
        idx = np.where(ground_truth_map > 0)
        
        if len(idx[0]) == 0 or len(idx[1]) == 0:
            # If there are no positive pixels for this class, skip it
            continue

        x_indices = idx[1]
        y_indices = idx[0]
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        bbox = [x_min, y_min, x_max, y_max]
        bboxes.append([class_idx, x_min, y_min, x_max, y_max])

    # Convert list to tensor of shape [B, 5]
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    return bboxes_tensor

def save_yolo_format(bboxes, output_path, image_width, image_height):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    with open(output_path, 'w') as f:
        for bbox in bboxes:
            class_id, x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def process_image_and_mask(img_name, mask_dir, mask_suffix, output_dir):

    mask_path = os.path.join(mask_dir, img_name + mask_suffix)

    mask = Image.open(mask_path).convert('RGB')
    W, H = mask.size
    mask = np.array(mask)  # Convert to numpy array for further processing
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask = torch.tensor(mask, dtype=torch.float32)  # Ensure mask is a float tensor
    categories = get_categories_from_mask(mask)

    bbox_data = get_bounding_boxes(categories)

    output_path = os.path.join(output_dir, img_name + '.txt')
    save_yolo_format(bbox_data, output_path, W, H)

    return {
        "ground_truth": categories,
        "boxes": bbox_data,
    }

def get_categories_from_mask(mask):
    """
    Get category masks from color mask

    Arguments:
        mask: Input color mask image

    Returns:
        categories: Category masks, shape [num_classes, H, W]
    """
    if mask.ndim == 2:
        mask = mask[:, :, None]
    elif mask.shape[2] != 3:
        raise ValueError(f"Expected mask shape [H, W, 3], got {mask.shape}")

    categories = torch.zeros(mask.shape[0], mask.shape[1], 7, dtype=torch.bool, device=mask.device)

    fish_color = torch.tensor([1, 1, 0], device=mask.device)
    reef_color = torch.tensor([1, 0, 1], device=mask.device)
    plant_color = torch.tensor([0, 1, 0], device=mask.device)
    wreck_color = torch.tensor([0, 1, 1], device=mask.device)
    human_color = torch.tensor([0, 0, 1], device=mask.device)
    robot_color = torch.tensor([1, 0, 0], device=mask.device)
    sand_color = torch.tensor([1, 1, 1], device=mask.device)

    #categories[:, :, 0] = torch.all(mask == fish_color, dim=-1)     # Fish and vertebrates
    #categories[:, :, 1] = torch.all(mask == reef_color, dim=-1)     # Reefs and invertebrates
    #categories[:, :, 2] = torch.all(mask == plant_color, dim=-1)    # Plants/sea-grass
    categories[:, :, 3] = torch.all(mask == wreck_color, dim=-1)    # Wrecks/ruins
    categories[:, :, 4] = torch.all(mask == human_color, dim=-1)    # Human divers
    categories[:, :, 5] = torch.all(mask == robot_color, dim=-1)    # Robots/instruments
    # categories[:, :, 6] = torch.all(mask == sand_color, dim=-1)     # Sand/sea-floor (& rocks)

    return categories.permute(2, 0, 1)  # Adjust channel order to [class, H, W]

def process_dataset(mask_dir, mask_suffix, output_dir):
    for mask_name in os.listdir(mask_dir):
        if mask_name.endswith(mask_suffix):
            mask_name = mask_name[:-len(mask_suffix)]  # Remove suffix to get base filename
            result = process_image_and_mask(mask_name, mask_dir, mask_suffix, output_dir)
            print(f"Processed {mask_name}.")

# Example usage
mask_dir = "datasets/SUIM/train_val/train_val/masks"
mask_suffix = ".bmp"
output_dir = "label"

process_dataset(mask_dir, mask_suffix, output_dir)
