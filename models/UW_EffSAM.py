import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientvit.sam_model_zoo import create_sam_model
from ultralytics import YOLO
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from PIL import Image
import numpy as np
# Label mapping
# 0: Background
# 1: Fish
# 2: Reefs
# 3: Aquatic Plants
# 4: Wrecks/Ruins
# 5: Human Divers
# 6: Robots
# 7: Sea-Floor

class UW_EffSSAM(nn.Module):
    def __init__(
            self, 
            yolo_ckp: str = "",
            yolo_type: str = "",
            effsam_ckp: str = "",
            effsam_type: str = "",
            multimask_output: bool = False,
            input_type: str = "",
            conf: float = 0.3 
    ):
        super(UW_EffSSAM, self).__init__()

        self.yolo_ckp = yolo_ckp
        self.yolo_type = yolo_type
        self.effsam_ckp = effsam_ckp
        self.effsam_type = effsam_type
        self.multimask_output = multimask_output
        self.input_type = input_type
        self.conf = conf

        # Initialize YOLO detector
        self.detector = YOLO(yolo_ckp)
        # Initialize SAM model
        self.effmodel = create_sam_model(name=effsam_type, weight_url=effsam_ckp)
        self.segmentor = EfficientViTSamPredictor(self.effmodel)

    def forward(self, images,img_id):

        images_np = images.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy array
        images_np = images_np.astype(np.uint8)  # Ensure the images are in uint8 forma

        results = self.detector.predict(images_np, conf=self.conf,verbose = False)
        output = []
        
        for image, result in zip(images, results):
            boxes = result.boxes
            bounding_boxes = boxes.xyxy
            classes = boxes.cls
            confidence = boxes.conf

            masks = self.predictmask(bounding_boxes, image,img_id)
            image_result = {
                "images": image.tolist(),
                "boxes": bounding_boxes.tolist(),
                "scores": confidence.tolist(),
                "labels": classes.tolist(),
                "masks": masks.tolist()
            }
            output.append(image_result)

        return output
    
    def predictmask(self, bounding_box, image,img_id):
        masks_list = []  
        image = image.cpu().numpy()    
        for box in bounding_box:
            self.segmentor.set_image(image)
            box_np = box.cpu().numpy().astype(float)  # Convert to NumPy array and then to float
            mask, _, _ = self.segmentor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=self.multimask_output,
            )
            

            mask = torch.squeeze(torch.from_numpy(mask), dim=0).bool().int().numpy().astype(np.uint8)  # Ensure the mask is in uint8 format

            if mask.any():  # Ensure the mask is not empty
                masks_list.append(mask)  # Add mask to the list

        if len(masks_list) == 0:
            print("No masks generated for this image.")
            print(img_id)
            return torch.empty(0, dtype=torch.uint8)  # Return an empty tensor if no masks were generated

        masks_tensor = torch.stack([torch.from_numpy(mask) for mask in masks_list])  # Convert list to tensor with shape (N, H, W)
    
        return masks_tensor
