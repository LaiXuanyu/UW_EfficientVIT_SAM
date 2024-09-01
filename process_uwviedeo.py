import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from models.UW_EffSAM import UW_EffSSAM
from PIL import Image
import time

def predict_mask(pred_masks, ignored_masks):
    pred_masks = torch.argmax(pred_masks[1:, ...], dim=0) + 1  
    if ignored_masks is not None:
        pred_masks[pred_masks == 0] = 0
    return pred_masks

def process_frame(frame, model, device):
    with torch.no_grad():
        output = model([frame])
    image_result = output[0]
    image = image_result['images']
    masks = np.array(image_result['masks'])
    boxes = np.array(image_result['boxes'])
    labels = image_result['labels']
    return frame, masks, boxes, labels

def overlay_mask_on_frame(frame, mask_cls_pred, num_classes, alpha=0.5):
    cmap = plt.get_cmap('tab10', num_classes)
    norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1)
    colored_mask = cmap(norm(mask_cls_pred))[..., :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def draw_boxes_and_labels_on_frame(frame, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        rect = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(frame, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def video_processing(input_video_path, output_video_path, model, device):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_inference_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame_resized, masks, boxes, labels = process_frame(frame, model, device)
        num_classes = 7


        result_frame = overlay_mask_on_frame(frame_resized, masks, num_classes)

        result_frame = draw_boxes_and_labels_on_frame(result_frame, boxes, labels)

        end_time = time.time()

        inference_time = end_time - start_time
        total_inference_time += inference_time
        frame_count += 1

        out.write(result_frame)


    cap.release()
    out.release()


    avg_inference_time = total_inference_time / frame_count
    avg_fps = 1.0 / avg_inference_time

    print(f"Average Inference Time per Frame: {avg_inference_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_ckp = "runs/detect/train4/weights/best.pt"
yolo_type = "YOLOv8"
effsam_ckp = "checkpoint/eff_sam_l0.pt"
effsam_type = "l0"
multimask_output = False
input_type = "image"
conf = 0.4

model = UW_EffSSAM(
    yolo_ckp=yolo_ckp,
    yolo_type=yolo_type,
    effsam_ckp=effsam_ckp,
    effsam_type=effsam_type,
    multimask_output=multimask_output,
    input_type=input_type,
    conf=conf
).to(device)

video_processing("demo/d_f.mp4", "yolov8s_new.mp4", model, device)
