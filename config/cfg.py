from box import Box

config = {
    "model": {
        "type": "uw_effsam" ,
        "yolo_ckp": 'runs/detect/train4/weights/best.pt',
        "yolo_type": 'YOLOv8',
        "effsam_ckp": "checkpoint/eff_sam_l0.pt",
        "effsam_type": "l0",
        "multimask_output": False,
        "input_type": "image",
        "conf": 0.25,

    },

    "input": {
        "image_dir": "datasets/UIIS/UDW/val",
        "ann_file": "datasets/UIIS/UDW/annotations/val.json",
    }
}

cfg = Box(config)