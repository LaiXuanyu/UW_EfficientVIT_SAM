from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt



def main():
    data_path = 'datasets/UIIS_yolo'
    yaml_path = os.path.join(data_path, 'UIIS.yaml')

    model_yaml = 'ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml'
    pre_model = 'checkpoint/yolov8s.pt'
    model = YOLO(model_yaml,task='detect').load(pre_model)  # build from YAML and transfer weights
    results = model.train(data=yaml_path,epochs=80,imgsz=640,batch=8,workers = 4)

if __name__ == "__main__":
     main()
