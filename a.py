import argparse
import datetime
import time
from pathlib import Path
import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from pymongo import MongoClient
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np


def detect(save_img=False):
    connection_string = "mongodb://Lset:LSET432@202.83.16.6:27017/"
    client = MongoClient(connection_string)
    database = client["analytics"]
    collection = database["source"]
    c=0

    model_data_frame = pd.DataFrame(list(collection.find({"_id":21})))
    print(model_data_frame)
    user_pass=model_data_frame['sourceUrl']
    ad=model_data_frame['userName']
    sour="rtsp://"+ad+":"+"Lset@123@"+user_pass
    url_string = str(sour[0])
    print(url_string)

    print("*"*30)
    #source, weights, view_img, save_txt, imgsz = "/media/lset/eizen/chiru/final_yolo_lite/YOLOv5-Lite/video.mp4", opt.weights, opt.view_img, opt.save_txt, opt.img_size

    source, weights, view_img, save_txt, imgsz = url_string, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        #view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    


    #print(list(dataset))
    print("**********************************************")

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        im0s = np.array(im0s)  # Convert im0s to NumPy array
        im0s = torch.from_numpy(im0s).to(device)  # Convert NumPy array to tensor

        # Inference
            
        pred = model(img, augment=opt.augment)[0]
            
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # Apply Classifier
        # Process detections
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            things_present=[]


            for c in det[:, -1].unique(): 
                 # detections per class
                things_present.append(names[int(c)])
            print("__________________________________________things present_______________________________________")
            print(things_present)
            DATA={
                "modelId": [
                    3
                ],
                "currentTime": datetime.datetime.now(),
                "startTime": datetime.datetime.now(),
                "endTime": datetime.datetime.now(),
                "zoneId": model_data_frame["zoneId"][0],
                "sourceId": int(model_data_frame["sourceId"][0]),
                "frameno": 1,
                "tenantId": int(model_data_frame["tenantId"][0]),
                "analyticsId": int(model_data_frame["analyticsId"][0]),
                "analyticsTypeId": int(model_data_frame["analyticsTypeId"][0]),
                "duration": 1,
                "fps": 25,
                "events": [],
                "activities": things_present
                }
            collection1 = database["edgeDeviceRawCollection"]

            print(DATA)


            # to store each document 1 second 3 or 4 frames 
            if c==75:

                collection1.insert_one(DATA)
                #print(DATA)
                c=0
            c+=1

            if view_img:
                cv2.waitKey(1)  # 1 millisecond


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
