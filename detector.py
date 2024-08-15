import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


class YOLOv5Detector(object):
    def __init__(self, dev='cpu'):
        self.weights=ROOT / 'yolov5s.pt'  # model path or triton URL
        self.source=ROOT / 'data/images'  # file/dir/URL/glob/screen/0(webcam)
        self.data=ROOT / 'data/coco128.yaml'  # dataset.yaml path
        self.imgsz=(640, 640)  # inference size (height width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        if dev == "gpu":
            self.device=0  # cuda device i.e. 0 or 0123 or cpu
        else:
            self.device="cpu"  # cuda device i.e. 0 or 0123 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0 or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/detect'  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.vid_stride=1

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size


    def detect(self, source):
        self.listObject = []
        
        self.source = str(source)
        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        self.is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        self.is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or (self.is_url and not self.is_file)
        self.screenshot = self.source.lower().startswith('screen')
        if self.is_url and self.is_file:
            self.source = check_file(self.source)

        if self.webcam:
            self.view_img = check_imshow()
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        elif self.screenshot:
            self.dataset = LoadScreenshots(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        self.bs = len(self.dataset)  # batch_size
        # self.bs = 1
        self.vid_path, self.vid_writer = [None] * self.bs, [None] * self.bs

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        for self.path, self.im, self.im0s, self.vid_cap, self.s in self.dataset:
            with self.dt[0]:
                self.im = torch.from_numpy(self.im).to(self.model.device)
                self.im = self.im.half() if self.model.fp16 else self.im.float()  # uint8 to fp16/32
                self.im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(self.im.shape) == 3:
                    self.im = self.im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                # self.visualize = increment_path(self.save_dir / Path(self.path).stem, mkdir=True) if self.visualize else False
                self.pred = self.model(self.im, augment=self.augment, visualize=self.visualize)

            # NMS
            with self.dt[2]:
                self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for self.i, self.det in enumerate(self.pred):  # per image
                
                self.listPositionX, self.listPositionY = [], []
                
                self.listPosition = []

                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    self.p, self.im0, self.frame = self.path[i], self.im0s[i].copy(), self.dataset.count
                    self.imcc = self.im0s[i].copy()
                    self.s += f'{i}: '
                else:
                    self.p, self.im0, self.frame = self.path, self.im0s.copy(), getattr(self.dataset, 'frame', 0)
                    self.imcc = self.im0s.copy()

                self.p = Path(self.p)  # to Path
                self.s += '%gx%g ' % self.im.shape[2:]  # print string
                self.gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                self.imc = self.im0.copy() if self.save_crop else self.im0  # for save_crop
                self.annotator = Annotator(self.im0, line_width=self.line_thickness, example=str(self.names))
                if len(self.det):
                    self.det[:, :4] = scale_boxes(self.im.shape[2:], self.det[:, :4], self.im0.shape).round()
                    
                    # Print results
                    for self.c in self.det[:, -1].unique():
                        self.n = (self.det[:, -1] == self.c).sum()  # detections per class
                        self.s += f"{self.n} {self.names[int(self.c)]}{'s' * (self.n > 1)}, "  # add to string

                    # Write results
                    for *self.xyxy, self.conf, cls in reversed(self.det):
                        self.listPositionX.append(self.xyxy[0].cpu().numpy())
                        self.listPositionX.append(self.xyxy[2].cpu().numpy())
                        self.listPositionY.append(self.xyxy[1].cpu().numpy())
                        self.listPositionY.append(self.xyxy[3].cpu().numpy())
                        if self.save_txt:  # Write to file
                            self.xywh = (xyxy2xywh(torch.tensor(self.xyxy).view(1, 4)) / self.gn).view(-1).tolist()  # normalized xywh
                            self.line = (cls, *self.xywh, self.conf) if self.save_conf else (cls, *self.xywh)  # label format
                            with open(f'{self.txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(self.line)).rstrip() % self.line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            obj = {}
                            c = int(cls)  # integer class
                            self.label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {self.conf:.2f}')
                            # self.annotator.box_label(self.xyxy, self.label, color=colors(c, True))
                            x = int(self.xyxy[0].cpu().numpy())
                            y = int(self.xyxy[1].cpu().numpy())
                            w = int(self.xyxy[2].cpu().numpy()) - x
                            h = int(self.xyxy[3].cpu().numpy()) - y
                            
                            # if self.names[c] == "bottle":
                            obj['name'] = self.names[c]
                            obj['confidence'] = float(self.conf)
                            obj['position'] = {
                                'x':x,
                                'y':y,
                                'w':w,
                                'h':h,
                            }

                            self.listObject.append(obj)
                        # if self.save_crop:
                        #     save_one_box(self.xyxy, self.imc, file=self.save_dir / 'crops' / self.names[c] / f'{self.p.stem}.jpg', BGR=True)

                # Stream results
                self.im0 = self.annotator.result()
                # cv2.imwrite("res_{}".format(self.p.name), self.im0)
                if self.view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)

        return self.listObject
