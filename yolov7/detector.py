import cv2
import numpy as np
import torch
from numpy import random

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.torch_utils import select_device


class YoloV7Detector:
    def __init__(self, model_name="yolov7.pt", img_size=640):
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(model_name, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.labels = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.labels]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(
                next(self.model.parameters())))  # run once

    def detect(self, image, thresh=0.25, iou_thres=0.45, classes=None, agnostic=True):
        # Padded resize
        img = letterbox(image, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, thresh, iou_thres=iou_thres, classes=classes, agnostic=agnostic)
        det = pred[0]
        boxes = []
        confidences = []
        class_ids = []
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                t, l, b, r = np.array(xyxy).astype(int)
                boxes.append([t, l, b, r])
                confidences.append(float(conf))
                class_ids.append(int(cls))
        return boxes, class_ids, confidences


if __name__ == '__main__':
    yolo = YoloV7Detector("yolov7.pt")
    images = ["inference/images/bus.jpg", "inference/images/horses.jpg", "inference/images/image1.jpg",
              "inference/images/image2.jpg"]
    for image in images:
        img = cv2.imread(image)
        boxes, class_ids, confidences = yolo.detect(img)
        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[3], box[1] + box[2]), (0, 255, 0), 2)
            text = str(yolo.labels[cls_id]) + ": {:.2f}".format(conf)
            cv2.putText(img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("image", img)
        cv2.waitKey(0)
